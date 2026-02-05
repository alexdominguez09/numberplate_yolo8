"""
Simplified integration tests focusing on core functionality.
"""
import pytest
import tempfile
import os
import numpy as np
import pandas as pd


class TestCSVFormatCompatibility:
    """Test CSV format compatibility between components."""
    
    def test_write_csv_produces_correct_format(self):
        """Test that write_csv produces the expected CSV format."""
        from utils import write_csv
        
        test_results = {
            0: {
                1: {
                    'car': {'bbox': [100, 100, 200, 200]},
                    'plate': {
                        'bbox': [110, 110, 140, 140],
                        'text': 'AB12CDE',
                        'bbox_score': 0.95,
                        'text_score': 0.90
                    }
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_file = f.name
        
        try:
            write_csv(test_results, temp_file)
            
            # Read back and verify format
            df = pd.read_csv(temp_file)
            
            # Check columns (note: write_csv adds spaces after commas)
            expected_columns = ['frame_nmb', 'car_id', 'car_bbox', 'plate_bbox', 
                               'plate_bbox_score', 'license_nmb', 'license_nmb_score']
            
            # Actual columns have leading spaces
            actual_columns = list(df.columns)
            
            # Check column names (ignore spaces)
            assert len(actual_columns) == len(expected_columns)
            for i, (actual, expected) in enumerate(zip(actual_columns, expected_columns)):
                assert actual.strip() == expected, f"Column {i}: '{actual}' != '{expected}'"
            
            # Check data
            assert len(df) == 1
            assert df.iloc[0]['frame_nmb'] == 0
            assert df.iloc[0][' car_id'] == 1  # Note: space before car_id
            assert 'AB12CDE' in df.iloc[0][' license_nmb']  # Note: space before license_nmb
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_bbox_parsing_compatibility(self):
        """Test that bbox formats are parsed consistently."""
        # Test the parsing logic used in interpolate_data.py
        bbox_str = ' [100 200 300 400]'  # Note leading space as in CSV
        
        # interpolate_data.py uses: row['car_bbox'][2:-1].split()
        parsed_interpolate = list(map(float, bbox_str[2:-1].split()))
        assert parsed_interpolate == [100.0, 200.0, 300.0, 400.0]
        
        # visualize_data.py uses: ast.literal_eval(bbox.replace(' ', ','))
        import ast
        # Need to strip leading space first
        parsed_visualize = ast.literal_eval(bbox_str.strip().replace(' ', ','))
        assert parsed_visualize == [100, 200, 300, 400]
        
        # Both should produce compatible integer results
        assert [int(x) for x in parsed_interpolate] == parsed_visualize


class TestPipelineLogic:
    """Test core pipeline logic without external dependencies."""
    
    def test_vehicle_filtering_logic(self):
        """Test the vehicle filtering logic from main.py."""
        vehicle_ids = [2, 3, 5, 7]  # From main.py
        
        test_vehicles = [
            [100, 100, 200, 200, 0.95, 2],  # car - should be included
            [300, 300, 400, 400, 0.90, 3],  # motorcycle - should be included
            [500, 500, 600, 600, 0.85, 5],  # bus - should be included
            [700, 700, 800, 800, 0.80, 7],  # truck - should be included
            [900, 900, 1000, 1000, 0.75, 1],  # bicycle - should NOT be included
            [1100, 1100, 1200, 1200, 0.70, 8],  # boat - should NOT be included
        ]
        
        filtered_vehicles = []
        for vehicle in test_vehicles:
            x1, y1, x2, y2, score, class_id = vehicle
            if int(class_id) in vehicle_ids:
                filtered_vehicles.append([x1, y1, x2, y2, score])
        
        # Should have 4 vehicles (classes 2, 3, 5, 7)
        assert len(filtered_vehicles) == 4
        
        # Verify the filtered format
        for i, vehicle in enumerate(filtered_vehicles):
            assert len(vehicle) == 5  # x1, y1, x2, y2, score
            assert vehicle[4] == test_vehicles[i][4]  # Score should match
    
    def test_plate_to_vehicle_mapping_logic(self):
        """Test the plate to vehicle mapping logic."""
        from utils import map_car
        
        # Test case 1: Plate inside vehicle
        plate = [110, 120, 140, 150, 0.95, 0]
        tracking_ids = np.array([
            [100, 100, 200, 200, 1],  # car_id 1
            [300, 300, 400, 400, 2],  # car_id 2
        ])
        
        result = map_car(plate, tracking_ids)
        expected = np.array([100, 100, 200, 200, 1])
        np.testing.assert_array_equal(result, expected)
        
        # Test case 2: Plate outside all vehicles
        plate = [10, 10, 50, 50, 0.95, 0]
        tracking_ids = np.array([
            [100, 100, 200, 200, 1],
            [300, 300, 400, 400, 2],
        ])
        
        result = map_car(plate, tracking_ids)
        expected = np.array([-1, -1, -1, -1, -1])
        np.testing.assert_array_equal(result, expected)


class TestDataTransformation:
    """Test data transformation between pipeline stages."""
    
    def test_license_plate_validation(self):
        """Test license plate validation and formatting."""
        from utils import check_license_plate_format, format_license_number
        
        # Valid UK plates
        valid_plates = ["AB12CDE", "A123BCD", "AB51CDE"]
        for plate in valid_plates:
            assert check_license_plate_format(plate) == True
        
        # Invalid plates
        invalid_plates = ["AB12CD", "AB12CDEF", "2B12CDE"]
        for plate in invalid_plates:
            assert check_license_plate_format(plate) == False
        
        # Formatting test
        test_cases = [
            ("0B12CDE", "OB12CDE"),  # 0 -> O
            ("A112CDE", "AI12CDE"),  # 1 -> I
        ]
        
        for input_plate, expected in test_cases:
            result = format_license_number(input_plate)
            assert result == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])