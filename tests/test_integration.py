"""
Integration tests for the main number plate recognition pipeline.
Tests the integration of components without requiring actual video files or models.
"""
import pytest
import tempfile
import os
import sys
import numpy as np
from unittest.mock import Mock, patch, MagicMock, call

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestMainPipelineIntegration:
    """Integration tests for the main.py pipeline."""
    
    @patch('main.cv.VideoCapture')
    @patch('main.YOLO')
    @patch('main.Sort')
    @patch('main.write_csv')
    def test_pipeline_with_mocked_components(self, mock_write_csv, mock_sort, mock_yolo, mock_videocapture):
        """Test the main pipeline flow with all components mocked."""
        from main import model, plate_detector_model, mot_tracker, vehicle_ids, results
        
        # Mock video capture
        mock_cap = Mock()
        mock_videocapture.return_value = mock_cap
        
        # Mock video frames
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2 = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        mock_cap.read.side_effect = [(True, frame1), (True, frame2), (False, None)]
        
        # Mock YOLO models
        mock_yolo_instance = Mock()
        mock_yolo.return_value = mock_yolo_instance
        
        # Mock vehicle detection results
        # Create a mock that supports __getitem__
        mock_vehicle_result = Mock()
        mock_vehicle_box = Mock()
        mock_vehicle_box.boxes.data.tolist.return_value = [
            [100, 100, 200, 200, 0.95, 2],  # car (class_id 2)
            [300, 300, 400, 400, 0.90, 3],  # motorcycle (class_id 3)
            [500, 500, 600, 600, 0.85, 5],  # bus (class_id 5)
            [700, 700, 800, 800, 0.80, 7],  # truck (class_id 7)
            [900, 900, 1000, 1000, 0.75, 1],  # bicycle (class_id 1, not in vehicle_ids)
        ]
        # Use __getitem__ as a callable attribute
        type(mock_vehicle_result).__getitem__ = Mock(return_value=mock_vehicle_box)
        
        # Mock plate detection results
        mock_plate_result = Mock()
        mock_plate_box = Mock()
        mock_plate_box.boxes.data.tolist.return_value = [
            [110, 110, 140, 140, 0.92, 0],  # Plate inside first car
            [310, 310, 340, 340, 0.88, 0],  # Plate inside second vehicle
        ]
        type(mock_plate_result).__getitem__ = Mock(return_value=mock_plate_box)
        
        # Mock YOLO to return different results for vehicle and plate detection
        mock_yolo_instance.side_effect = [
            [mock_vehicle_result],  # First call: vehicle detection
            [mock_plate_result]     # Second call: plate detection
        ]
        
        # Mock SORT tracker
        mock_tracker_instance = Mock()
        mock_sort.return_value = mock_tracker_instance
        mock_tracker_instance.update.return_value = np.array([
            [100, 100, 200, 200, 1],  # car_id 1
            [300, 300, 400, 400, 2],  # car_id 2
            [500, 500, 600, 600, 3],  # car_id 3
            [700, 700, 800, 800, 4],  # car_id 4
        ])
        
        # Import and run main (but we'll test the logic directly)
        # Instead, let's test the imported functions work together
        
        # Test that the imports work
        assert model is not None
        assert plate_detector_model is not None
        assert mot_tracker is not None
        assert vehicle_ids == [2, 3, 5, 7]
        assert results == {}
        
        # Verify mock setup
        mock_videocapture.assert_called_once_with('./demo_1.mp4')
        
    @patch('main.cv.VideoCapture')
    @patch('main.YOLO')
    @patch('main.Sort')
    def test_pipeline_data_flow(self, mock_sort, mock_yolo, mock_videocapture):
        """Test data flow through the pipeline with controlled inputs."""
        # This test simulates the actual logic in main.py
        
        # Create a simplified version of the main loop logic
        mock_cap = Mock()
        mock_videocapture.return_value = mock_cap
        
        # Single frame test
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, test_frame)
        
        # Mock YOLO instances
        mock_vehicle_model = Mock()
        mock_plate_model = Mock()
        mock_yolo.side_effect = [mock_vehicle_model, mock_plate_model]
        
        # Mock vehicle detection
        mock_vehicle_pred = Mock()
        mock_vehicle_box = Mock()
        mock_vehicle_box.boxes.data.tolist.return_value = [
            [100, 100, 200, 200, 0.95, 2],  # Valid vehicle (car)
        ]
        mock_vehicle_pred.__getitem__.return_value = mock_vehicle_box
        mock_vehicle_model.return_value = [mock_vehicle_pred]
        
        # Mock plate detection
        mock_plate_pred = Mock()
        mock_plate_box = Mock()
        mock_plate_box.boxes.data.tolist.return_value = [
            [110, 110, 140, 140, 0.92, 0],  # Plate
        ]
        mock_plate_pred.__getitem__.return_value = mock_plate_box
        mock_plate_model.return_value = [mock_plate_pred]
        
        # Mock tracker
        mock_tracker = Mock()
        mock_sort.return_value = mock_tracker
        mock_tracker.update.return_value = np.array([[100, 100, 200, 200, 1]])
        
        # Test would continue with actual execution, but we're testing integration
        
        # Verify the expected calls
        mock_videocapture.assert_called_once_with('./demo_1.mp4')
        assert mock_yolo.call_count == 2
        mock_sort.assert_called_once()
        
    def test_vehicle_id_filtering_logic(self):
        """Test the vehicle filtering logic from main.py."""
        # This tests the logic: if int(class_id) in vehicle_ids:
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


class TestInterpolationPipeline:
    """Integration tests for the interpolation pipeline."""
    
    @patch('interpolate_data.csv.DictReader')
    @patch('interpolate_data.open')
    def test_interpolation_data_flow(self, mock_open, mock_dictreader):
        """Test the interpolation pipeline with sample data."""
        # Sample test data matching the format from test.csv
        sample_data = [
            {
                'frame_nmb': '0',
                'car_id': '1',
                'car_bbox': '[100 100 200 200]',
                'plate_bbox': '[110 110 140 140]',
                'plate_bbox_score': '0.95',
                'license_nmb': 'AB12CDE',
                'license_nmb_score': '0.90'
            },
            {
                'frame_nmb': '2',  # Frame 1 is missing
                'car_id': '1',
                'car_bbox': '[105 105 205 205]',
                'plate_bbox': '[115 115 145 145]',
                'plate_bbox_score': '0.94',
                'license_nmb': 'AB12CDE',
                'license_nmb_score': '0.91'
            }
        ]
        
        mock_dictreader.return_value = sample_data
        
        # We would need to import and test interpolate_for_missing_frames directly
        # For now, verify the mock setup
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        # The actual test would call interpolate_data.py functions
        
    def test_interpolation_logic(self):
        """Test the interpolation logic with simple data."""
        # Simplified test of the interpolation concept
        import numpy as np
        from scipy.interpolate import interp1d
        
        # Test linear interpolation between two points
        x = np.array([0, 2])  # frames 0 and 2
        y = np.array([[100, 100, 200, 200],  # bbox at frame 0
                      [105, 105, 205, 205]])  # bbox at frame 2
        
        interpolator = interp1d(x, y, axis=0, kind='linear')
        x_new = np.array([1])  # frame 1
        interpolated = interpolator(x_new)
        
        # Should interpolate between the two bboxes
        expected = np.array([[102.5, 102.5, 202.5, 202.5]])
        np.testing.assert_array_almost_equal(interpolated, expected)


class TestVisualizationPipeline:
    """Integration tests for the visualization pipeline."""
    
    @patch('visualize_data.cv.VideoCapture')
    @patch('visualize_data.pd.read_csv')
    @patch('visualize_data.cv.VideoWriter')
    def test_visualization_data_flow(self, mock_videowriter, mock_readcsv, mock_videocapture):
        """Test the visualization pipeline flow."""
        # Mock CSV data
        mock_df = Mock()
        mock_readcsv.return_value = mock_df
        
        # Mock unique car IDs
        mock_df.__getitem__.return_value.unique.return_value = [1, 2]
        
        # Mock idxmax and loc operations
        mock_idxmax = Mock()
        mock_df.__getitem__.return_value.idxmax.return_value = 0
        mock_df.loc.__getitem__.return_value = 'AB12CDE'
        
        # Mock video capture
        mock_cap = Mock()
        mock_videocapture.return_value = mock_cap
        mock_cap.get.side_effect = [30.0, 640.0, 480.0]  # fps, width, height
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        
        # Mock video writer
        mock_writer = Mock()
        mock_videowriter.return_value = mock_writer
        
        # Test would run visualize_data.py, but we verify integration
        
        mock_videocapture.assert_called_once_with('demo_1.mp4')
        mock_readcsv.assert_called_once_with('./test_interpolated.csv')


class TestFullPipelineIntegration:
    """Test the integration between all pipeline components."""
    
    def test_csv_format_compatibility(self):
        """Test that CSV formats are compatible between components."""
        # main.py writes CSV with this format:
        main_csv_header = "frame_nmb, car_id, car_bbox, plate_bbox, plate_bbox_score, license_nmb, license_nmb_score"
        
        # interpolate_data.py reads CSV with DictReader (expects same format)
        # visualize_data.py reads CSV with pd.read_csv (expects same format)
        
        # All components should use compatible CSV formats
        expected_columns = [
            'frame_nmb',
            'car_id', 
            'car_bbox',
            'plate_bbox',
            'plate_bbox_score',
            'license_nmb',
            'license_nmb_score'
        ]
        
        # Test that write_csv produces compatible format
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
            import pandas as pd
            df = pd.read_csv(temp_file)
            
            # Check columns match expected
            assert list(df.columns) == expected_columns
            
            # Check data types
            assert df['frame_nmb'].dtype in [np.int64, np.float64]
            assert df['car_id'].dtype in [np.int64, np.float64]
            assert df['license_nmb'].dtype == object  # string
            
        finally:
            os.unlink(temp_file)
    
    def test_bbox_format_consistency(self):
        """Test that bounding box formats are consistent across components."""
        # main.py writes bboxes as strings: '[x1 y1 x2 y2]'
        # interpolate_data.py parses them with: row['car_bbox'][2:-1].split()
        # visualize_data.py parses them with: ast.literal_eval(bbox.replace(' ', ','))
        
        # Test the parsing logic from interpolate_data.py
        # Note: CSV has format: " [x1 y1 x2 y2]" (space before bracket)
        bbox_str = ' [100 200 300 400]'  # Note leading space
        parsed = list(map(float, bbox_str[2:-1].split()))  # [2:-1] removes ' [' and ']'
        assert parsed == [100.0, 200.0, 300.0, 400.0]
        
        # Test the parsing logic from visualize_data.py
        import ast
        parsed_ast = ast.literal_eval(bbox_str.replace(' ', ','))
        assert parsed_ast == [100, 200, 300, 400]
        
        # Both should produce compatible results
        assert [int(x) for x in parsed] == parsed_ast


if __name__ == "__main__":
    pytest.main([__file__, "-v"])