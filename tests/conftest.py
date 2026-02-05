"""
Pytest configuration and fixtures for number plate recognition tests.
"""
import pytest
import tempfile
import os
import numpy as np
from unittest.mock import Mock, MagicMock


@pytest.fixture
def temp_csv_file():
    """Create a temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_path = f.name
    yield temp_path
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def sample_license_plate_image():
    """Create a sample license plate image for testing."""
    # Create a simple 100x50 grayscale image
    img = np.zeros((50, 100), dtype=np.uint8)
    # Add some "text" region (brighter area)
    img[20:30, 30:70] = 200
    return img


@pytest.fixture
def sample_vehicle_tracking_ids():
    """Create sample vehicle tracking IDs for testing."""
    return np.array([
        [100, 100, 200, 200, 1],  # car_id 1
        [300, 300, 400, 400, 2],  # car_id 2
        [500, 500, 600, 600, 3],  # car_id 3
    ])


@pytest.fixture
def sample_plate_detection():
    """Create a sample plate detection."""
    return [110, 120, 140, 150, 0.95, 0]  # x1, y1, x2, y2, score, class_id


@pytest.fixture
def sample_detection_results():
    """Create sample detection results matching the format from main.py."""
    return {
        0: {  # frame 0
            1: {  # car_id 1
                'car': {'bbox': [100, 100, 200, 200]},
                'plate': {
                    'bbox': [110, 110, 140, 140],
                    'text': 'AB12CDE',
                    'bbox_score': 0.95,
                    'text_score': 0.90
                }
            }
        },
        1: {  # frame 1
            1: {  # car_id 1
                'car': {'bbox': [105, 105, 205, 205]},
                'plate': {
                    'bbox': [115, 115, 145, 145],
                    'text': 'AB12CDE',
                    'bbox_score': 0.94,
                    'text_score': 0.91
                }
            },
            2: {  # car_id 2
                'car': {'bbox': [300, 300, 400, 400]},
                'plate': {
                    'bbox': [310, 310, 340, 340],
                    'text': 'CD34EFG',
                    'bbox_score': 0.88,
                    'text_score': 0.85
                }
            }
        }
    }


@pytest.fixture
def mock_easyocr_reader():
    """Create a mock EasyOCR reader."""
    mock_reader = Mock()
    
    def mock_readtext(img):
        # Simulate OCR detection based on image content
        height, width = img.shape[:2]
        
        # If image has bright region (simulating text), return a detection
        if np.max(img) > 100:
            return [
                ([(0, 0), (width, 0), (width, height), (0, height)], "AB12CDE", 0.95)
            ]
        else:
            return []
    
    mock_reader.readtext.side_effect = mock_readtext
    return mock_reader


@pytest.fixture
def valid_uk_license_plates():
    """Return a list of valid UK license plate formats."""
    return [
        "AB12CDE",  # Standard format
        "A123BCD",  # Older format
        "AB51CDE",  # With numbers in middle
        "0B12CDE",  # First char could be 0 (mapped from O)
        "A112CDE",  # Second char could be 1 (mapped from I)
    ]


@pytest.fixture
def invalid_license_plates():
    """Return a list of invalid license plate formats."""
    return [
        "AB12CD",     # Too short
        "AB12CDEF",   # Too long
        "2B12CDE",    # Invalid first character
        "ABABCDE",    # Invalid middle characters
        "AB12C1E",    # Invalid character in position 5
        "AB12CD ",    # Trailing space
        " AB12CDE",   # Leading space
        "AB12-CDE",   # Contains hyphen
    ]


@pytest.fixture
def plates_with_ambiguous_chars():
    """Return plates with ambiguous characters for testing formatting."""
    return [
        ("0B12CDE", "OB12CDE"),  # 0 -> O
        ("A112CDE", "A1I2CDE"),  # 1 -> I
        ("AB02CDE", "ABO2CDE"),  # 0 -> O
        ("AB13CDE", "AB1JCDE"),  # 3 -> J
        ("AB12C4E", "AB12CAE"),  # 4 -> A
        ("AB12CD5", "AB12CDS"),  # 5 -> S
        ("AB12C6E", "AB12CGE"),  # 6 -> G
    ]


@pytest.fixture
def mock_video_capture():
    """Create a mock video capture object."""
    mock_cap = Mock()
    
    # Create test frames
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame2 = np.ones((480, 640, 3), dtype=np.uint8) * 128
    frame3 = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    mock_cap.read.side_effect = [
        (True, frame1),
        (True, frame2),
        (True, frame3),
        (False, None)  # End of video
    ]
    
    # Mock video properties
    mock_cap.get.side_effect = lambda prop: {
        'CAP_PROP_FPS': 30.0,
        'CAP_PROP_FRAME_WIDTH': 640.0,
        'CAP_PROP_FRAME_HEIGHT': 480.0,
        'CAP_PROP_POS_FRAMES': 0.0,
    }.get(prop, 0.0)
    
    mock_cap.set.return_value = True
    
    return mock_cap


@pytest.fixture
def mock_yolo_model():
    """Create a mock YOLO model."""
    mock_model = Mock()
    
    # Mock prediction
    mock_prediction = Mock()
    mock_box = Mock()
    
    # Default detection (can be overridden in tests)
    mock_box.boxes.data.tolist.return_value = [
        [100, 100, 200, 200, 0.95, 2],  # car
    ]
    
    mock_prediction.__getitem__.return_value = mock_box
    mock_model.return_value = [mock_prediction]
    
    return mock_model


@pytest.fixture
def mock_sort_tracker():
    """Create a mock SORT tracker."""
    mock_tracker = Mock()
    
    # Default tracking result
    mock_tracker.update.return_value = np.array([
        [100, 100, 200, 200, 1],  # car_id 1
    ])
    
    return mock_tracker


@pytest.fixture
def sample_interpolation_data():
    """Create sample data for interpolation testing."""
    return [
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


@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Clean up temporary files after each test."""
    # Store list of temp files created during test
    temp_files = []
    
    yield
    
    # Cleanup
    for file_path in temp_files:
        if os.path.exists(file_path):
            try:
                os.unlink(file_path)
            except:
                pass  # Ignore cleanup errors