import easyocr
import PIL
import cv2
import numpy as np
from typing import Tuple, Optional, List
import re

PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Character conversion dictionaries for ambiguous characters
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5',
                    'Z': '2',
                    'B': '8',
                    'Q': '0',
                    'D': '0'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S',
                    '2': 'Z',
                    '8': 'B'}

def write_csv(results, output_path):
    """Write results to CSV file."""
    with open(output_path, 'w') as f:
        f.write('{}, {}, {}, {}, {}, {}, {}\n'.format('frame_nmb', 'car_id', 'car_bbox',
                                                      'plate_bbox', 'plate_bbox_score', 'license_nmb',
                                                      'license_nmb_score'))

        for frame_nmb in results.keys():
            for car_id in results[frame_nmb].keys():
                if 'car' in results[frame_nmb][car_id].keys() and \
                        'plate' in results[frame_nmb][car_id].keys() and \
                        'text' in results[frame_nmb][car_id]['plate'].keys():
                    car_bbox = results[frame_nmb][car_id]['car']['bbox']
                    lp = results[frame_nmb][car_id]['plate']
                    f.write('{}, {}, {}, {}, {}, {}, {}\n'.format(frame_nmb,
                                                                  car_id,
                                                                  '[{} {} {} {}]'.format(
                                                                      car_bbox[0],
                                                                      car_bbox[1],
                                                                      car_bbox[2],
                                                                      car_bbox[3]),
                                                                  '[{} {} {} {}]'.format(
                                                                      lp['bbox'][0],
                                                                      lp['bbox'][1],
                                                                      lp['bbox'][2],
                                                                      lp['bbox'][3]),
                                                                  lp['bbox_score'],
                                                                  lp['text'],
                                                                  lp['text_score']))

def preprocess_plate_image(img: np.ndarray) -> np.ndarray:
    """
    Enhanced preprocessing for license plate images.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Morphological operations to clean up
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    return cleaned

def clean_ocr_text(text: str) -> str:
    """
    Clean OCR text by removing invalid characters and normalizing.
    """
    # Convert to uppercase and remove spaces
    text = text.upper().replace(' ', '')
    
    # Remove common OCR errors and invalid characters
    invalid_chars = r'[^A-Z0-9]'
    text = re.sub(invalid_chars, '', text)
    
    # Replace common OCR misreads
    replacements = {
        '0': 'O', '1': 'I', '2': 'Z', '3': 'J', '4': 'A',
        '5': 'S', '6': 'G', '7': 'T', '8': 'B', '9': 'P'
    }
    
    cleaned = ''
    for char in text:
        if char in replacements:
            cleaned += replacements[char]
        else:
            cleaned += char
    
    return cleaned

def validate_plate_format_flexible(text: str) -> Tuple[bool, Optional[str]]:
    """
    Flexible validation for various license plate formats.
    Returns (is_valid, formatted_text)
    """
    if not text:
        return False, None
    
    # Clean the text first
    cleaned = clean_ocr_text(text)
    
    # Check various common formats
    formats = [
        # UK format: LL##LLL (7 chars)
        (r'^[A-Z]{2}\d{2}[A-Z]{3}$', cleaned),
        # European format 1: ###LLL (6 chars)
        (r'^\d{3}[A-Z]{3}$', cleaned),
        # European format 2: LL#### (6 chars)
        (r'^[A-Z]{2}\d{4}$', cleaned),
        # European format 3: L####L (6 chars)
        (r'^[A-Z]\d{4}[A-Z]$', cleaned),
        # US/Canada format: ###LLL (6 chars)
        (r'^\d{3}[A-Z]{3}$', cleaned),
        # US/Canada format: LLL### (6 chars)
        (r'^[A-Z]{3}\d{3}$', cleaned),
        # Mixed format 1: L#L#L# (6 chars)
        (r'^[A-Z]\d[A-Z]\d[A-Z]\d$', cleaned),
        # Mixed format 2: #L#L#L (6 chars)
        (r'^\d[A-Z]\d[A-Z]\d[A-Z]$', cleaned),
        # Short format: ###LL (5 chars)
        (r'^\d{3}[A-Z]{2}$', cleaned),
        # Short format: LL### (5 chars)
        (r'^[A-Z]{2}\d{3}$', cleaned),
        # Long format: ###LLLL (7 chars)
        (r'^\d{3}[A-Z]{4}$', cleaned),
        # Long format: LLLL### (7 chars)
        (r'^[A-Z]{4}\d{3}$', cleaned),
        # All digits (common in some countries)
        (r'^\d{5,8}$', cleaned),
        # All letters (less common but possible)
        (r'^[A-Z]{5,8}$', cleaned),
    ]
    
    for pattern, test_text in formats:
        if re.match(pattern, test_text):
            return True, test_text
    
    # If no format matches but text looks reasonable (4-8 alphanumeric chars)
    if 4 <= len(cleaned) <= 8 and cleaned.isalnum():
        return True, cleaned
    
    return False, None

def validate_plate_format_strict(text: str) -> Tuple[bool, Optional[str]]:
    """
    Strict validation for UK license plate format.
    """
    if len(text) != 7:
        return False, None
    
    # Check UK format: LL##LLL
    pattern = r'^[A-Z]{2}\d{2}[A-Z]{3}$'
    if re.match(pattern, text):
        return True, text
    
    return False, None

def format_license_number_enhanced(text: str) -> str:
    """
    Enhanced formatting with intelligent character mapping.
    """
    if not text:
        return ""
    
    # Clean the text
    cleaned = clean_ocr_text(text)
    
    # Apply format-specific corrections
    if len(cleaned) == 7:
        # Try UK format correction
        formatted = ''
        for i, char in enumerate(cleaned):
            if i in [0, 1, 4, 5, 6]:  # Should be letters
                if char.isdigit() and char in dict_int_to_char:
                    formatted += dict_int_to_char[char]
                else:
                    formatted += char
            elif i in [2, 3]:  # Should be digits
                if char.isalpha() and char in dict_char_to_int:
                    formatted += dict_char_to_int[char]
                else:
                    formatted += char
            else:
                formatted += char
        return formatted
    else:
        # For other formats, just return cleaned text
        return cleaned

def read_license_plate_enhanced(img: np.ndarray, use_preprocessing: bool = True) -> Tuple[str, float]:
    """
    Enhanced license plate reading with better preprocessing and validation.
    """
    # Preprocess image if requested
    if use_preprocessing:
        processed_img = preprocess_plate_image(img)
    else:
        if len(img.shape) == 3:
            processed_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            processed_img = img
    
    # Read text with EasyOCR
    detections = reader.readtext(processed_img)
    
    best_text = ""
    best_score = 0.0
    
    for bbox, text, score in detections:
        # Try flexible validation first
        is_valid, formatted_text = validate_plate_format_flexible(text)
        
        if is_valid and score > best_score:
            best_text = formatted_text
            best_score = score
    
    # If no valid plate found with flexible validation, try with cleaned text
    if not best_text and detections:
        # Get the highest confidence detection
        bbox, text, score = max(detections, key=lambda x: x[2])
        cleaned = clean_ocr_text(text)
        
        # If cleaned text looks reasonable, use it
        if 4 <= len(cleaned) <= 8 and cleaned.isalnum():
            best_text = cleaned
            best_score = score
    
    if best_text:
        return best_text, best_score
    
    return "", 0.0

def map_car(plate, tracking_ids):
    """Map detected plate to tracked car."""
    x1, y1, x2, y2, score, class_id = plate
    
    found = False
    for j in range(len(tracking_ids)):
        x_car1, y_car1, x_car2, y_car2, car_id = tracking_ids[j]
        
        if x1 > x_car1 and y1 > y_car1 and x2 < x_car2 and y2 < y_car2:
            car_index = j
            found = True
            break
    
    if found:
        return tracking_ids[car_index]
    
    return -1, -1, -1, -1, -1

def get_car(license_plate, vehicle_track_ids):
    """Get car information for a license plate."""
    x1, y1, x2, y2, score, class_id = license_plate
    
    for vehicle_track_id in vehicle_track_ids:
        x_car1, y_car1, x_car2, y_car2, car_id = vehicle_track_id
        
        if x1 > x_car1 and y1 > y_car1 and x2 < x_car2 and y2 < y_car2:
            return vehicle_track_id
    
    return -1, -1, -1, -1, -1