"""
Spanish License Plate Recognition Utilities
Optimized for Spanish plate formats with GPU acceleration
"""
import easyocr
import PIL
import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict
import re
import torch

PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

# Initialize EasyOCR reader with GPU if available
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

# Spanish license plate specific dictionaries
# Spanish plates: format ####-LLL or LL-#### (7 chars with hyphen)
# Letters used: B, C, D, F, G, H, J, K, L, M, N, P, R, S, T, V, W, X, Y, Z
# No vowels (A, E, I, O, U) and no Q in standard plates

# Character conversion for Spanish plates
spanish_char_to_int = {
    'O': '0', 'Q': '0',  # O and Q often confused with 0
    'I': '1', 'L': '1',  # I and L often confused with 1
    'Z': '2', 
    'A': '4',
    'S': '5',
    'G': '6',
    'T': '7',
    'B': '8',
    'D': '0',  # D can look like 0
}

spanish_int_to_char = {
    '0': 'O',
    '1': 'I',
    '2': 'Z',
    '4': 'A',
    '5': 'S',
    '6': 'G',
    '7': 'T',
    '8': 'B',
    '9': 'P',  # 9 can look like P
}

# Spanish plate validation patterns
SPANISH_PLATE_PATTERNS = [
    # Current format (2000-present): NNNN-LLL
    (r'^\d{4}[-\s]?[BCDFGHJKLMNPRSTVWXYZ]{3}$', 'current_format'),
    # Old format (pre-2000): LL-NNNN  
    (r'^[BCDFGHJKLMNPRSTVWXYZ]{2}[-\s]?\d{4}$', 'old_format'),
    # European format with E
    (r'^E[-\s]?\d{4}[-\s]?[BCDFGHJKLMNPRSTVWXYZ]{2}$', 'european_format'),
    # Without hyphen (common OCR output)
    (r'^\d{4}[BCDFGHJKLMNPRSTVWXYZ]{3}$', 'current_no_hyphen'),
    (r'^[BCDFGHJKLMNPRSTVWXYZ]{2}\d{4}$', 'old_no_hyphen'),
    # Partial matches (for testing)
    (r'^\d{3}[BCDFGHJKLMNPRSTVWXYZ]{3}$', 'partial_current'),
    (r'^[BCDFGHJKLMNPRSTVWXYZ]{2}\d{3}$', 'partial_old'),
]

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

def preprocess_spanish_plate(img: np.ndarray) -> np.ndarray:
    """
    Specialized preprocessing for Spanish license plates.
    Spanish plates have white background with black text and blue EU strip.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    
    # Enhance contrast for black text on white background
    # Spanish plates are high contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # Invert if needed (ensure dark text on light background)
    mean_intensity = np.mean(blurred)
    if mean_intensity < 127:
        blurred = cv2.bitwise_not(blurred)
    
    # Adaptive thresholding for Spanish plates
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Remove blue EU strip area (left side) if present
    height, width = thresh.shape
    if width > height * 1.5:  # Wide plate likely has EU strip
        strip_width = int(width * 0.15)  # EU strip is about 15% of width
        thresh[:, :strip_width] = 255  # Make EU strip area white
    
    # Morphological operations to clean characters
    kernel = np.ones((2, 1), np.uint8)  # Vertical kernel for character cleaning
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    return cleaned

def clean_spanish_plate_text(text: str) -> str:
    """
    Clean and normalize Spanish license plate text.
    IMPORTANT: For Spanish plates, we should NOT convert digits to letters
    because Spanish plates have specific digit positions.
    """
    # Convert to uppercase and remove spaces/hyphens
    text = text.upper().replace(' ', '').replace('-', '')
    
    # Remove invalid characters for Spanish plates
    # Spanish plates only use: 0-9 and B,C,D,F,G,H,J,K,L,M,N,P,R,S,T,V,W,X,Y,Z
    valid_chars = set('0123456789BCDFGHJKLMNPRSTVWXYZ')
    cleaned = ''.join([c for c in text if c in valid_chars])
    
    # Only convert ambiguous characters that are commonly misread
    # But preserve digits as digits for Spanish plates!
    corrections = {
        'O': '0',  # Letter O is often 0 in plates
        'Q': '0',  # Q is not used in Spanish plates, likely 0
        'I': '1',  # I is often 1
        'L': '1',  # L can be confused with 1
        'Z': '2',  # Z is 2
        'A': '4',  # A is 4
        'S': '5',  # S is 5
        'G': '6',  # G is 6
        'T': '7',  # T is 7
        'B': '8',  # B is 8
        'P': '9',  # P is 9
        'U': 'V',  # U is not used, likely V
        'E': 'F',  # E is not used, likely F
    }
    
    result = ''
    for char in cleaned:
        if char in corrections:
            result += corrections[char]
        else:
            result += char
    
    return result

def validate_spanish_plate(text: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate Spanish license plate format.
    Returns (is_valid, formatted_text, format_type)
    """
    if not text or len(text) < 4:
        return False, None, None
    
    cleaned = clean_spanish_plate_text(text)
    
    # Check all Spanish plate patterns
    for pattern, format_type in SPANISH_PLATE_PATTERNS:
        if re.match(pattern, cleaned):
            # Format with hyphen for display
            if format_type in ['current_format', 'current_no_hyphen', 'partial_current']:
                if len(cleaned) >= 4:
                    # Try to format as ####-LLL
                    numbers = ''.join([c for c in cleaned if c.isdigit()])
                    letters = ''.join([c for c in cleaned if c.isalpha()])
                    if len(numbers) >= 3 and len(letters) >= 2:
                        formatted = f"{numbers[:4]}-{letters[:3]}"
                    else:
                        formatted = cleaned
                else:
                    formatted = cleaned
            elif format_type in ['old_format', 'old_no_hyphen', 'partial_old']:
                if len(cleaned) >= 4:
                    # Try to format as LL-####
                    letters = ''.join([c for c in cleaned if c.isalpha()])
                    numbers = ''.join([c for c in cleaned if c.isdigit()])
                    if len(letters) >= 2 and len(numbers) >= 3:
                        formatted = f"{letters[:2]}-{numbers[:4]}"
                    else:
                        formatted = cleaned
                else:
                    formatted = cleaned
            elif format_type == 'european_format':
                formatted = cleaned
            else:
                formatted = cleaned
            
            return True, formatted, format_type
    
    # If no pattern matches but looks like a Spanish plate (4-7 alphanumeric chars)
    if 4 <= len(cleaned) <= 7 and cleaned.isalnum():
        # Try to infer format
        letters = ''.join([c for c in cleaned if c.isalpha()])
        numbers = ''.join([c for c in cleaned if c.isdigit()])
        
        if len(letters) >= 2 and len(numbers) >= 2:
            if cleaned[:2].isalpha() and cleaned[2:].isdigit():
                return True, f"{letters[:2]}-{numbers[:4]}", "inferred_old"
            elif cleaned[:4].isdigit() and cleaned[4:].isalpha():
                return True, f"{numbers[:4]}-{letters[:3]}", "inferred_current"
            else:
                return True, cleaned, "mixed_spanish"
    
    return False, None, None

def format_spanish_plate(text: str) -> str:
    """
    Format Spanish license plate with proper hyphenation.
    """
    cleaned = clean_spanish_plate_text(text)
    
    if len(cleaned) == 7:
        # Current format: 1234-ABC
        if cleaned[:4].isdigit() and cleaned[4:].isalpha():
            return f"{cleaned[:4]}-{cleaned[4:]}"
        # Old format: AB-1234
        elif cleaned[:2].isalpha() and cleaned[2:].isdigit():
            return f"{cleaned[:2]}-{cleaned[2:]}"
    
    elif len(cleaned) == 6:
        # Could be old format without leading zero
        if cleaned[:2].isalpha() and cleaned[2:].isdigit():
            return f"{cleaned[:2]}-{cleaned[2:]}"
    
    return cleaned

def read_spanish_license_plate(img: np.ndarray, use_preprocessing: bool = True) -> Tuple[str, float]:
    """
    Read Spanish license plate with specialized processing.
    """
    # Preprocess image for Spanish plates
    if use_preprocessing:
        processed_img = preprocess_spanish_plate(img)
    else:
        if len(img.shape) == 3:
            processed_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            processed_img = img
    
    # Read text with EasyOCR (using GPU if available)
    detections = reader.readtext(processed_img)
    
    best_text = ""
    best_score = 0.0
    best_format = ""
    
    for bbox, text, score in detections:
        # Validate as Spanish plate
        is_valid, formatted_text, format_type = validate_spanish_plate(text)
        
        if is_valid and score > best_score:
            best_text = formatted_text
            best_score = score
            best_format = format_type
    
    # If no valid plate found, try with the highest confidence detection
    if not best_text and detections:
        bbox, text, score = max(detections, key=lambda x: x[2])
        cleaned = clean_spanish_plate_text(text)
        
        # If cleaned text looks reasonable, use it
        if 4 <= len(cleaned) <= 8:
            best_text = format_spanish_plate(cleaned)
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

def test_spanish_plate_recognition():
    """Test function for Spanish plate recognition."""
    test_plates = [
        "1234ABC", "1234-ABC", "AB1234", "AB-1234",
        "E1234AB", "E-1234-AB", "5678XYZ", "CD-9876",
        "1234AB", "AB123", "123ABC",  # Edge cases
    ]
    
    print("Testing Spanish plate validation:")
    print("-" * 50)
    
    for plate in test_plates:
        is_valid, formatted, format_type = validate_spanish_plate(plate)
        status = "✓" if is_valid else "✗"
        print(f"{status} '{plate}' -> {formatted} ({format_type})")