"""
Fixed Spanish License Plate Recognition Utilities
With proper character handling for Spanish plates.
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

# Spanish license plate specific handling
# Spanish plates have specific formats:
# Current (2000+): ####-LLL (e.g., 1234-ABC)
# Old (pre-2000): LL-#### (e.g., AB-1234)
# European: E-####-LL (e.g., E-1234-AB)

# Character sets for Spanish plates
SPANISH_LETTERS = set('BCDFGHJKLMNPRSTVWXYZ')  # No vowels AEIOU, no Q
SPANISH_DIGITS = set('0123456789')

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
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # Ensure dark text on light background (Spanish plates are white background)
    mean_intensity = np.mean(blurred)
    if mean_intensity < 127:
        blurred = cv2.bitwise_not(blurred)
    
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Remove potential EU strip (blue area on left)
    height, width = thresh.shape
    if width > height * 1.5:
        strip_width = int(width * 0.15)
        thresh[:, :strip_width] = 255
    
    # Clean up characters
    kernel = np.ones((2, 1), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    return cleaned

def clean_spanish_text_simple(text: str) -> str:
    """
    Simple cleaning for Spanish plate text.
    Preserves digits as digits, letters as letters.
    """
    # Convert to uppercase, remove spaces and hyphens
    text = text.upper().replace(' ', '').replace('-', '')
    
    # Keep only valid Spanish plate characters
    valid_chars = SPANISH_LETTERS.union(SPANISH_DIGITS)
    cleaned = ''.join([c for c in text if c in valid_chars])
    
    return cleaned

def clean_spanish_text_intelligent(text: str) -> str:
    """
    Intelligent cleaning with context-aware corrections.
    For Spanish plates, we need to be careful about digit/letter positions.
    """
    text = text.upper().replace(' ', '').replace('-', '')
    
    # First pass: remove invalid characters
    valid_chars = SPANISH_LETTERS.union(SPANISH_DIGITS)
    cleaned = ''.join([c for c in text if c in valid_chars])
    
    if len(cleaned) < 4:
        return cleaned
    
    # Try to determine if it's current format (####LLL) or old format (LL####)
    # Count digits and letters
    digit_count = sum(1 for c in cleaned if c in SPANISH_DIGITS)
    letter_count = sum(1 for c in cleaned if c in SPANISH_LETTERS)
    
    # Apply corrections based on likely format
    result = ''
    for i, char in enumerate(cleaned):
        # Common OCR errors and their corrections
        if char == 'O':
            # O is often 0, but could be letter in letter position
            if i < 4 and digit_count > letter_count:
                result += '0'  # Likely digit in current format
            elif i >= 4 and letter_count > digit_count:
                result += '0'  # Should be letter but OCR misread
            else:
                result += '0'  # Default to 0
        elif char == 'I':
            # I is often 1
            result += '1'
        elif char == 'Z':
            # Z is 2
            result += '2'
        elif char == 'A':
            # A is 4
            result += '4'
        elif char == 'S':
            # S is 5
            result += '5'
        elif char == 'G':
            # G is 6
            result += '6'
        elif char == 'T':
            # T is 7
            result += '7'
        elif char == 'B':
            # B is 8
            result += '8'
        elif char == 'P':
            # P is 9
            result += '9'
        elif char == 'Q':
            # Q not used, likely 0
            result += '0'
        elif char == 'U':
            # U not used, likely V
            result += 'V'
        elif char == 'E':
            # E not used, likely F
            result += 'F'
        else:
            result += char
    
    return result

def validate_spanish_plate_flexible(text: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Flexible validation for Spanish plates.
    Accepts various formats and attempts to correct them.
    """
    if not text or len(text) < 4:
        return False, None, None
    
    # Clean the text
    cleaned = clean_spanish_text_simple(text)
    
    # Check if it looks like a Spanish plate
    if len(cleaned) < 4:
        return False, None, None
    
    # Count digits and letters
    digits = ''.join([c for c in cleaned if c in SPANISH_DIGITS])
    letters = ''.join([c for c in cleaned if c in SPANISH_LETTERS])
    
    # Determine likely format
    if len(digits) >= 3 and len(letters) >= 2:
        # Could be current format (####-LLL)
        if len(digits) >= 4:
            plate_num = digits[:4]
            plate_letters = letters[:3]
            formatted = f"{plate_num}-{plate_letters}"
            return True, formatted, "current_format"
        elif len(digits) >= 3 and len(letters) >= 3:
            # Partial current format
            formatted = f"{digits[:3]}-{letters[:3]}"
            return True, formatted, "partial_current"
    
    if len(letters) >= 2 and len(digits) >= 3:
        # Could be old format (LL-####)
        if len(letters) >= 2 and len(digits) >= 4:
            formatted = f"{letters[:2]}-{digits[:4]}"
            return True, formatted, "old_format"
        elif len(letters) >= 2 and len(digits) >= 3:
            formatted = f"{letters[:2]}-{digits[:3]}"
            return True, formatted, "partial_old"
    
    # If we have a reasonable mix, accept it
    if len(cleaned) >= 5 and (len(digits) >= 2 and len(letters) >= 2):
        return True, cleaned, "mixed_format"
    
    return False, None, None

def format_spanish_plate_nicely(text: str) -> str:
    """
    Format Spanish plate nicely with hyphen.
    """
    cleaned = clean_spanish_text_simple(text)
    
    if len(cleaned) >= 6:
        # Try current format first
        digits = ''.join([c for c in cleaned if c in SPANISH_DIGITS])
        letters = ''.join([c for c in cleaned if c in SPANISH_LETTERS])
        
        if len(digits) >= 4 and len(letters) >= 3:
            return f"{digits[:4]}-{letters[:3]}"
        elif len(letters) >= 2 and len(digits) >= 4:
            return f"{letters[:2]}-{digits[:4]}"
        elif len(cleaned) == 7:
            # Guess format based on position
            if cleaned[:4].isdigit() and cleaned[4:].isalpha():
                return f"{cleaned[:4]}-{cleaned[4:]}"
            elif cleaned[:2].isalpha() and cleaned[2:].isdigit():
                return f"{cleaned[:2]}-{cleaned[2:]}"
    
    return cleaned

def read_spanish_license_plate_optimized(img: np.ndarray) -> Tuple[str, float]:
    """
    Optimized Spanish license plate reading.
    """
    # Preprocess for Spanish plates
    processed_img = preprocess_spanish_plate(img)
    
    # Read with EasyOCR
    detections = reader.readtext(processed_img)
    
    best_text = ""
    best_score = 0.0
    
    for bbox, text, score in detections:
        # Validate as Spanish plate
        is_valid, formatted_text, format_type = validate_spanish_plate_flexible(text)
        
        if is_valid and score > best_score:
            best_text = formatted_text
            best_score = score
    
    # If no valid plate found, try with highest confidence
    if not best_text and detections:
        bbox, text, score = max(detections, key=lambda x: x[2])
        cleaned = clean_spanish_text_simple(text)
        
        if len(cleaned) >= 4:
            best_text = format_spanish_plate_nicely(cleaned)
            best_score = score * 0.5  # Reduce confidence for unvalidated plates
    
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

def test_spanish_validation():
    """Test function."""
    test_cases = [
        "1234ABC", "1234-ABC", "AB1234", "AB-1234",
        "5678XYZ", "CD9876", "1234AB", "AB123",
        "O123ABC", "1234OBC", "ABI234",
        "BSZX", "1234-BCD", "AB-5678"
    ]
    
    print("Testing Spanish plate validation:")
    print("-" * 50)
    
    for plate in test_cases:
        is_valid, formatted, format_type = validate_spanish_plate_flexible(plate)
        status = "✓" if is_valid else "✗"
        print(f"{status} '{plate}' -> {formatted} ({format_type})")

if __name__ == "__main__":
    test_spanish_validation()