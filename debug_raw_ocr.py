"""
Debug raw OCR output without format validation.
"""
import cv2 as cv
import numpy as np
from utils import reader  # Import the EasyOCR reader directly

# Test with a sample plate image
print("Testing OCR directly...")

# Create a simple test image with text
test_img = np.zeros((50, 200), dtype=np.uint8)
cv.putText(test_img, "AB12CDE", (10, 35), cv.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

# Test UK format plate
print("\n1. Testing UK format plate 'AB12CDE':")
detections = reader.readtext(test_img)
for bbox, text, score in detections:
    print(f"  OCR read: '{text}' (score: {score:.3f})")
    text_clean = text.upper().replace(' ', '')
    print(f"  Cleaned: '{text_clean}'")
    from utils import check_license_plate_format
    is_valid = check_license_plate_format(text_clean)
    print(f"  Valid UK plate: {is_valid}")

# Test what the actual video plate might look like
print("\n2. Testing what OCR might be seeing from video...")
# Let's check what read_license_plate actually does
from utils import read_license_plate

result = read_license_plate(test_img)
print(f"  read_license_plate result: {result}")

# The issue might be that plates in the video are not UK format
print("\n3. Testing non-UK format plates:")
test_cases = ["ABC123", "123ABC", "AB-1234", "AB 1234"]
for test_text in test_cases:
    test_img = np.zeros((50, 200), dtype=np.uint8)
    cv.putText(test_img, test_text, (10, 35), cv.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    
    detections = reader.readtext(test_img)
    if detections:
        text = detections[0][1]
        text_clean = text.upper().replace(' ', '')
        from utils import check_license_plate_format
        is_valid = check_license_plate_format(text_clean)
        print(f"  '{test_text}' -> OCR: '{text}' -> Cleaned: '{text_clean}' -> Valid UK: {is_valid}")

print("\nThe system expects UK license plates (7 characters like 'AB12CDE').")
print("If your video has different plate formats, you may need to adjust the validation logic.")