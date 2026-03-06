"""
Main script with flexible plate format validation.
"""
from ultralytics import YOLO
import cv2 as cv
import numpy as np
import time
import re

from sort.sort import *
from utils import *

def flexible_plate_validation(text):
    """
    More flexible plate validation that accepts various formats.
    Returns (is_valid, cleaned_text)
    """
    # Clean the text
    text = text.upper().replace(' ', '').replace('-', '').replace('.', '')
    
    # Original UK format check (from utils.py)
    if len(text) == 7:
        # Check if it matches UK format
        from utils import check_license_plate_format, dict_int_to_char
        if check_license_plate_format(text):
            from utils import format_license_number
            return True, format_license_number(text)
    
    # Accept other common formats
    # 1. 6 characters (e.g., ABC123)
    if len(text) == 6 and text.isalnum():
        return True, text
    
    # 2. 7-8 characters with mix of letters and numbers
    if 6 <= len(text) <= 8 and any(c.isdigit() for c in text) and any(c.isalpha() for c in text):
        return True, text
    
    # 3. European style (e.g., AB-123-CD)
    if 5 <= len(text) <= 10 and any(c.isdigit() for c in text) and any(c.isalpha() for c in text):
        return True, text
    
    return False, text

def read_license_plate_flexible(img):
    """
    Flexible version of read_license_plate that accepts more formats.
    """
    detection = reader.readtext(img)
    
    for det in detection:
        bbox, text, score = det
        text_clean = text.upper().replace(' ', '').replace('-', '').replace('.', '')
        
        # Try flexible validation
        is_valid, formatted_text = flexible_plate_validation(text)
        if is_valid:
            return formatted_text, score
        
        # Also try original UK format
        from utils import check_license_plate_format
        if check_license_plate_format(text_clean):
            from utils import format_license_number
            return format_license_number(text_clean), score
    
    return None, None

# load models
print("Loading YOLO models...")
model = YOLO('yolov8n.pt')
plate_detector_model = YOLO('./models/license_plate_detector.pt')

mot_tracker = Sort()

# load video
video_path = '/home/alex/Downloads/video_carplates1.mkv'
print(f"Loading video: {video_path}")
cap = cv.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file: {video_path}")
    exit(1)

# Get video properties
fps = cap.get(cv.CAP_PROP_FPS)
total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
print(f"Video info: {fps:.1f} FPS, {total_frames} frames")

vehicle_ids = [2, 3, 5, 7]  # car, motorcycle, bus, truck
results = {}

# read frames - process more frames to get better detection
frame_nmb = 0
max_frames = 50  # Process more frames
start_time = time.time()

print(f"Processing first {max_frames} frames with flexible validation...")
for frame_nmb in range(max_frames):
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_nmb % 10 == 0:
        print(f"Frame {frame_nmb}/{max_frames}")
    
    results[frame_nmb] = {}
    
    # detect vehicles
    vehicles = model(frame, verbose=False)[0]
    vehicles_ = []
    for vehicle in vehicles.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = vehicle
        if int(class_id) in vehicle_ids:
            vehicles_.append([x1, y1, x2, y2, score])
    
    # track vehicles
    tracking_ids = mot_tracker.update(np.asarray(vehicles_))

    # detect plates
    plates = plate_detector_model(frame, verbose=False)[0]
    plates_data = plates.boxes.data.tolist()

    for plate in plates_data:
        x1, y1, x2, y2, score, class_id = plate

        # map plate -> car
        x1car, y1car, x2car, y2car, car_id = map_car(plate, tracking_ids)

        if car_id != -1:
            # crop plate
            cropped_plate = frame[int(y1): int(y2), int(x1): int(x2), :]

            # process plate - try different processing
            cropped_plate_gray = cv.cvtColor(cropped_plate, cv.COLOR_BGR2GRAY)
            
            # Try multiple threshold values
            best_result = None
            best_score = 0
            
            for thresh_val in [64, 96, 128]:
                # Try inverted
                _, plate_thresholded = cv.threshold(cropped_plate_gray, thresh_val, 255, cv.THRESH_BINARY_INV)
                result = read_license_plate_flexible(plate_thresholded)
                
                if result[0] is not None:
                    license_number, license_number_score = result
                    try:
                        score_float = float(license_number_score)
                        if score_float > best_score:
                            best_result = (license_number, license_number_score)
                            best_score = score_float
                    except:
                        if best_result is None:
                            best_result = (license_number, license_number_score)
                
                # Try normal threshold
                _, plate_normal = cv.threshold(cropped_plate_gray, thresh_val, 255, cv.THRESH_BINARY)
                result = read_license_plate_flexible(plate_normal)
                
                if result[0] is not None:
                    license_number, license_number_score = result
                    try:
                        score_float = float(license_number_score)
                        if score_float > best_score:
                            best_result = (license_number, license_number_score)
                            best_score = score_float
                    except:
                        if best_result is None:
                            best_result = (license_number, license_number_score)
            
            if best_result is not None:
                license_number, license_number_score = best_result
                results[frame_nmb][car_id] = {'car': {'bbox': [x1car, y1car, x2car, y2car]},
                                              'plate': {'bbox': [x1, y1, x2, y2],
                                                        'text': license_number,
                                                        'bbox_score': score,
                                                        'text_score': license_number_score}}
                
                if frame_nmb % 10 == 0:  # Only print every 10 frames to avoid spam
                    try:
                        score_float = float(license_number_score)
                        print(f"  ✓ Plate: {license_number} (score: {score_float:.2f})")
                    except:
                        print(f"  ✓ Plate: {license_number}")

# Release video capture
cap.release()

# write results
output_csv = './test_flexible.csv'
print(f"\nWriting results to {output_csv}")
write_csv(results, output_csv)

# Print summary
total_time = time.time() - start_time
total_detections = sum(len(frame_results) for frame_results in results.values())
print(f"\nProcessing complete!")
print(f"  - Total time: {total_time:.1f} seconds")
print(f"  - Frames processed: {frame_nmb}")
print(f"  - Processing speed: {frame_nmb/total_time:.1f} fps")
print(f"  - Total detections: {total_detections}")
print(f"  - Results saved to: {output_csv}")

# Show all detections
if total_detections > 0:
    print("\nAll plate detections:")
    for frame_num in sorted(results.keys()):
        if results[frame_num]:
            for car_id, detection in results[frame_num].items():
                plate_text = detection['plate']['text']
                if plate_text not in ['-1', '0', '']:
                    bbox_score = detection['plate']['bbox_score']
                    text_score = detection['plate']['text_score']
                    print(f"  Frame {frame_num}, Car {car_id}: {plate_text} (bbox: {bbox_score:.2f}, text: {text_score})")