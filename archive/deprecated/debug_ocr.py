"""
Debug OCR to see what's being detected.
"""
from ultralytics import YOLO
import cv2 as cv
import numpy as np
import time
import os

from sort.sort import *
from utils import *

# load models
print("Loading YOLO models...")
model = YOLO('yolov8n.pt')
plate_detector_model = YOLO('./models/license_plate_detector.pt')

# load video
video_path = '/home/alex/Downloads/video_carplates1.mkv'
print(f"Loading video: {video_path}")
cap = cv.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file: {video_path}")
    exit(1)

# Create debug directory
debug_dir = './debug_plates'
os.makedirs(debug_dir, exist_ok=True)

# Process a few frames
for frame_nmb in range(5):
    ret, frame = cap.read()
    if not ret:
        break
    
    print(f"\n=== Frame {frame_nmb} ===")
    
    # detect plates
    plates = plate_detector_model(frame, verbose=False)[0]
    plates_data = plates.boxes.data.tolist()
    print(f"Plates detected: {len(plates_data)}")
    
    for i, plate in enumerate(plates_data):
        x1, y1, x2, y2, score, class_id = plate
        print(f"  Plate {i}: bbox [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}], score: {score:.3f}")
        
        # crop plate
        cropped_plate = frame[int(y1): int(y2), int(x1): int(x2), :]
        
        if cropped_plate.size == 0:
            print("    Warning: Empty crop!")
            continue
            
        # Save original crop
        orig_path = f'{debug_dir}/frame{frame_nmb}_plate{i}_orig.jpg'
        cv.imwrite(orig_path, cropped_plate)
        
        # process plate with different thresholds
        cropped_plate_gray = cv.cvtColor(cropped_plate, cv.COLOR_BGR2GRAY)
        
        # Try different threshold values
        for thresh_val in [64, 96, 128, 160]:
            _, plate_thresholded = cv.threshold(cropped_plate_gray, thresh_val, 255, cv.THRESH_BINARY_INV)
            
            # Save thresholded image
            thresh_path = f'{debug_dir}/frame{frame_nmb}_plate{i}_thresh{thresh_val}.jpg'
            cv.imwrite(thresh_path, plate_thresholded)
            
            # Also try without inversion
            _, plate_normal = cv.threshold(cropped_plate_gray, thresh_val, 255, cv.THRESH_BINARY)
            normal_path = f'{debug_dir}/frame{frame_nmb}_plate{i}_normal{thresh_val}.jpg'
            cv.imwrite(normal_path, plate_normal)
            
            # read the number plate
            result = read_license_plate(plate_thresholded)
            if result is not None:
                license_number, license_number_score = result
                print(f"    Thresh {thresh_val} (inverted): {license_number} (score: {license_number_score})")
            else:
                print(f"    Thresh {thresh_val} (inverted): No valid plate")
                
            # Try normal threshold
            result_normal = read_license_plate(plate_normal)
            if result_normal is not None:
                license_number, license_number_score = result_normal
                print(f"    Thresh {thresh_val} (normal): {license_number} (score: {license_number_score})")
            else:
                print(f"    Thresh {thresh_val} (normal): No valid plate")

cap.release()
print(f"\nDebug images saved to: {debug_dir}")
print("Check the images to see what the OCR is seeing.")