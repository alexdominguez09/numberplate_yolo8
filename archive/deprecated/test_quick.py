"""
Quick test to process only first 10 frames of the video.
"""
from ultralytics import YOLO
import cv2 as cv
import numpy as np
import time

from sort.sort import *
from utils import *

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

# read frames - only first 10 for quick test
frame_nmb = 0
max_frames = 10
start_time = time.time()

print(f"Processing first {max_frames} frames...")
for frame_nmb in range(max_frames):
    ret, frame = cap.read()
    if not ret:
        break
    
    print(f"\nFrame {frame_nmb}/{max_frames}")
    results[frame_nmb] = {}
    
    # detect vehicles
    vehicles = model(frame, verbose=False)[0]
    vehicles_ = []
    for vehicle in vehicles.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = vehicle
        if int(class_id) in vehicle_ids:
            vehicles_.append([x1, y1, x2, y2, score])
    
    print(f"  Vehicles detected: {len(vehicles_)}")
    
    # track vehicles
    tracking_ids = mot_tracker.update(np.asarray(vehicles_))
    print(f"  Vehicles tracked: {len(tracking_ids)}")

    # detect plates
    plates = plate_detector_model(frame, verbose=False)[0]
    plates_data = plates.boxes.data.tolist()
    print(f"  Plates detected: {len(plates_data)}")

    for plate in plates_data:
        x1, y1, x2, y2, score, class_id = plate

        # map plate -> car
        x1car, y1car, x2car, y2car, car_id = map_car(plate, tracking_ids)

        if car_id != -1:
            # crop plate
            cropped_plate = frame[int(y1): int(y2), int(x1): int(x2), :]

            # process plate
            cropped_plate_gray = cv.cvtColor(cropped_plate, cv.COLOR_BGR2GRAY)
            _, plate_thresholded = cv.threshold(cropped_plate_gray, 64, 255, cv.THRESH_BINARY_INV)

            # read the number plate
            result = read_license_plate(plate_thresholded)
            if result is not None:
                license_number, license_number_score = result
            else:
                license_number, license_number_score = "-1", "-1"

            if license_number != -1 and license_number is not None:
                results[frame_nmb][car_id] = {'car': {'bbox': [x1car, y1car, x2car, y2car]},
                                              'plate': {'bbox': [x1, y1, x2, y2],
                                                        'text': license_number,
                                                        'bbox_score': score,
                                                        'text_score': license_number_score}}
                try:
                    score_float = float(license_number_score)
                    print(f"  ✓ Plate detected: {license_number} (confidence: {score_float:.2f})")
                except (ValueError, TypeError):
                    print(f"  ✓ Plate detected: {license_number} (score: {license_number_score})")

# Release video capture
cap.release()

# write results
output_csv = './test_quick.csv'
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