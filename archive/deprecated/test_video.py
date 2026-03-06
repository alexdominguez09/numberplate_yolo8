"""
Simple test script to run the number plate recognition on a video file.
"""
import os
import argparse
import cv2 as cv
import numpy as np
from ultralytics import YOLO
from sort.sort import *
from utils import *

def main(video_path, output_csv='./test_output.csv'):
    """Run number plate recognition on a video file."""
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found!")
        print("Available video files:")
        import glob
        for f in glob.glob("*.mp4") + glob.glob("*.avi") + glob.glob("*.mov"):
            print(f"  - {f}")
        return
    
    # Load models
    print("Loading YOLO models...")
    model = YOLO('yolov8n.pt')
    plate_detector_model = YOLO('./models/yolov8n_license_plate.pt')
    
    mot_tracker = Sort()
    vehicle_ids = [2, 3, 5, 7]  # car, motorcycle, bus, truck
    results = {}
    
    # Load video
    print(f"Processing video: {video_path}")
    cap = cv.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return
    
    # Get video properties
    fps = cap.get(cv.CAP_PROP_FPS)
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    print(f"Video info: {fps:.1f} FPS, {total_frames} frames")
    
    # Process frames
    frame_nmb = 0
    ret = True
    
    print("Processing frames... (Press Ctrl+C to stop early)")
    try:
        while ret:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_nmb % 30 == 0:  # Print progress every ~1 second at 30fps
                print(f"  Frame {frame_nmb}/{total_frames}")
            
            results[frame_nmb] = {}
            
            # Detect vehicles
            vehicles = model(frame)[0]
            vehicles_ = []
            for vehicle in vehicles.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = vehicle
                if int(class_id) in vehicle_ids:
                    vehicles_.append([x1, y1, x2, y2, score])
            
            # Track vehicles
            tracking_ids = mot_tracker.update(np.asarray(vehicles_))
            
            # Detect plates
            plates = plate_detector_model(frame)[0]
            
            for plate in plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = plate
                
                # Map plate to car
                x1car, y1car, x2car, y2car, car_id = map_car(plate, tracking_ids)
                
                if car_id != -1:
                    # Crop plate
                    cropped_plate = frame[int(y1): int(y2), int(x1): int(x2), :]
                    
                    # Process plate
                    cropped_plate_gray = cv.cvtColor(cropped_plate, cv.COLOR_BGR2GRAY)
                    _, plate_thresholded = cv.threshold(cropped_plate_gray, 64, 255, cv.THRESH_BINARY_INV)
                    
                    # Read the number plate
                    result = read_license_plate(plate_thresholded)
                    if result is not None:
                        license_number, license_number_score = result
                    else:
                        license_number, license_number_score = "-1", "-1"
                    
                    if license_number != -1 and license_number is not None:
                        results[frame_nmb][car_id] = {
                            'car': {'bbox': [x1car, y1car, x2car, y2car]},
                            'plate': {
                                'bbox': [x1, y1, x2, y2],
                                'text': license_number,
                                'bbox_score': score,
                                'text_score': license_number_score
                            }
                        }
            
            frame_nmb += 1
            
            # Optional: Limit processing for testing
            if frame_nmb >= 300:  # Process only first 300 frames for quick test
                print(f"Stopping after {frame_nmb} frames (test mode)")
                break
                
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    
    # Release video capture
    cap.release()
    
    # Write results
    print(f"Writing results to {output_csv}")
    write_csv(results, output_csv)
    
    # Print summary
    total_detections = sum(len(frame_results) for frame_results in results.values())
    print(f"\nProcessing complete!")
    print(f"  - Frames processed: {frame_nmb}")
    print(f"  - Total detections: {total_detections}")
    print(f"  - Results saved to: {output_csv}")
    
    # Show sample of detected plates
    if total_detections > 0:
        print("\nSample detections:")
        for frame_num in list(results.keys())[:3]:  # First 3 frames with detections
            if results[frame_num]:
                for car_id, detection in results[frame_num].items():
                    plate_text = detection['plate']['text']
                    if plate_text not in ['-1', '0']:
                        print(f"  Frame {frame_num}, Car {car_id}: {plate_text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test number plate recognition on a video')
    parser.add_argument('video', nargs='?', default='./out.mp4', 
                       help='Path to video file (default: ./out.mp4)')
    parser.add_argument('--output', '-o', default='./test_output.csv',
                       help='Output CSV file path (default: ./test_output.csv)')
    
    args = parser.parse_args()
    
    main(args.video, args.output)
