import os
import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import *
from utils_enhanced import write_csv, read_license_plate_enhanced, get_car
import time
from tqdm import tqdm

def process_video(video_path: str, output_csv: str = "results_enhanced.csv", 
                  max_frames: int = None, show_progress: bool = True):
    """
    Process video for license plate detection and recognition.
    
    Args:
        video_path: Path to input video file
        output_csv: Path to output CSV file
        max_frames: Maximum number of frames to process (None for all)
        show_progress: Whether to show progress bar
    """
    print(f"Processing video: {video_path}")
    print(f"Output will be saved to: {output_csv}")
    
    # Load models
    print("Loading models...")
    coco_model = YOLO('yolov8n.pt')
    license_plate_detector = YOLO('./models/license_plate_detector.pt')
    
    # Initialize tracker
    mot_tracker = Sort()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info: {width}x{height} @ {fps} fps, {total_frames} frames total")
    
    if max_frames:
        total_frames = min(total_frames, max_frames)
        print(f"Processing first {max_frames} frames")
    
    # Dictionary to store results
    results = {}
    
    # Process frames
    frame_nmr = -1
    processed_frames = 0
    plates_detected = 0
    
    # Create progress bar
    if show_progress:
        pbar = tqdm(total=total_frames, desc="Processing video")
    
    start_time = time.time()
    
    while True:
        frame_nmr += 1
        
        # Check if we've reached max frames
        if max_frames and frame_nmr >= max_frames:
            break
        
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            
            # Filter for cars, trucks, motorcycles, buses (COCO classes 2,3,5,7)
            if int(class_id) in [2, 3, 5, 7]:
                detections_.append([x1, y1, x2, y2, score])
        
        # Track vehicles
        if detections_:
            track_ids = mot_tracker.update(np.asarray(detections_))
        else:
            track_ids = []
        
        # Detect license plates
        license_plates = license_plate_detector(frame)[0]
        
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            
            # Assign plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
            
            if car_id != -1:
                # Crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                
                # Process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                
                # Read license plate number with enhanced OCR
                license_plate_text, license_plate_text_score = read_license_plate_enhanced(
                    license_plate_crop_gray, use_preprocessing=True
                )
                
                if license_plate_text:
                    plates_detected += 1
                    
                    # Store results
                    if frame_nmr not in results:
                        results[frame_nmr] = {}
                    
                    if car_id not in results[frame_nmr]:
                        results[frame_nmr][car_id] = {
                            'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                            'plate': {
                                'bbox': [x1, y1, x2, y2],
                                'bbox_score': score,
                                'text': license_plate_text,
                                'text_score': license_plate_text_score
                            }
                        }
        
        processed_frames += 1
        
        # Update progress bar
        if show_progress:
            pbar.update(1)
        
        # Print status every 50 frames
        if frame_nmr % 50 == 0:
            elapsed = time.time() - start_time
            fps_processed = processed_frames / elapsed if elapsed > 0 else 0
            print(f"\nProcessed {processed_frames}/{total_frames} frames "
                  f"({processed_frames/total_frames*100:.1f}%) "
                  f"at {fps_processed:.1f} fps, "
                  f"detected {plates_detected} plates")
    
    # Clean up
    if show_progress:
        pbar.close()
    cap.release()
    
    # Calculate statistics
    elapsed_time = time.time() - start_time
    processing_fps = processed_frames / elapsed_time if elapsed_time > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*50}")
    print(f"Total frames processed: {processed_frames}")
    print(f"Total plates detected: {plates_detected}")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    print(f"Processing speed: {processing_fps:.2f} fps")
    print(f"Plates per frame: {plates_detected/processed_frames:.3f}" if processed_frames > 0 else "Plates per frame: N/A")
    
    # Write results to CSV
    if results:
        write_csv(results, output_csv)
        print(f"Results saved to: {output_csv}")
        
        # Show sample results
        print(f"\nSample results (first 5 plates):")
        frame_keys = sorted(results.keys())
        count = 0
        for frame_key in frame_keys:
            for car_key in results[frame_key]:
                plate_info = results[frame_key][car_key]['plate']
                print(f"Frame {frame_key}, Car {car_key}: {plate_info['text']} "
                      f"(confidence: {plate_info['text_score']:.3f})")
                count += 1
                if count >= 5:
                    break
            if count >= 5:
                break
    else:
        print("No license plates detected.")
    
    return results

def main():
    """Main function to process video."""
    # Configuration
    video_path = "/home/alex/Downloads/video_carplates1.mkv"
    output_csv = "results_enhanced.csv"
    max_frames = 100  # Process first 100 frames for quick testing
    show_progress = True
    
    print("="*60)
    print("ENHANCED LICENSE PLATE RECOGNITION SYSTEM")
    print("="*60)
    
    # Process video
    results = process_video(
        video_path=video_path,
        output_csv=output_csv,
        max_frames=max_frames,
        show_progress=show_progress
    )
    
    print("\n" + "="*60)
    print("Processing complete!")
    print("="*60)

if __name__ == "__main__":
    main()