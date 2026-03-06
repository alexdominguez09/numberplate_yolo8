import os
import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import *
from utils_enhanced import write_csv, read_license_plate_enhanced, get_car
import time
from tqdm import tqdm

def process_video_optimized(video_path: str, output_csv: str = "results_optimized.csv", 
                           max_frames: int = None, show_progress: bool = True,
                           skip_frames: int = 0, confidence_threshold: float = 0.1):
    """
    Optimized video processing with performance improvements.
    
    Args:
        video_path: Path to input video file
        output_csv: Path to output CSV file
        max_frames: Maximum number of frames to process
        show_progress: Whether to show progress bar
        skip_frames: Number of frames to skip between processing (0 = process all)
        confidence_threshold: Minimum confidence for plate detection
    """
    print(f"Processing video: {video_path}")
    print(f"Output will be saved to: {output_csv}")
    print(f"Optimization settings: skip_frames={skip_frames}, confidence_threshold={confidence_threshold}")
    
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
    
    # Performance tracking
    detection_times = []
    ocr_times = []
    
    while True:
        frame_nmr += 1
        
        # Check if we've reached max frames
        if max_frames and frame_nmr >= max_frames:
            break
        
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames if configured
        if skip_frames > 0 and frame_nmr % (skip_frames + 1) != 0:
            if show_progress:
                pbar.update(1)
            continue
        
        # OPTIMIZATION 1: Resize frame for faster processing (maintain aspect ratio)
        target_width = 640
        scale_factor = target_width / width
        target_height = int(height * scale_factor)
        
        if scale_factor != 1.0:
            frame_resized = cv2.resize(frame, (target_width, target_height))
        else:
            frame_resized = frame
        
        # Detect vehicles
        det_start = time.time()
        detections = coco_model(frame_resized, verbose=False)[0]
        detections_ = []
        
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            
            # Filter for cars, trucks, motorcycles, buses (COCO classes 2,3,5,7)
            if int(class_id) in [2, 3, 5, 7] and score > 0.3:  # Confidence threshold
                # Scale back to original coordinates if needed
                if scale_factor != 1.0:
                    x1, y1, x2, y2 = (x1/scale_factor, y1/scale_factor, 
                                     x2/scale_factor, y2/scale_factor)
                detections_.append([x1, y1, x2, y2, score])
        
        detection_times.append(time.time() - det_start)
        
        # Track vehicles
        if detections_:
            track_ids = mot_tracker.update(np.asarray(detections_))
        else:
            track_ids = []
        
        # Detect license plates
        plate_start = time.time()
        license_plates = license_plate_detector(frame, verbose=False)[0]
        
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            
            # Apply confidence threshold
            if score < confidence_threshold:
                continue
            
            # Assign plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
            
            if car_id != -1:
                # Crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                
                # Skip if crop is too small
                if license_plate_crop.size == 0 or license_plate_crop.shape[0] < 10 or license_plate_crop.shape[1] < 10:
                    continue
                
                # Process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                
                # Read license plate number with enhanced OCR
                ocr_start = time.time()
                license_plate_text, license_plate_text_score = read_license_plate_enhanced(
                    license_plate_crop_gray, use_preprocessing=True
                )
                ocr_times.append(time.time() - ocr_start)
                
                if license_plate_text and license_plate_text_score > 0.05:  # OCR confidence threshold
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
        
        ocr_times.append(time.time() - plate_start)
        processed_frames += 1
        
        # Update progress bar
        if show_progress:
            pbar.update(1)
        
        # Print status every 100 frames
        if processed_frames % 100 == 0:
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
    
    # Performance metrics
    avg_detection_time = sum(detection_times)/len(detection_times) if detection_times else 0
    avg_ocr_time = sum(ocr_times)/len(ocr_times) if ocr_times else 0
    
    print(f"\n{'='*50}")
    print(f"OPTIMIZED PROCESSING COMPLETE")
    print(f"{'='*50}")
    print(f"Total frames processed: {processed_frames}")
    print(f"Total plates detected: {plates_detected}")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    print(f"Processing speed: {processing_fps:.2f} fps")
    print(f"Plates per frame: {plates_detected/processed_frames:.3f}" if processed_frames > 0 else "Plates per frame: N/A")
    print(f"\nPerformance metrics:")
    print(f"  Average detection time: {avg_detection_time*1000:.1f} ms")
    print(f"  Average OCR time: {avg_ocr_time*1000:.1f} ms")
    print(f"  Frames skipped: {skip_frames}")
    
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
    
    return results, processing_fps

def compare_optimizations():
    """Compare different optimization settings."""
    video_path = "/home/alex/Downloads/video_carplates1.mkv"
    
    print("="*60)
    print("OPTIMIZATION COMPARISON")
    print("="*60)
    
    test_configs = [
        {"name": "Baseline", "skip_frames": 0, "confidence_threshold": 0.1},
        {"name": "Skip 1 frame", "skip_frames": 1, "confidence_threshold": 0.1},
        {"name": "Skip 2 frames", "skip_frames": 2, "confidence_threshold": 0.1},
        {"name": "Higher confidence", "skip_frames": 0, "confidence_threshold": 0.3},
        {"name": "Skip 1 + High conf", "skip_frames": 1, "confidence_threshold": 0.3},
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\n{'='*40}")
        print(f"Testing: {config['name']}")
        print(f"Settings: skip_frames={config['skip_frames']}, confidence_threshold={config['confidence_threshold']}")
        print('='*40)
        
        output_csv = f"results_{config['name'].replace(' ', '_').lower()}.csv"
        
        result, fps = process_video_optimized(
            video_path=video_path,
            output_csv=output_csv,
            max_frames=100,  # Test with 100 frames for comparison
            show_progress=False,
            skip_frames=config['skip_frames'],
            confidence_threshold=config['confidence_threshold']
        )
        
        results.append({
            "config": config['name'],
            "fps": fps,
            "plates": len(result) if result else 0,
            "settings": config
        })
    
    # Print comparison
    print(f"\n{'='*60}")
    print("OPTIMIZATION COMPARISON RESULTS")
    print('='*60)
    print(f"{'Configuration':<20} {'FPS':<10} {'Plates':<10} {'Settings'}")
    print('-'*60)
    
    for result in results:
        settings_str = f"skip={result['settings']['skip_frames']}, conf={result['settings']['confidence_threshold']}"
        print(f"{result['config']:<20} {result['fps']:<10.2f} {result['plates']:<10} {settings_str}")
    
    # Find best configuration
    best = max(results, key=lambda x: x['fps'])
    print(f"\nBest configuration: {best['config']} ({best['fps']:.2f} fps)")
    
    return results

def main():
    """Main function to run optimized processing."""
    # Configuration
    video_path = "/home/alex/Downloads/video_carplates1.mkv"
    output_csv = "results_optimized_final.csv"
    max_frames = 200  # Process 200 frames for better testing
    show_progress = True
    skip_frames = 1  # Skip every other frame (process at 15 fps instead of 30)
    confidence_threshold = 0.2  # Higher confidence threshold
    
    print("="*60)
    print("OPTIMIZED LICENSE PLATE RECOGNITION SYSTEM")
    print("="*60)
    
    # Process video with optimizations
    results, fps = process_video_optimized(
        video_path=video_path,
        output_csv=output_csv,
        max_frames=max_frames,
        show_progress=show_progress,
        skip_frames=skip_frames,
        confidence_threshold=confidence_threshold
    )
    
    print(f"\n{'='*60}")
    print(f"Final optimized processing: {fps:.2f} fps")
    print("="*60)
    
    # Optional: Run comparison tests
    run_comparison = False  # Set to True to run comparison tests
    if run_comparison:
        compare_optimizations()

if __name__ == "__main__":
    main()