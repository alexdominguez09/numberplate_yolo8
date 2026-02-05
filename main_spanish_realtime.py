"""
Spanish License Plate Recognition System
With GPU acceleration and real-time visualization
"""
import os
import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import *
from utils_spanish import write_csv, read_spanish_license_plate, get_car
import time
from datetime import datetime
import torch

class SpanishLicensePlateDetector:
    def __init__(self, video_path: str, output_csv: str = "results_spanish.csv"):
        """
        Initialize Spanish license plate detector with GPU acceleration.
        
        Args:
            video_path: Path to input video file
            output_csv: Path to output CSV file
        """
        self.video_path = video_path
        self.output_csv = output_csv
        
        # Check GPU availability
        self.use_gpu = torch.cuda.is_available()
        print(f"{'='*60}")
        print("SPANISH LICENSE PLATE RECOGNITION SYSTEM")
        print(f"{'='*60}")
        print(f"GPU Acceleration: {'ENABLED' if self.use_gpu else 'DISABLED'}")
        if self.use_gpu:
            print(f"GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Load models with GPU if available
        print("\nLoading models...")
        device = 'cuda' if self.use_gpu else 'cpu'
        
        # Load YOLO models
        self.coco_model = YOLO('yolov8n.pt')
        self.license_plate_detector = YOLO('./models/license_plate_detector.pt')
        
        # Move models to GPU if available
        if self.use_gpu:
            self.coco_model.to(device)
            self.license_plate_detector.to(device)
        
        # Initialize tracker
        self.mot_tracker = Sort()
        
        # Results storage
        self.results = {}
        self.frame_count = 0
        self.plates_detected = 0
        
        # Performance tracking
        self.processing_times = []
        self.detection_times = []
        self.ocr_times = []
        
        # Visualization settings
        self.colors = {
            'car': (0, 255, 0),      # Green for cars
            'plate': (0, 0, 255),    # Red for plates
            'text': (255, 255, 0),   # Cyan for text
            'info': (255, 255, 255), # White for info
        }
        
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2
        
    def draw_bounding_boxes(self, frame, detections, plates, tracked_cars):
        """
        Draw bounding boxes and information on frame.
        
        Args:
            frame: Input frame
            detections: Vehicle detections
            plates: License plate detections
            tracked_cars: Tracked vehicles
        """
        frame_copy = frame.copy()
        
        # Draw vehicle bounding boxes
        for detection in detections:
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in [2, 3, 5, 7]:  # Cars, trucks, motorcycles, buses
                cv2.rectangle(frame_copy, (int(x1), int(y1)), (int(x2), int(y2)), 
                            self.colors['car'], 2)
                label = f"Vehicle: {score:.2f}"
                cv2.putText(frame_copy, label, (int(x1), int(y1)-10), 
                          self.font, self.font_scale, self.colors['car'], 
                          self.font_thickness)
        
        # Draw tracked vehicles with IDs
        for track in tracked_cars:
            x1, y1, x2, y2, track_id = track
            cv2.rectangle(frame_copy, (int(x1), int(y1)), (int(x2), int(y2)), 
                        (255, 165, 0), 2)  # Orange for tracked vehicles
            label = f"ID: {int(track_id)}"
            cv2.putText(frame_copy, label, (int(x1), int(y1)-30), 
                      self.font, self.font_scale, (255, 165, 0), 
                      self.font_thickness)
        
        # Draw license plate bounding boxes and text
        for plate in plates:
            x1, y1, x2, y2, score, class_id = plate
            
            # Draw plate bounding box
            cv2.rectangle(frame_copy, (int(x1), int(y1)), (int(x2), int(y2)), 
                        self.colors['plate'], 2)
            
            # Try to read plate
            plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
            if plate_crop.size > 0:
                plate_crop_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                plate_text, plate_score = read_spanish_license_plate(plate_crop_gray)
                
                if plate_text:
                    # Draw plate text above bounding box
                    text_label = f"{plate_text} ({plate_score:.2f})"
                    text_size = cv2.getTextSize(text_label, self.font, 
                                               self.font_scale, self.font_thickness)[0]
                    
                    # Background for text
                    text_bg = (int(x1), int(y1) - text_size[1] - 10,
                              int(x1) + text_size[0] + 10, int(y1) - 5)
                    cv2.rectangle(frame_copy, 
                                (text_bg[0], text_bg[1]),
                                (text_bg[2], text_bg[3]),
                                (0, 0, 0), -1)  # Black background
                    
                    # Plate text
                    cv2.putText(frame_copy, text_label, 
                              (int(x1) + 5, int(y1) - 10),
                              self.font, self.font_scale, self.colors['text'],
                              self.font_thickness)
        
        # Add performance info
        fps_text = f"FPS: {self.current_fps:.1f}" if hasattr(self, 'current_fps') else "FPS: --"
        plates_text = f"Plates: {self.plates_detected}"
        gpu_text = f"GPU: {'ON' if self.use_gpu else 'OFF'}"
        
        info_y = 30
        for text in [fps_text, plates_text, gpu_text, f"Frame: {self.frame_count}"]:
            cv2.putText(frame_copy, text, (10, info_y), 
                      self.font, 0.7, self.colors['info'], 2)
            info_y += 25
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame_copy, timestamp, (frame_copy.shape[1] - 250, 30),
                  self.font, 0.6, self.colors['info'], 2)
        
        return frame_copy
    
    def process_frame(self, frame):
        """
        Process a single frame for license plate detection.
        
        Args:
            frame: Input frame
            
        Returns:
            Processed results and visualization frame
        """
        frame_start = time.time()
        
        # Detect vehicles
        det_start = time.time()
        detections = self.coco_model(frame, verbose=False)[0]
        detections_ = []
        
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            
            # Filter for vehicles
            if int(class_id) in [2, 3, 5, 7] and score > 0.3:
                detections_.append([x1, y1, x2, y2, score])
        
        self.detection_times.append(time.time() - det_start)
        
        # Track vehicles
        if detections_:
            track_ids = self.mot_tracker.update(np.asarray(detections_))
        else:
            track_ids = []
        
        # Detect license plates
        plate_start = time.time()
        license_plates = self.license_plate_detector(frame, verbose=False)[0]
        
        frame_results = {}
        
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            
            # Apply confidence threshold
            if score < 0.2:
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
                
                # Read Spanish license plate
                ocr_start = time.time()
                license_plate_text, license_plate_text_score = read_spanish_license_plate(
                    license_plate_crop_gray, use_preprocessing=True
                )
                self.ocr_times.append(time.time() - ocr_start)
                
                if license_plate_text and license_plate_text_score > 0.1:
                    self.plates_detected += 1
                    
                    # Store results
                    if car_id not in frame_results:
                        frame_results[car_id] = {
                            'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                            'plate': {
                                'bbox': [x1, y1, x2, y2],
                                'bbox_score': score,
                                'text': license_plate_text,
                                'text_score': license_plate_text_score
                            }
                        }
        
        self.ocr_times.append(time.time() - plate_start)
        
        # Store frame results
        if frame_results:
            self.results[self.frame_count] = frame_results
        
        # Create visualization
        viz_frame = self.draw_bounding_boxes(frame, 
                                           detections.boxes.data.tolist() if hasattr(detections, 'boxes') else [],
                                           license_plates.boxes.data.tolist() if hasattr(license_plates, 'boxes') else [],
                                           track_ids)
        
        self.processing_times.append(time.time() - frame_start)
        
        return viz_frame, frame_results
    
    def process_video(self, max_frames: int = None, show_video: bool = True):
        """
        Process video file with real-time visualization.
        
        Args:
            max_frames: Maximum number of frames to process
            show_video: Whether to show real-time visualization
        """
        print(f"\nProcessing video: {self.video_path}")
        
        # Open video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {self.video_path}")
            return
        
        # Get video properties
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video info: {width}x{height} @ {self.fps} fps, {total_frames} frames total")
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
            print(f"Processing first {max_frames} frames")
        
        # Create window for display
        if show_video:
            cv2.namedWindow('Spanish License Plate Recognition', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Spanish License Plate Recognition', 1280, 720)
        
        # Process frames
        self.frame_count = 0
        start_time = time.time()
        
        print("\nStarting real-time processing...")
        print("Press 'q' to quit, 'p' to pause, 's' to save screenshot")
        
        paused = False
        
        while True:
            if not paused:
                # Read frame
                ret, frame = cap.read()
                if not ret or (max_frames and self.frame_count >= max_frames):
                    break
                
                # Process frame
                viz_frame, frame_results = self.process_frame(frame)
                
                # Calculate current FPS
                elapsed = time.time() - start_time
                self.current_fps = self.frame_count / elapsed if elapsed > 0 else 0
                
                # Display frame
                if show_video:
                    cv2.imshow('Spanish License Plate Recognition', viz_frame)
                
                self.frame_count += 1
                
                # Print status every 50 frames
                if self.frame_count % 50 == 0:
                    print(f"Processed {self.frame_count}/{total_frames} frames "
                          f"({self.frame_count/total_frames*100:.1f}%) "
                          f"at {self.current_fps:.1f} fps, "
                          f"detected {self.plates_detected} plates")
            
            # Handle keyboard input
            key = cv2.waitKey(1 if not paused else 0) & 0xFF
            
            if key == ord('q'):  # Quit
                print("\nQuitting...")
                break
            elif key == ord('p'):  # Pause/Resume
                paused = not paused
                print(f"{'Paused' if paused else 'Resumed'}")
            elif key == ord('s'):  # Save screenshot
                screenshot_path = f"screenshot_frame_{self.frame_count}.jpg"
                cv2.imwrite(screenshot_path, viz_frame if 'viz_frame' in locals() else frame)
                print(f"Screenshot saved: {screenshot_path}")
            elif key == ord('d'):  # Debug mode
                print(f"\nDebug info:")
                print(f"  Frame: {self.frame_count}")
                print(f"  FPS: {self.current_fps:.1f}")
                print(f"  Plates detected: {self.plates_detected}")
                if self.processing_times:
                    avg_time = sum(self.processing_times)/len(self.processing_times)
                    print(f"  Avg frame time: {avg_time*1000:.1f}ms")
        
        # Clean up
        cap.release()
        if show_video:
            cv2.destroyAllWindows()
        
        # Calculate final statistics
        elapsed_time = time.time() - start_time
        avg_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        print(f"\n{'='*50}")
        print("PROCESSING COMPLETE")
        print(f"{'='*50}")
        print(f"Total frames processed: {self.frame_count}")
        print(f"Total plates detected: {self.plates_detected}")
        print(f"Total processing time: {elapsed_time:.2f} seconds")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Plates per frame: {self.plates_detected/self.frame_count:.3f}" if self.frame_count > 0 else "Plates per frame: N/A")
        
        if self.processing_times:
            avg_processing = sum(self.processing_times)/len(self.processing_times)
            avg_detection = sum(self.detection_times)/len(self.detection_times) if self.detection_times else 0
            avg_ocr = sum(self.ocr_times)/len(self.ocr_times) if self.ocr_times else 0
            
            print(f"\nPerformance metrics:")
            print(f"  Average frame time: {avg_processing*1000:.1f} ms")
            print(f"  Average detection time: {avg_detection*1000:.1f} ms")
            print(f"  Average OCR time: {avg_ocr*1000:.1f} ms")
            print(f"  GPU acceleration: {'Enabled' if self.use_gpu else 'Disabled'}")
        
        # Save results to CSV
        if self.results:
            write_csv(self.results, self.output_csv)
            print(f"\nResults saved to: {self.output_csv}")
            
            # Show sample results
            print(f"\nSample results (first 5 plates):")
            frame_keys = sorted(self.results.keys())
            count = 0
            for frame_key in frame_keys:
                for car_key in self.results[frame_key]:
                    plate_info = self.results[frame_key][car_key]['plate']
                    print(f"Frame {frame_key}, Car {car_key}: {plate_info['text']} "
                          f"(confidence: {plate_info['text_score']:.3f})")
                    count += 1
                    if count >= 5:
                        break
                if count >= 5:
                    break
        else:
            print("\nNo license plates detected.")
        
        return self.results, avg_fps

def test_spanish_plates():
    """Test Spanish plate recognition on sample images."""
    print("\nTesting Spanish plate recognition on sample images...")
    print("="*60)
    
    test_dir = "test_spanish_plates"
    if not os.path.exists(test_dir):
        print(f"Test directory not found: {test_dir}")
        return
    
    import glob
    test_images = glob.glob(os.path.join(test_dir, "*.png")) + \
                  glob.glob(os.path.join(test_dir, "*.jpg"))
    
    if not test_images:
        print("No test images found")
        return
    
    print(f"Found {len(test_images)} test images")
    
    for img_path in test_images[:5]:  # Test first 5 images
        print(f"\nProcessing: {os.path.basename(img_path)}")
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"  Could not read image")
            continue
        
        # Convert to grayscale for OCR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Read Spanish plate
        plate_text, plate_score = read_spanish_license_plate(gray, use_preprocessing=True)
        
        if plate_text:
            print(f"  Detected: {plate_text} (confidence: {plate_score:.3f})")
            
            # Display image with result
            display = cv2.resize(img, (400, 150))
            cv2.putText(display, plate_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow(f"Test: {os.path.basename(img_path)}", display)
            cv2.waitKey(500)  # Show for 500ms
            cv2.destroyAllWindows()
        else:
            print(f"  No plate detected")
    
    print("\nSpanish plate testing complete!")

def main():
    """Main function."""
    # Configuration
    video_path = "/home/alex/Downloads/video_carplates1.mkv"
    output_csv = "results_spanish_realtime.csv"
    max_frames = 300  # Process first 300 frames for testing
    show_video = True  # Show real-time visualization
    
    # Optional: Test Spanish plates first
    test_first = False
    if test_first:
        test_spanish_plates()
    
    # Create and run detector
    detector = SpanishLicensePlateDetector(video_path, output_csv)
    
    # Process video
    results, fps = detector.process_video(
        max_frames=max_frames,
        show_video=show_video
    )
    
    print(f"\n{'='*60}")
    print(f"Spanish License Plate Recognition Complete!")
    print(f"Final performance: {fps:.1f} FPS")
    print(f"{'='*60}")
    
    # Save performance report
    report = {
        'timestamp': datetime.now().isoformat(),
        'video': video_path,
        'frames_processed': detector.frame_count,
        'plates_detected': detector.plates_detected,
        'average_fps': fps,
        'gpu_acceleration': detector.use_gpu,
        'output_file': output_csv
    }
    
    import json
    with open('spanish_detection_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Performance report saved to: spanish_detection_report.json")

if __name__ == "__main__":
    main()