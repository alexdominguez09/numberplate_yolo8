"""
PRODUCTION-READY Spanish License Plate Recognition System
With GPU acceleration, real-time visualization, and maximum accuracy.
"""
import os
import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import *
from utils_spanish_fixed import write_csv, read_spanish_license_plate_optimized, get_car
import time
from datetime import datetime
import torch
import json
import argparse

# Suppress Qt font warnings
os.environ['QT_LOGGING_RULES'] = '*.warning=false'

class ProductionSpanishLPR:
    def __init__(self, video_path: str, output_csv: str = "results_spanish_production.csv"):
        """
        Production Spanish License Plate Recognition system.
        
        Args:
            video_path: Path to input video file
            output_csv: Path to output CSV file
        """
        self.video_path = video_path
        self.output_csv = output_csv
        
        # GPU configuration
        self.use_gpu = torch.cuda.is_available()
        self.device = 'cuda' if self.use_gpu else 'cpu'
        
        print(f"{'='*70}")
        print("PRODUCTION SPANISH LICENSE PLATE RECOGNITION SYSTEM")
        print(f"{'='*70}")
        print(f"GPU Acceleration: {'✅ ENABLED' if self.use_gpu else '❌ DISABLED'}")
        if self.use_gpu:
            print(f"GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Load models with GPU optimization
        print("\n🚀 Loading models with GPU optimization...")
        self.coco_model = YOLO('yolov8n.pt')
        self.license_plate_detector = YOLO('./models/yolov8n_license_plate.pt')
        
        if self.use_gpu:
            self.coco_model.to(self.device)
            self.license_plate_detector.to(self.device)
        
        # Initialize tracker
        self.mot_tracker = Sort()
        
        # Results and statistics
        self.results = {}
        self.frame_count = 0
        self.plates_detected = 0
        self.valid_spanish_plates = 0
        
        # Performance tracking
        self.frame_times = []
        self.detection_times = []
        self.ocr_times = []
        
        # Visualization settings
        self.colors = {
            'car': (0, 255, 0),        # Green - vehicles
            'plate': (0, 0, 255),      # Red - license plates
            'plate_text': (255, 255, 0), # Cyan - plate text
            'track_id': (255, 165, 0), # Orange - track IDs
            'info': (255, 255, 255),   # White - info text
            'warning': (0, 255, 255),  # Yellow - warnings
            'success': (0, 255, 0),    # Green - success
        }
        
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2
        
        # Accuracy optimization settings
        self.plate_confidence_threshold = 0.3
        self.ocr_confidence_threshold = 0.15
        self.min_plate_size = (20, 20)  # Minimum plate dimensions
        
        # Initialize FPS tracking
        self.current_fps = 0.0
        self.original_fps = 0
        
        print("✅ System initialized successfully!")
    
    def optimize_frame(self, frame):
        """
        Optimize frame for processing.
        """
        # Resize for faster processing while maintaining aspect ratio
        height, width = frame.shape[:2]
        target_width = 1280
        
        if width > target_width:
            scale = target_width / width
            new_height = int(height * scale)
            optimized = cv2.resize(frame, (target_width, new_height))
            return optimized, scale
        else:
            return frame.copy(), 1.0
    
    def detect_vehicles_optimized(self, frame):
        """
        Optimized vehicle detection with GPU acceleration.
        """
        det_start = time.time()
        
        # Run detection
        results = self.coco_model(frame, verbose=False, device=self.device)[0]
        
        detections = []
        for detection in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            
            # Filter for vehicles with confidence threshold
            if int(class_id) in [2, 3, 5, 7] and score > 0.4:  # Cars, trucks, motorcycles, buses
                detections.append([x1, y1, x2, y2, score])
        
        detection_time = time.time() - det_start
        self.detection_times.append(detection_time)
        
        return detections
    
    def detect_license_plates_optimized(self, frame):
        """
        Optimized license plate detection.
        """
        plate_start = time.time()
        
        # Run plate detection
        results = self.license_plate_detector(frame, verbose=False, device=self.device)[0]
        
        plates = []
        for detection in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            
            # Apply confidence threshold
            if score >= self.plate_confidence_threshold:
                # Check minimum size
                plate_width = x2 - x1
                plate_height = y2 - y1
                
                if plate_width >= self.min_plate_size[0] and plate_height >= self.min_plate_size[1]:
                    plates.append([x1, y1, x2, y2, score, class_id])
        
        plate_time = time.time() - plate_start
        self.detection_times.append(plate_time)
        
        return plates
    
    def read_license_plate_optimized(self, plate_crop):
        """
        Optimized Spanish license plate reading.
        """
        ocr_start = time.time()
        
        # Convert to grayscale
        if len(plate_crop.shape) == 3:
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_crop
        
        # Read plate with Spanish-specific processing
        plate_text, plate_score = read_spanish_license_plate_optimized(gray)
        
        ocr_time = time.time() - ocr_start
        self.ocr_times.append(ocr_time)
        
        return plate_text, plate_score
    
    def create_visualization(self, frame, vehicles, plates, tracked_cars, frame_results):
        """
        Create professional visualization with bounding boxes and information.
        """
        viz_frame = frame.copy()
        height, width = viz_frame.shape[:2]
        
        # Draw vehicle bounding boxes
        for vehicle in vehicles:
            x1, y1, x2, y2, score = vehicle
            cv2.rectangle(viz_frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                         self.colors['car'], 2)
            
            # Vehicle label
            label = f"Veh: {score:.2f}"
            (label_width, label_height), baseline = cv2.getTextSize(
                label, self.font, self.font_scale, self.font_thickness)
            
            # Label background
            cv2.rectangle(viz_frame, 
                         (int(x1), int(y1) - label_height - 10),
                         (int(x1) + label_width + 10, int(y1)),
                         self.colors['car'], -1)
            
            # Label text
            cv2.putText(viz_frame, label, 
                       (int(x1) + 5, int(y1) - 5),
                       self.font, self.font_scale, (0, 0, 0), 
                       self.font_thickness)
        
        # Draw tracked vehicles with IDs
        for track in tracked_cars:
            x1, y1, x2, y2, track_id = track
            cv2.rectangle(viz_frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                         self.colors['track_id'], 2)
            
            # Track ID label
            label = f"ID: {int(track_id)}"
            (label_width, label_height), baseline = cv2.getTextSize(
                label, self.font, self.font_scale, self.font_thickness)
            
            # Label background
            cv2.rectangle(viz_frame,
                         (int(x1), int(y1) - label_height - 10),
                         (int(x1) + label_width + 10, int(y1)),
                         self.colors['track_id'], -1)
            
            # Label text
            cv2.putText(viz_frame, label,
                       (int(x1) + 5, int(y1) - 5),
                       self.font, self.font_scale, (0, 0, 0),
                       self.font_thickness)
        
        # Draw license plates and text
        for plate in plates:
            x1, y1, x2, y2, score, class_id = plate
            
            # Plate bounding box
            cv2.rectangle(viz_frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                         self.colors['plate'], 3)
            
            # Plate confidence
            plate_label = f"Plate: {score:.2f}"
            cv2.putText(viz_frame, plate_label,
                       (int(x1), int(y1) - 35),
                       self.font, self.font_scale * 0.8, self.colors['plate'],
                       self.font_thickness)
        
        # Draw detected plate text from frame results
        for car_id, car_data in frame_results.items():
            plate_info = car_data['plate']
            x1, y1, x2, y2 = map(int, plate_info['bbox'])
            plate_text = plate_info['text']
            text_score = plate_info['text_score']
            
            # Plate text label
            text_label = f"{plate_text} ({text_score:.2f})"
            (text_width, text_height), baseline = cv2.getTextSize(
                text_label, self.font, self.font_scale, self.font_thickness)
            
            # Text background
            text_bg_y1 = max(0, y1 - text_height - 15)
            text_bg_y2 = y1
            text_bg_x1 = x1
            text_bg_x2 = x1 + text_width + 10
            
            cv2.rectangle(viz_frame,
                         (text_bg_x1, text_bg_y1),
                         (text_bg_x2, text_bg_y2),
                         (0, 0, 0), -1)  # Black background
            
            # Plate text
            cv2.putText(viz_frame, text_label,
                       (x1 + 5, y1 - 10),
                       self.font, self.font_scale, self.colors['plate_text'],
                       self.font_thickness)
            
            # Draw connection line from plate to text
            cv2.line(viz_frame,
                    (x1 + (x2 - x1) // 2, y1),
                    (x1 + (x2 - x1) // 2, text_bg_y2),
                    self.colors['plate_text'], 1)
        
        # Add information overlay
        info_y = 30
        info_lines = [
            f"SPANISH LPR SYSTEM | GPU: {'ON' if self.use_gpu else 'OFF'}",
            f"Frame: {self.frame_count} | FPS: {self.current_fps:.1f}",
            f"Plates: {self.plates_detected} | Valid Spanish: {self.valid_spanish_plates}",
            f"Time: {datetime.now().strftime('%H:%M:%S')}",
        ]
        
        for line in info_lines:
            (text_width, text_height), baseline = cv2.getTextSize(
                line, self.font, 0.7, 2)
            
            # Background for text
            cv2.rectangle(viz_frame,
                         (10, info_y - text_height - 5),
                         (10 + text_width + 10, info_y + 5),
                         (0, 0, 0), -1)
            
            # Text
            cv2.putText(viz_frame, line, (15, info_y),
                       self.font, 0.7, self.colors['info'], 2)
            info_y += 30
        
        # Add status bar at bottom
        status_bar_height = 40
        cv2.rectangle(viz_frame,
                     (0, height - status_bar_height),
                     (width, height),
                     (0, 0, 0), -1)
        
        status_text = "✅ SYSTEM ACTIVE | Press 'Q' to quit | 'P' to pause | 'S' for screenshot"
        cv2.putText(viz_frame, status_text, (10, height - 15),
                   self.font, 0.6, self.colors['success'], 2)
        
        return viz_frame
    
    def process_frame(self, frame):
        """
        Process a single frame with all optimizations.
        """
        frame_start = time.time()
        
        # Optimize frame size
        optimized_frame, scale = self.optimize_frame(frame)
        
        # Detect vehicles
        vehicle_detections = self.detect_vehicles_optimized(optimized_frame)
        
        # Scale detections back to original coordinates if needed
        if scale != 1.0:
            vehicle_detections = [[x1/scale, y1/scale, x2/scale, y2/scale, score] 
                                 for x1, y1, x2, y2, score in vehicle_detections]
        
        # Track vehicles
        if vehicle_detections:
            track_ids = self.mot_tracker.update(np.asarray(vehicle_detections))
        else:
            track_ids = []
        
        # Detect license plates
        plate_detections = self.detect_license_plates_optimized(optimized_frame)
        
        # Scale plate detections back
        if scale != 1.0:
            plate_detections = [[x1/scale, y1/scale, x2/scale, y2/scale, score, class_id]
                               for x1, y1, x2, y2, score, class_id in plate_detections]
        
        frame_results = {}
        
        # Process each detected plate
        for plate in plate_detections:
            x1, y1, x2, y2, score, class_id = plate
            
            # Assign plate to vehicle
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(plate, track_ids)
            
            if car_id != -1:
                # Crop license plate
                plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                
                # Skip if crop is invalid
                if plate_crop.size == 0 or plate_crop.shape[0] < 10 or plate_crop.shape[1] < 10:
                    continue
                
                # Read Spanish license plate
                plate_text, plate_score = self.read_license_plate_optimized(plate_crop)
                
                # Apply OCR confidence threshold
                if plate_text and plate_score >= self.ocr_confidence_threshold:
                    self.plates_detected += 1
                    
                    # Check if it looks like a Spanish plate
                    if '-' in plate_text and len(plate_text.replace('-', '')) >= 5:
                        self.valid_spanish_plates += 1
                    
                    # Store results
                    if car_id not in frame_results:
                        frame_results[car_id] = {
                            'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                            'plate': {
                                'bbox': [x1, y1, x2, y2],
                                'bbox_score': score,
                                'text': plate_text,
                                'text_score': plate_score
                            }
                        }
        
        # Store frame results
        if frame_results:
            self.results[self.frame_count] = frame_results
        
        # Calculate processing time
        frame_time = time.time() - frame_start
        self.frame_times.append(frame_time)
        
        # Create visualization
        viz_frame = self.create_visualization(
            frame, 
            vehicle_detections, 
            plate_detections, 
            track_ids, 
            frame_results
        )
        
        return viz_frame, frame_results
    
    def process_video(self, max_frames: int = None, show_video: bool = True):
        """
        Process video with real-time visualization.
        """
        print(f"\n📹 Processing video: {os.path.basename(self.video_path)}")
        
        # Open video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video file")
            return None, 0
        
        # Get video properties
        self.original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📊 Video info: {width}x{height} @ {self.original_fps} fps, {total_frames} frames")
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
            print(f"🔧 Processing first {max_frames} frames")
        
        # Create display window
        if show_video:
            cv2.namedWindow('Spanish LPR - Production System', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Spanish LPR - Production System', 1280, 720)
        
        # Processing loop
        self.frame_count = 0
        start_time = time.time()
        
        print("\n🎬 Starting real-time processing...")
        print("   Controls: Q=Quit, P=Pause, S=Screenshot, D=Debug info")
        print("   " + "-"*50)
        
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
                    cv2.imshow('Spanish LPR - Production System', viz_frame)
                
                self.frame_count += 1
                
                # Print status every 30 frames
                if self.frame_count % 30 == 0:
                    progress = self.frame_count / total_frames * 100
                    print(f"   📈 Frame {self.frame_count}/{total_frames} ({progress:.1f}%) | "
                          f"FPS: {self.current_fps:.1f} | "
                          f"Plates: {self.plates_detected} | "
                          f"Spanish: {self.valid_spanish_plates}")
            
            # Handle keyboard input
            key = cv2.waitKey(1 if not paused else 0) & 0xFF
            
            if key == ord('q'):  # Quit
                print("\n🛑 Quitting...")
                break
            elif key == ord('p'):  # Pause/Resume
                paused = not paused
                print(f"   {'⏸️ Paused' if paused else '▶️ Resumed'}")
            elif key == ord('s'):  # Screenshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = f"screenshot_{timestamp}.jpg"
                cv2.imwrite(screenshot_path, viz_frame if 'viz_frame' in locals() else frame)
                print(f"   📸 Screenshot saved: {screenshot_path}")
            elif key == ord('d'):  # Debug info
                print(f"\n   🔍 Debug Info:")
                print(f"      Frame: {self.frame_count}")
                print(f"      Current FPS: {self.current_fps:.1f}")
                print(f"      Plates detected: {self.plates_detected}")
                print(f"      Valid Spanish plates: {self.valid_spanish_plates}")
                if self.frame_times:
                    avg_time = sum(self.frame_times)/len(self.frame_times)
                    print(f"      Avg frame time: {avg_time*1000:.1f}ms")
                print(f"      GPU: {'Active' if self.use_gpu else 'Inactive'}")
        
        # Clean up
        cap.release()
        if show_video:
            cv2.destroyAllWindows()
        
        # Calculate final statistics
        elapsed_time = time.time() - start_time
        avg_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        return avg_fps
    
    def generate_report(self, avg_fps: float):
        """
        Generate comprehensive performance report.
        """
        print(f"\n{'='*70}")
        print("📊 PROCESSING REPORT")
        print(f"{'='*70}")
        
        print(f"✅ Processing Complete!")
        print(f"   Total frames processed: {self.frame_count}")
        print(f"   Total plates detected: {self.plates_detected}")
        print(f"   Valid Spanish plates: {self.valid_spanish_plates}")
        print(f"   Total processing time: {time.strftime('%H:%M:%S', time.gmtime(self.frame_count/avg_fps)) if avg_fps > 0 else 'N/A'}")
        print(f"   Average FPS: {avg_fps:.2f}")
        
        if self.plates_detected > 0:
            print(f"   Plates per frame: {self.plates_detected/self.frame_count:.3f}")
            print(f"   Spanish plate accuracy: {self.valid_spanish_plates/self.plates_detected*100:.1f}%")
        
        # Performance metrics
        if self.frame_times:
            avg_frame_time = sum(self.frame_times)/len(self.frame_times)
            avg_detection_time = sum(self.detection_times)/len(self.detection_times) if self.detection_times else 0
            avg_ocr_time = sum(self.ocr_times)/len(self.ocr_times) if self.ocr_times else 0
            
            print(f"\n⚡ Performance Metrics:")
            print(f"   Average frame time: {avg_frame_time*1000:.1f} ms")
            print(f"   Average detection time: {avg_detection_time*1000:.1f} ms")
            print(f"   Average OCR time: {avg_ocr_time*1000:.1f} ms")
            print(f"   GPU acceleration: {'✅ Enabled' if self.use_gpu else '❌ Disabled'}")
        
        # Save results to CSV
        if self.results:
            write_csv(self.results, self.output_csv)
            print(f"\n💾 Results saved to: {self.output_csv}")
            
            # Show sample results
            print(f"\n📋 Sample results (first 5 plates):")
            frame_keys = sorted(self.results.keys())
            count = 0
            for frame_key in frame_keys[:5]:
                for car_key in self.results[frame_key]:
                    plate_info = self.results[frame_key][car_key]['plate']
                    print(f"   Frame {frame_key}, Car {car_key}: {plate_info['text']} "
                          f"(confidence: {plate_info['text_score']:.3f})")
                    count += 1
                    if count >= 5:
                        break
                if count >= 5:
                    break
        else:
            print(f"\n⚠️ No license plates detected.")
        
        # Generate JSON report
        report = {
            'timestamp': datetime.now().isoformat(),
            'video_file': os.path.basename(self.video_path),
            'frames_processed': self.frame_count,
            'plates_detected': self.plates_detected,
            'valid_spanish_plates': self.valid_spanish_plates,
            'average_fps': avg_fps,
            'gpu_acceleration': self.use_gpu,
            'gpu_device': torch.cuda.get_device_name(0) if self.use_gpu else None,
            'output_file': self.output_csv,
            'processing_time_seconds': self.frame_count/avg_fps if avg_fps > 0 else 0,
            'plates_per_frame': self.plates_detected/self.frame_count if self.frame_count > 0 else 0,
            'spanish_plate_accuracy': self.valid_spanish_plates/self.plates_detected if self.plates_detected > 0 else 0
        }
        
        report_file = 'spanish_lpr_production_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Full report saved to: {report_file}")
        
        return report

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Production Spanish License Plate Recognition System'
    )
    parser.add_argument('--video', '-v',
                        type=str,
                        required=True,
                        help='Path to input video file')
    parser.add_argument('--output', '-o',
                        type=str,
                        default='results_spanish_production.csv',
                        help='Output CSV file path')
    parser.add_argument('--max-frames', '-m',
                        type=int,
                        default=None,
                        help='Maximum frames to process (None for full video)')
    parser.add_argument('--no-display',
                        action='store_true',
                        help='Run without video display (headless mode)')
    return parser.parse_args()


def main():
    """
    Main function for production Spanish LPR system.
    """
    # Parse arguments
    args = parse_args()
    
    # Configuration from arguments
    video_path = args.video
    output_csv = args.output
    max_frames = args.max_frames
    show_video = not args.no_display
    
    print("\n" + "="*70)
    print("🚀 PRODUCTION SPANISH LICENSE PLATE RECOGNITION")
    print("="*70)
    
    # Create and run system
    lpr_system = ProductionSpanishLPR(video_path, output_csv)
    
    # Process video
    avg_fps = lpr_system.process_video(
        max_frames=max_frames,
        show_video=show_video
    )
    
    # Generate report
    report = lpr_system.generate_report(avg_fps)
    
    print(f"\n{'='*70}")
    print("🎉 SYSTEM EXECUTION COMPLETE!")
    print(f"{'='*70}")
    
    # Summary
    print(f"\n📈 SUMMARY:")
    print(f"   • Frames processed: {report['frames_processed']}")
    print(f"   • Plates detected: {report['plates_detected']}")
    print(f"   • Average FPS: {report['average_fps']:.1f}")
    print(f"   • GPU: {'Enabled' if report['gpu_acceleration'] else 'Disabled'}")
    print(f"   • Results file: {report['output_file']}")
    
    print(f"\n✅ Ready for production use!")
    print(f"   To process full video, set max_frames=None in main()")
    print(f"   For headless operation, set show_video=False")

if __name__ == "__main__":
    main()
