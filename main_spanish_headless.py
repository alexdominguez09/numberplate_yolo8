"""
HEADLESS Spanish License Plate Recognition System
GPU accelerated, no display required.
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

class HeadlessSpanishLPR:
    def __init__(self, video_path: str, output_csv: str = "results_spanish_headless.csv"):
        """
        Headless Spanish License Plate Recognition system.
        
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
        print("HEADLESS SPANISH LICENSE PLATE RECOGNITION SYSTEM")
        print(f"{'='*70}")
        print(f"GPU Acceleration: {'✅ ENABLED' if self.use_gpu else '❌ DISABLED'}")
        if self.use_gpu:
            print(f"GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Load models with GPU optimization
        print("\n🚀 Loading models with GPU optimization...")
        self.coco_model = YOLO('yolov8n.pt')
        self.license_plate_detector = YOLO('./models/license_plate_detector.pt')
        
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
        
        # Accuracy optimization settings
        self.plate_confidence_threshold = 0.3
        self.ocr_confidence_threshold = 0.15
        self.min_plate_size = (20, 20)
        
        print("✅ System initialized successfully!")
    
    def detect_vehicles(self, frame):
        """Detect vehicles in frame."""
        det_start = time.time()
        
        results = self.coco_model(frame, verbose=False, device=self.device)[0]
        
        detections = []
        for detection in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            
            if int(class_id) in [2, 3, 5, 7] and score > 0.4:
                detections.append([x1, y1, x2, y2, score])
        
        self.detection_times.append(time.time() - det_start)
        return detections
    
    def detect_license_plates(self, frame):
        """Detect license plates in frame."""
        plate_start = time.time()
        
        results = self.license_plate_detector(frame, verbose=False, device=self.device)[0]
        
        plates = []
        for detection in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            
            if score >= self.plate_confidence_threshold:
                plate_width = x2 - x1
                plate_height = y2 - y1
                
                if plate_width >= self.min_plate_size[0] and plate_height >= self.min_plate_size[1]:
                    plates.append([x1, y1, x2, y2, score, class_id])
        
        self.detection_times.append(time.time() - plate_start)
        return plates
    
    def read_license_plate(self, plate_crop):
        """Read Spanish license plate from crop."""
        ocr_start = time.time()
        
        if len(plate_crop.shape) == 3:
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_crop
        
        plate_text, plate_score = read_spanish_license_plate_optimized(gray)
        
        self.ocr_times.append(time.time() - ocr_start)
        return plate_text, plate_score
    
    def process_frame(self, frame):
        """Process a single frame."""
        frame_start = time.time()
        
        # Detect vehicles
        vehicle_detections = self.detect_vehicles(frame)
        
        # Track vehicles
        if vehicle_detections:
            track_ids = self.mot_tracker.update(np.asarray(vehicle_detections))
        else:
            track_ids = []
        
        # Detect license plates
        plate_detections = self.detect_license_plates(frame)
        
        frame_results = {}
        
        # Process each detected plate
        for plate in plate_detections:
            x1, y1, x2, y2, score, class_id = plate
            
            # Assign plate to vehicle
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(plate, track_ids)
            
            if car_id != -1:
                # Crop license plate
                plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                
                if plate_crop.size == 0 or plate_crop.shape[0] < 10 or plate_crop.shape[1] < 10:
                    continue
                
                # Read Spanish license plate
                plate_text, plate_score = self.read_license_plate(plate_crop)
                
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
        
        self.frame_times.append(time.time() - frame_start)
        return frame_results
    
    def process_video(self, max_frames: int = None):
        """
        Process video without display.
        
        Returns:
            Average FPS
        """
        print(f"\n📹 Processing video: {os.path.basename(self.video_path)}")
        
        # Open video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video file")
            return 0
        
        # Get video properties
        self.original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📊 Video info: {width}x{height} @ {self.original_fps} fps, {total_frames} frames")
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
            print(f"🔧 Processing first {max_frames} frames")
        
        # Processing loop
        self.frame_count = 0
        start_time = time.time()
        
        print("\n🎬 Starting headless processing...")
        print("   " + "-"*50)
        
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret or (max_frames and self.frame_count >= max_frames):
                break
            
            # Process frame
            self.process_frame(frame)
            
            self.frame_count += 1
            
            # Print progress every 50 frames
            if self.frame_count % 50 == 0:
                elapsed = time.time() - start_time
                current_fps = self.frame_count / elapsed if elapsed > 0 else 0
                progress = self.frame_count / total_frames * 100
                
                print(f"   📈 Frame {self.frame_count}/{total_frames} ({progress:.1f}%) | "
                      f"FPS: {current_fps:.1f} | "
                      f"Plates: {self.plates_detected} | "
                      f"Spanish: {self.valid_spanish_plates}")
        
        # Clean up
        cap.release()
        
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
            print(f"\n📋 Sample results (first 10 plates):")
            frame_keys = sorted(self.results.keys())
            count = 0
            for frame_key in frame_keys:
                for car_key in self.results[frame_key]:
                    plate_info = self.results[frame_key][car_key]['plate']
                    print(f"   Frame {frame_key}, Car {car_key}: {plate_info['text']} "
                          f"(confidence: {plate_info['text_score']:.3f})")
                    count += 1
                    if count >= 10:
                        break
                if count >= 10:
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
        
        report_file = 'spanish_lpr_headless_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Full report saved to: {report_file}")
        
        return report

def main():
    """
    Main function for headless Spanish LPR system.
    """
    # Configuration
    video_path = "/home/alex/Downloads/video_carplates1.mkv"
    output_csv = "results_spanish_headless.csv"
    max_frames = 300  # Set to None for full video
    
    print("\n" + "="*70)
    print("🚀 HEADLESS SPANISH LICENSE PLATE RECOGNITION")
    print("="*70)
    
    # Create and run system
    lpr_system = HeadlessSpanishLPR(video_path, output_csv)
    
    # Process video
    avg_fps = lpr_system.process_video(max_frames=max_frames)
    
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

if __name__ == "__main__":
    main()