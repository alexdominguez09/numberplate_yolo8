"""
HEADLESS SPANISH LICENSE PLATE RECOGNITION SYSTEM - v1.02
IMPROVED: Better OCR accuracy, deduplication, and confidence filtering
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


class HeadlessSpanishLPR_v1_02:
    """
    Headless Spanish License Plate Recognition system v1.02
    With improved OCR accuracy and per-car deduplication.
    """
    
    def __init__(self, video_path: str, output_csv: str = "results_spanish_headless_v1.02.csv",
                 output_text_file: str = "unique_plates_v1.02.txt",
                 ground_truth_file: str = None, min_confidence: float = 0.40):
        """
        Headless Spanish License Plate Recognition system.
        
        Args:
            video_path: Path to input video file
            output_csv: Path to output CSV file
            output_text_file: Path to output text file with car_id, plate, score
            ground_truth_file: Path to ground truth text file
            min_confidence: Minimum OCR confidence to track (increased from 0.15 to 0.40)
        """
        self.video_path = video_path
        self.output_csv = output_csv
        self.output_text_file = output_text_file
        self.ground_truth_file = ground_truth_file
        self.min_confidence = min_confidence
        self.version = "1.02"
        
        # GPU configuration
        self.use_gpu = torch.cuda.is_available()
        self.device = 'cuda' if self.use_gpu else 'cpu'
        
        print(f"{'='*70}")
        print(f"HEADLESS SPANISH LICENSE PLATE RECOGNITION SYSTEM - v{self.version}")
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
        
        # NEW: Unique plates per car with deduplication
        self.best_results_per_car = {}  # car_id -> {text, confidence, frame}
        
        # Ground truth plates
        self.ground_truth_plates = []
        if self.ground_truth_file:
            self.ground_truth_plates = self.load_ground_truth(self.ground_truth_file)
            print(f"✅ Loaded {len(self.ground_truth_plates)} ground truth plates from {self.ground_truth_file}")
        
        # Performance tracking
        self.frame_times = []
        self.detection_times = []
        self.ocr_times = []
        
        # IMPROVED: Stricter accuracy optimization settings
        self.plate_confidence_threshold = 0.4  # Increased from 0.3
        self.ocr_confidence_threshold = self.min_confidence  # Increased to 0.40
        self.min_plate_size = (30, 15)  # Increased minimum plate size
        
        # Multiple OCR attempts per plate with different preprocessing
        self.ocr_attempts = 3  # Try OCR 3 times with different preprocessing
        
        print(f"⚙️  Configuration:")
        print(f"   - OCR confidence threshold: {self.ocr_confidence_threshold:.2f}")
        print(f"   - Plate confidence threshold: {self.plate_confidence_threshold:.2f}")
        print(f"   - Min plate size: {self.min_plate_size}")
        print(f"   - OCR attempts per plate: {self.ocr_attempts}")
        
        print("✅ System initialized successfully!")
    
    def load_ground_truth(self, filepath: str) -> list:
        """
        Load ground truth plates from text file.
        One plate per line, format: "8314 JSP" or "8314-JSP"
        """
        plates = []
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    plate = line.strip().upper()
                    if plate and len(plate) > 0:
                        # Normalize: add hyphen if missing
                        if '-' not in plate and len(plate) == 7:
                            # Try to infer format
                            if plate[:4].isdigit():
                                plate = f"{plate[:4]}-{plate[4:]}"
                            elif plate[:2].isalpha():
                                plate = f"{plate[:2]}-{plate[2:]}"
                        plates.append(plate)
        except Exception as e:
            print(f"❌ Error loading ground truth file: {e}")
        
        return plates
    
    def track_best_plate_per_car(self, car_id: int, plate_text: str, confidence: float):
        """
        Track the best OCR result for each car (highest confidence).
        """
        if car_id not in self.best_results_per_car:
            # First time seeing this car
            self.best_results_per_car[car_id] = {
                'text': plate_text,
                'confidence': confidence,
                'frame_count': self.frame_count,
            }
        else:
            # Update only if this result has higher confidence
            current_best = self.best_results_per_car[car_id]
            if confidence > current_best['confidence']:
                self.best_results_per_car[car_id] = {
                    'text': plate_text,
                    'confidence': confidence,
                    'frame_count': self.frame_count,
                }
    
    def calculate_character_accuracy(self, ocr_text: str, gt_text: str) -> float:
        """
        Calculate character-level accuracy between OCR and ground truth.
        """
        if not ocr_text or not gt_text:
            return 0.0
        
        # Remove hyphens for comparison
        ocr_clean = ocr_text.replace('-', '')
        gt_clean = gt_text.replace('-', '')
        
        # Calculate character matches
        matches = 0
        total_chars = max(len(ocr_clean), len(gt_clean))
        
        for i in range(min(len(ocr_clean), len(gt_clean))):
            if ocr_clean[i] == gt_clean[i]:
                matches += 1
        
        accuracy = matches / total_chars if total_chars > 0 else 0.0
        return accuracy
    
    def preprocess_plate_improved(self, plate_crop: np.ndarray) -> list:
        """
        Improved preprocessing for Spanish plates.
        Returns list of (processed_image, method_name)
        """
        methods = []
        
        # Convert to grayscale
        if len(plate_crop.shape) == 3:
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_crop.copy()
        
        # Method 1: Original (no preprocessing)
        methods.append((gray.copy(), 'original'))
        
        # Method 2: CLAHE + Gaussian blur
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        methods.append((blurred, 'clahe_blur'))
        
        # Method 3: Adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        methods.append((thresh, 'adaptive_thresh'))
        
        # Method 4: Otsu thresholding
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        methods.append((otsu, 'otsu_thresh'))
        
        # Method 5: Bilateral filter (edge-preserving denoising)
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        _, thresh_bilateral = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        methods.append((thresh_bilateral, 'bilateral_otsu'))
        
        return methods
    
    def read_license_plate_multi(self, plate_crop: np.ndarray) -> tuple:
        """
        Try multiple preprocessing methods and return best result.
        """
        methods = self.preprocess_plate_improved(plate_crop)
        
        best_text = ""
        best_score = 0.0
        best_method = ""
        
        for processed_img, method_name in methods:
            text, score = read_spanish_license_plate_optimized(processed_img)
            
            # Check format validity
            is_valid_spanish = '-' in text and len(text.replace('-', '')) >= 5
            
            if text and score >= best_score:
                best_text = text
                best_score = score
                best_method = method_name
        
        return best_text, best_score, best_method
    
    def is_valid_spanish_format(self, text: str) -> bool:
        """
        Check if text matches Spanish plate format.
        Current: ####-LLL (e.g., 8314-JSP)
        Old: LL-#### (e.g., AB-1234)
        """
        if not text or len(text) < 7:
            return False
        
        # Check for hyphen
        if '-' not in text:
            return False
        
        parts = text.split('-')
        if len(parts) != 2:
            return False
        
        part1, part2 = parts
        
        # Current format: ####-LLL
        if len(part1) == 4 and part1.isdigit() and len(part2) == 3 and part2.isalpha():
            return True
        
        # Old format: LL-####
        if len(part1) == 2 and part1.isalpha() and len(part2) == 4 and part2.isdigit():
            return True
        
        return False
    
    def detect_vehicles(self, frame):
        """Detect vehicles in frame."""
        det_start = time.time()
        
        results = self.coco_model(frame, verbose=False, device=self.device)[0]
        
        detections = []
        for detection in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            
            # Stricter filtering for better results
            if int(class_id) in [2, 3, 5, 7] and score > 0.5:  # Increased threshold
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
            
            # Apply confidence threshold
            if score >= self.plate_confidence_threshold:
                # Check minimum size
                plate_width = x2 - x1
                plate_height = y2 - y1
                
                # Stricter size filter
                aspect_ratio = plate_width / plate_height if plate_height > 0 else 0
                
                if plate_width >= self.min_plate_size[0] and plate_height >= self.min_plate_size[1]:
                    # Check aspect ratio (Spanish plates are typically 2:1 to 4:1)
                    if 2.0 <= aspect_ratio <= 4.5:
                        plates.append([x1, y1, x2, y2, score, class_id])
        
        self.detection_times.append(time.time() - plate_start)
        return plates
    
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
                # Crop license plate with padding
                pad = 5
                y1_pad = max(0, int(y1) - pad)
                y2_pad = min(frame.shape[0], int(y2) + pad)
                x1_pad = max(0, int(x1) - pad)
                x2_pad = min(frame.shape[1], int(x2) + pad)
                
                plate_crop = frame[y1_pad:y2_pad, x1_pad:x2_pad, :]
                
                if plate_crop.size == 0 or plate_crop.shape[0] < 20 or plate_crop.shape[1] < 60:
                    continue
                
                # IMPROVED: Read Spanish license plate with multiple preprocessing methods
                plate_text, plate_score, ocr_method = self.read_license_plate_multi(plate_crop)
                
                # Only track if confidence is above threshold
                if plate_text and plate_score >= self.ocr_confidence_threshold:
                    # IMPROVED: Check Spanish format before accepting
                    if self.is_valid_spanish_format(plate_text):
                        self.plates_detected += 1
                        self.valid_spanish_plates += 1
                        
                        # Track best plate per car (deduplication)
                        self.track_best_plate_per_car(car_id, plate_text, plate_score)
                        
                        # Store frame results
                        frame_results[car_id] = {
                            'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                            'plate': {
                                'bbox': [x1, y1, x2, y2],
                                'bbox_score': score,
                                'text': plate_text,
                                'text_score': plate_score,
                                'ocr_method': ocr_method
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
                
                unique_cars = len(self.best_results_per_car)
                
                print(f"   📈 Frame {self.frame_count}/{total_frames} ({progress:.1f}%) | "
                      f"FPS: {current_fps:.1f} | "
                      f"Plates: {self.plates_detected} | "
                      f"Unique Cars: {unique_cars} | "
                      f"Valid: {self.valid_spanish_plates}")
        
        # Clean up
        cap.release()
        
        # Calculate final statistics
        elapsed_time = time.time() - start_time
        avg_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        return avg_fps
    
    def compare_plates(self) -> dict:
        """
        Compare unique OCR results with ground truth plates.
        """
        if not self.ground_truth_plates:
            print("⚠️  No ground truth provided, skipping accuracy comparison")
            return None
        
        # Get unique OCR results (best per car)
        unique_ocr_results = list(self.best_results_per_car.values())
        ocr_plates = [result['text'] for result in unique_ocr_results]
        
        print(f"\n📊 Comparing {len(ocr_plates)} unique OCR results with {len(self.ground_truth_plates)} ground truth plates")
        
        # Initialize comparison results
        comparison = {
            'exact_matches': [],
            'partial_matches': [],
            'no_matches': [],
            'ocr_plates': ocr_plates,
            'ground_truth_plates': self.ground_truth_plates,
            'character_accuracies': []
        }
        
        # Compare each OCR result with ground truth
        for ocr_idx, ocr_plate in enumerate(ocr_plates):
            matched = False
            match_type = None
            char_accuracy = 0.0
            
            # Try exact match first
            for gt_plate in self.ground_truth_plates:
                if ocr_plate == gt_plate:
                    comparison['exact_matches'].append({
                        'ocr_plate': ocr_plate,
                        'ground_truth': gt_plate,
                        'confidence': unique_ocr_results[ocr_idx]['confidence']
                    })
                    matched = True
                    match_type = 'exact'
                    break
            
            # If no exact match, try character-level accuracy
            if not matched:
                for gt_plate in self.ground_truth_plates:
                    char_accuracy = self.calculate_character_accuracy(ocr_plate, gt_plate)
                    if char_accuracy >= 0.857:  # 6/7 characters match for partial
                        comparison['partial_matches'].append({
                            'ocr_plate': ocr_plate,
                            'ground_truth': gt_plate,
                            'confidence': unique_ocr_results[ocr_idx]['confidence'],
                            'character_accuracy': char_accuracy
                        })
                        matched = True
                        match_type = 'partial'
                        comparison['character_accuracies'].append(char_accuracy)
                        break
            
            # If still no match, record as no match
            if not matched:
                comparison['no_matches'].append({
                    'ocr_plate': ocr_plate,
                    'confidence': unique_ocr_results[ocr_idx]['confidence']
                })
        
        return comparison
    
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
        print(f"   Unique cars detected: {len(self.best_results_per_car)}")
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
        
        # Generate accuracy comparison
        comparison = self.compare_plates()
        
        if comparison:
            # Show sample results
            print(f"\n📋 Sample results (first 5 unique plates per car):")
            unique_ocr_results = list(self.best_results_per_car.values())
            count = 0
            for result in sorted(unique_ocr_results, key=lambda x: x['confidence'], reverse=True)[:5]:
                car_id = [k for k, v in self.best_results_per_car.items() if v == result][0]
                print(f"   Car {car_id}: {result['text']} (confidence: {result['confidence']:.3f})")
                count += 1
        
        # Generate JSON report
        report = {
            'timestamp': datetime.now().isoformat(),
            'video_file': os.path.basename(self.video_path),
            'frames_processed': self.frame_count,
            'plates_detected': self.plates_detected,
            'valid_spanish_plates': self.valid_spanish_plates,
            'unique_cars_detected': len(self.best_results_per_car),
            'average_fps': avg_fps,
            'gpu_acceleration': self.use_gpu,
            'gpu_device': torch.cuda.get_device_name(0) if self.use_gpu else None,
            'output_file': self.output_csv,
            'processing_time_seconds': self.frame_count/avg_fps if avg_fps > 0 else 0,
            'plates_per_frame': self.plates_detected/self.frame_count if self.frame_count > 0 else 0,
            'spanish_plate_accuracy': self.valid_spanish_plates/self.plates_detected if self.plates_detected > 0 else 0,
            'version': self.version
        }
        
        report_file = f'spanish_lpr_headless_v{self.version.replace(".", "")}_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Full report saved to: {report_file}")
        
        return report
    
    def write_unique_plates_file(self):
        """
        Write unique plates to text file with format: car_id    license_nmb    license_nmb_score
        """
        print(f"\n💾 Writing unique plates to: {self.output_text_file}")
        
        with open(self.output_text_file, 'w') as f:
            f.write("car_id\tlicense_nmb\tlicense_nmb_score\n")
            
            # Sort by first appearance (frame_count) - NOT by confidence
            sorted_results = sorted(self.best_results_per_car.items(), 
                                  key=lambda x: x[1]['frame_count'])
            
            for car_id, result in sorted_results:
                f.write(f"{car_id}\t{result['text']}\t{result['confidence']:.5f}\n")
        
        print(f"   Wrote {len(self.best_results_per_car)} unique plates")


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Headless Spanish License Plate Recognition System v1.02 - IMPROVED'
    )
    parser.add_argument('--video', '-v',
                        type=str,
                        required=True,
                        help='Path to input video file')
    parser.add_argument('--output', '-o',
                        type=str,
                        default='results_spanish_headless_v1.02.csv',
                        help='Output CSV file path')
    parser.add_argument('--plates-file', '-pf',
                        type=str,
                        default='unique_plates_v1.02.txt',
                        help='Output text file path with car_id, license_nmb, score')
    parser.add_argument('--max-frames', '-m',
                        type=int,
                        default=None,
                        help='Maximum frames to process (None for full video)')
    parser.add_argument('--ground-truth', '-gt',
                        type=str,
                        default=None,
                        help='Path to ground truth text file (one plate per line)')
    parser.add_argument('--min-confidence', '-mc',
                        type=float,
                        default=0.40,
                        help='Minimum OCR confidence to track (default: 0.40)')
    return parser.parse_args()


def main():
    """
    Main function for headless Spanish LPR system v1.02.
    """
    # Parse arguments
    args = parse_args()
    
    # Configuration from arguments
    video_path = args.video
    output_csv = args.output
    output_text_file = args.plates_file
    max_frames = args.max_frames
    ground_truth_file = args.ground_truth
    min_confidence = args.min_confidence
    
    print("\n" + "=" * 70)
    print(f"🚀 HEADLESS SPANISH LPR - v1.02 (IMPROVED)")
    print("=" * 70)
    print(f"Video: {video_path}")
    if ground_truth_file:
        print(f"Ground Truth: {ground_truth_file}")
    print(f"Min Confidence: {min_confidence}")
    print(f"Plates File: {output_text_file}")
    print("=" * 70)
    
    # Create and run system
    lpr_system = HeadlessSpanishLPR_v1_02(
        video_path=video_path,
        output_csv=output_csv,
        output_text_file=output_text_file,
        ground_truth_file=ground_truth_file,
        min_confidence=min_confidence
    )
    
    # Process video
    avg_fps = lpr_system.process_video(max_frames=max_frames)
    
    # Generate report
    report = lpr_system.generate_report(avg_fps)
    
    # Write unique plates to text file
    lpr_system.write_unique_plates_file()
    
    print(f"\n{'='*70}")
    print("🎉 SYSTEM EXECUTION COMPLETE!")
    print(f"{'='*70}")
    
    # Summary
    print(f"\n📈 SUMMARY:")
    print(f"   • Version: v{report.get('version', '1.02')}")
    print(f"   • Frames processed: {report['frames_processed']}")
    print(f"   • Plates detected: {report['plates_detected']}")
    print(f"   • Unique cars: {report['unique_cars_detected']}")
    print(f"   • Average FPS: {report['average_fps']:.1f}")
    print(f"   • GPU: {'Enabled' if report['gpu_acceleration'] else 'Disabled'}")
    print(f"   • CSV results: {report['output_file']}")
    print(f"   • Plates file: {output_text_file}")
    
    if report.get('ground_truth_file'):
        # Get comparison from memory
        comparison = lpr_system.compare_plates()
        if comparison:
            exact_matches = len(comparison['exact_matches'])
            partial_matches = len(comparison['partial_matches'])
            total_matches = exact_matches + partial_matches
            total_ocr = len(comparison['ocr_plates'])
            
            print(f"   • Ground truth: {report['ground_truth_count']} plates")
            print(f"   • Exact matches: {exact_matches}")
            print(f"   • Partial matches: {partial_matches}")
            print(f"   • Exact match rate: {exact_matches/total_ocr*100:.1f}%")
            print(f"   • Total match rate: {total_matches/total_ocr*100:.1f}%")
    
    print(f"\n✅ Ready for production with improved accuracy!")


if __name__ == "__main__":
    main()
