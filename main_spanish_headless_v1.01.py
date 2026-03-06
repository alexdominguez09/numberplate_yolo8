"""
HEADLESS SPANISH LICENSE PLATE RECOGNITION SYSTEM - v1.01
GPU accelerated, no display required, with OCR accuracy tracking.
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

class HeadlessSpanishLPR_v1_01:
    """
    Headless Spanish License Plate Recognition system v1.01
    With ground truth comparison and accuracy tracking.
    """
    
    def __init__(self, video_path: str, output_csv: str = "results_spanish_headless_v1.01.csv",
                 ground_truth_file: str = None, min_confidence: float = 0.15):
        """
        Headless Spanish License Plate Recognition system.
        
        Args:
            video_path: Path to input video file
            output_csv: Path to output CSV file
            ground_truth_file: Path to ground truth text file
            min_confidence: Minimum OCR confidence to track
        """
        self.video_path = video_path
        self.output_csv = output_csv
        self.ground_truth_file = ground_truth_file
        self.min_confidence = min_confidence
        self.version = "1.01"
        
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
        
        # NEW: Best results tracking per car (deduplication)
        self.best_results_per_car = {}
        
        # NEW: Ground truth plates
        self.ground_truth_plates = []
        if self.ground_truth_file:
            self.ground_truth_plates = self.load_ground_truth(self.ground_truth_file)
            print(f"✅ Loaded {len(self.ground_truth_plates)} ground truth plates from {self.ground_truth_file}")
        
        # Performance tracking
        self.frame_times = []
        self.detection_times = []
        self.ocr_times = []
        
        # Accuracy optimization settings
        self.plate_confidence_threshold = 0.3
        self.ocr_confidence_threshold = self.min_confidence
        self.min_plate_size = (20, 20)
        
        print("✅ System initialized successfully!")
    
    def load_ground_truth(self, filepath: str) -> list:
        """
        Load ground truth plates from text file.
        One plate per line, uppercase, stripped.
        """
        plates = []
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    plate = line.strip().upper()
                    # Skip empty lines and comments
                    if plate and not plate.startswith('#'):
                        plates.append(plate)
            print(f"📋 Loaded {len(plates)} ground truth plates")
            return plates
        except FileNotFoundError:
            print(f"⚠️  Warning: Ground truth file not found: {filepath}")
            return []
        except Exception as e:
            print(f"❌ Error loading ground truth file: {e}")
            return []
    
    def track_best_plate_per_car(self, car_id: int, plate_text: str, confidence: float):
        """
        Track the best OCR result for each car (highest confidence).
        """
        if car_id not in self.best_results_per_car:
            # First time seeing this car
            self.best_results_per_car[car_id] = {
                'text': plate_text,
                'confidence': confidence,
                'frame_count': self.frame_count
            }
        else:
            # Update if this result has higher confidence
            current_best = self.best_results_per_car[car_id]
            if confidence > current_best['confidence']:
                self.best_results_per_car[car_id] = {
                    'text': plate_text,
                    'confidence': confidence,
                    'frame_count': self.frame_count
                }
    
    def calculate_character_accuracy(self, ocr_text: str, gt_text: str) -> float:
        """
        Calculate character-level accuracy between OCR and ground truth.
        
        Args:
            ocr_text: OCR result (e.g., "1234-ABO")
            gt_text: Ground truth (e.g., "1234-ABC")
        
        Returns:
            Character accuracy (0.0 to 1.0)
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
    
    def compare_plates(self) -> dict:
        """
        Compare unique OCR results with ground truth plates.
        
        Returns:
            dict with comparison results
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
                    if char_accuracy >= 0.7:  # 70% character accuracy threshold for partial match
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
    
    def detect_vehicles(self, frame):
        """Detect vehicles in frame."""
        det_start = time.time()
        
        results = self.coco_model(frame, verbose=False, device=self.device)[0]
        
        detections = []
        for detection in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            
            if int(class_id) in [2, 3, 5, 7] and score > 0.4:  # Cars, trucks, motorcycles, buses
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
                    
                    # NEW: Track best plate per car
                    self.track_best_plate_per_car(car_id, plate_text, plate_score)
                    
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
    
    def generate_accuracy_report(self, comparison: dict) -> str:
        """
        Generate text accuracy report.
        """
        if not comparison:
            return ""
        
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append(f"OCR ACCURACY REPORT - v{self.version}")
        report_lines.append("=" * 70)
        report_lines.append(f"Video: {os.path.basename(self.video_path)}")
        report_lines.append(f"Ground Truth File: {self.ground_truth_file}")
        report_lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Summary
        report_lines.append("SUMMARY:")
        report_lines.append("-" * 70)
        report_lines.append(f"Ground Truth Plates: {len(comparison['ground_truth_plates'])}")
        report_lines.append(f"Unique OCR Results: {len(comparison['ocr_plates'])}")
        report_lines.append("")
        
        # Matching results
        report_lines.append("MATCHING RESULTS:")
        report_lines.append("-" * 70)
        exact_count = len(comparison['exact_matches'])
        partial_count = len(comparison['partial_matches'])
        no_match_count = len(comparison['no_matches'])
        
        report_lines.append(f"- Exact Matches: {exact_count}")
        report_lines.append(f"- Partial Matches: {partial_count}")
        report_lines.append(f"- No Matches: {no_match_count}")
        report_lines.append("")
        
        # Accuracy metrics
        total_ocr = len(comparison['ocr_plates'])
        exact_rate = exact_count / total_ocr if total_ocr > 0 else 0
        partial_rate = (exact_count + partial_count) / total_ocr if total_ocr > 0 else 0
        total_rate = (exact_count + partial_count) / total_ocr if total_ocr > 0 else 0
        
        report_lines.append("ACCURACY METRICS:")
        report_lines.append("-" * 70)
        report_lines.append(f"- Exact Match Rate: {exact_rate*100:.2f}%")
        report_lines.append(f"- Partial Match Rate: {partial_rate*100:.2f}%")
        report_lines.append(f"- Total Match Rate: {total_rate*100:.2f}%")
        report_lines.append(f"- False Positive Rate: {no_match_count/total_ocr*100:.2f}%")
        report_lines.append("")
        
        # Character-level accuracy
        if comparison['character_accuracies']:
            avg_char_acc = np.mean(comparison['character_accuracies'])
            report_lines.append("CHARACTER-LEVEL ACCURACY:")
            report_lines.append("-" * 70)
            report_lines.append(f"- Overall Character Accuracy: {avg_char_acc*100:.1f}%")
            
            if len(comparison['character_accuracies']) > 0:
                report_lines.append(f"- Min Accuracy: {min(comparison['character_accuracies'])*100:.1f}%")
                report_lines.append(f"- Max Accuracy: {max(comparison['character_accuracies'])*100:.1f}%")
        report_lines.append("")
        
        # Confidence analysis
        if comparison['exact_matches']:
            confidences_correct = [m['confidence'] for m in comparison['exact_matches']]
            avg_conf_correct = np.mean(confidences_correct) if confidences_correct else 0
            report_lines.append(f"CONFIDENCE ANALYSIS:")
            report_lines.append("-" * 70)
            report_lines.append(f"- Avg Confidence (Correct): {avg_conf_correct:.3f}")
        
        if comparison['no_matches']:
            confidences_incorrect = [m['confidence'] for m in comparison['no_matches']]
            avg_conf_incorrect = np.mean(confidences_incorrect) if confidences_incorrect else 0
            if confidences_incorrect:
                report_lines.append(f"- Avg Confidence (Incorrect): {avg_conf_incorrect:.3f}")
        report_lines.append("")
        
        # Error patterns
        error_patterns = {}
        all_matches = comparison['exact_matches'] + comparison['partial_matches']
        for match in all_matches:
            if match['character_accuracy'] < 1.0:
                for i, (ocr_char, gt_char) in enumerate(zip(match['ocr_plate'].replace('-', ''), 
                                                               match['ground_truth'].replace('-', ''))):
                    if ocr_char != gt_char:
                        error = f"{ocr_char}→{gt_char}"
                        error_patterns[error] = error_patterns.get(error, 0) + 1
        
        if error_patterns:
            report_lines.append("ERROR PATTERNS:")
            report_lines.append("-" * 70)
            sorted_errors = sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)
            for error, count in sorted_errors[:10]:  # Top 10 errors
                report_lines.append(f"- {error}: {count} occurrences ({count/len(all_matches)*100:.1f}%)")
        
        report_lines.append("=" * 70)
        
        return "\n".join(report_lines)
    
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
        print("   " + "-" * 50)
        
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
                      f"Plates: {self.plates_detected}")
        
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
            print(f"\n📋 Sample results (first 5 unique plates per car):")
            count = 0
            for car_id, result in self.best_results_per_car.items():
                if count >= 5:
                    break
                print(f"   Car {car_id}: {result['text']} (confidence: {result['confidence']:.3f})")
                count += 1
        
        # NEW: Compare with ground truth and generate accuracy report
        if self.ground_truth_plates and self.best_results_per_car:
            comparison = self.compare_plates()
            
            # Generate text accuracy report
            accuracy_report = self.generate_accuracy_report(comparison)
            
            # Save accuracy report to file
            accuracy_report_file = f"accuracy_report_v{self.version}.txt"
            with open(accuracy_report_file, 'w') as f:
                f.write(accuracy_report)
            print(f"\n📄 Accuracy report saved to: {accuracy_report_file}")
            
            # Print accuracy report to console
            print("\n" + accuracy_report)
        else:
            if not self.ground_truth_plates:
                print(f"\n⚠️  No ground truth file provided. Accuracy comparison skipped.")
            else:
                print(f"\n⚠️  No valid OCR results found. Accuracy comparison skipped.")
        
        # Generate JSON report
        report_data = {
            'version': self.version,
            'timestamp': datetime.now().isoformat(),
            'video_file': os.path.basename(self.video_path),
            'ground_truth_file': os.path.basename(self.ground_truth_file) if self.ground_truth_file else None,
            'frames_processed': self.frame_count,
            'plates_detected': self.plates_detected,
            'valid_spanish_plates': self.valid_spanish_plates,
            'unique_ocr_results': len(self.best_results_per_car),
            'ground_truth_count': len(self.ground_truth_plates),
            'average_fps': avg_fps,
            'gpu_acceleration': self.use_gpu,
            'gpu_device': torch.cuda.get_device_name(0) if self.use_gpu else None,
            'output_file': self.output_csv,
            'processing_time_seconds': self.frame_count/avg_fps if avg_fps > 0 else 0,
            'plates_per_frame': self.plates_detected/self.frame_count if self.frame_count > 0 else 0,
            'min_confidence_threshold': self.min_confidence
        }
        
        # Add accuracy metrics if available
        if self.ground_truth_plates and self.best_results_per_car:
            comparison = self.compare_plates()
            report_data.update({
                'exact_matches': len(comparison['exact_matches']),
                'partial_matches': len(comparison['partial_matches']),
                'no_matches': len(comparison['no_matches']),
                'exact_match_rate': len(comparison['exact_matches'])/len(comparison['ocr_plates']),
                'partial_match_rate': len(comparison['partial_matches'])/len(comparison['ocr_plates']),
                'total_match_rate': (len(comparison['exact_matches']) + len(comparison['partial_matches']))/len(comparison['ocr_plates']),
                'false_positive_rate': len(comparison['no_matches'])/len(comparison['ocr_plates']),
                'character_accuracy': np.mean(comparison['character_accuracies']) if comparison['character_accuracies'] else 0
            })
        
        report_file = f'spanish_lpr_headless_v{self.version}_report.json'
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n📄 Full report saved to: {report_file}")
        
        return report_data
    
    def parse_args():
        """
        Parse command-line arguments.
        """
        parser = argparse.ArgumentParser(
            description='Headless Spanish License Plate Recognition System v1.01 - with OCR accuracy tracking'
        )
        parser.add_argument('--video', '-v',
                            type=str,
                            required=True,
                            help='Path to input video file')
        parser.add_argument('--output', '-o',
                            type=str,
                            default='results_spanish_headless_v1.01.csv',
                            help='Output CSV file path')
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
                            default=0.15,
                            help='Minimum OCR confidence to track (default: 0.15)')
        return parser.parse_args()


def main():
    """
    Main function for headless Spanish LPR system v1.01.
    """
    # Parse arguments
    args = parse_args()
    
    # Configuration from arguments
    video_path = args.video
    output_csv = args.output
    max_frames = args.max_frames
    ground_truth_file = args.ground_truth
    min_confidence = args.min_confidence
    
    print("\n" + "=" * 70)
    print(f"🚀 HEADLESS SPANISH LPR - v1.01")
    print("=" * 70)
    print(f"Video: {video_path}")
    if ground_truth_file:
        print(f"Ground Truth: {ground_truth_file}")
    print(f"Min Confidence: {min_confidence}")
    print("=" * 70)
    
    # Create and run system
    lpr_system = HeadlessSpanishLPR_v1_01(
        video_path=video_path,
        output_csv=output_csv,
        ground_truth_file=ground_truth_file,
        min_confidence=min_confidence
    )
    
    # Process video
    avg_fps = lpr_system.process_video(max_frames=max_frames)
    
    # Generate report
    report = lpr_system.generate_report(avg_fps)
    
    print(f"\n{'='*70}")
    print("🎉 SYSTEM EXECUTION COMPLETE!")
    print(f"{'='*70}")
    
    # Summary
    print(f"\n📈 SUMMARY:")
    print(f"   • Version: v{report.get('version', '1.00')}")
    print(f"   • Frames processed: {report['frames_processed']}")
    print(f"   • Plates detected: {report['plates_detected']}")
    print(f"   • Unique OCR results: {report.get('unique_ocr_results', 0)}")
    print(f"   • Average FPS: {report['average_fps']:.1f}")
    print(f"   • GPU: {'Enabled' if report['gpu_acceleration'] else 'Disabled'}")
    print(f"   • Results file: {report['output_file']}")
    if report.get('ground_truth_file'):
        print(f"   • Ground truth: {report['ground_truth_count']} plates")
        print(f"   • Exact matches: {report.get('exact_matches', 0)}")
        print(f"   • Partial matches: {report.get('partial_matches', 0)}")
        print(f"   • Exact match rate: {report.get('exact_match_rate', 0)*100:.1f}%")
    
    print(f"\n✅ Ready for OCR accuracy analysis!")


if __name__ == "__main__":
    main()
