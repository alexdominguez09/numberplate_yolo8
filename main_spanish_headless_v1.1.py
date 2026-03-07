"""
HEADLESS SPANISH LICENSE PLATE RECOGNITION SYSTEM - v1.1
IMPROVED: Fixed duplicate plate reporting, ground-truth-first matching, Hungarian-style assignment
Uses config.py for configurable settings.
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
import re
from collections import Counter

# Import configuration settings
from config import (
    YOLO_IMGSZ,
    VEHICLE_CONF_THRESHOLD,
    PLATE_CONF_THRESHOLD,
    OCR_UPSCALE_FACTOR,
    OCR_MIN_CONFIDENCE,
    MIN_PLATE_SIZE,
    TRACKER_TYPE,
    TEMPORAL_VOTING_ENABLED,
    OCR_CORRECTION_ENABLED
)


class HeadlessSpanishLPR_v1_1:
    """
    Headless Spanish License Plate Recognition system v1.1
    With fixed duplicate plate reporting and ground-truth-first matching.
    """
    
    def __init__(self, video_path: str, output_csv: str = "results_spanish_headless.csv",
                 output_text_file: str = "unique_plates.txt",
                 ground_truth_file: str = None, min_confidence: float = 0.40,
                 enable_display: bool = False, enable_display_plate: bool = False):
        """
        Headless Spanish License Plate Recognition system.
        
        Args:
            video_path: Path to input video file
            output_csv: Path to output CSV file
            output_text_file: Path to output text file with car_id, plate, score
            ground_truth_file: Path to ground truth text file
            min_confidence: Minimum OCR confidence to track (increased from 0.15 to 0.40)
            enable_display: Enable cv2.imshow display window
            enable_display_plate: Display cropped license plate before OCR
        """
        self.video_path = video_path
        self.output_csv = output_csv
        self.output_text_file = output_text_file
        self.ground_truth_file = ground_truth_file
        self.min_confidence = min_confidence
        self.enable_display = enable_display
        self.enable_display_plate = enable_display_plate
        self.version = "1.1"
        
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
        
        # Load single fine-tuned model (detects vehicles + license plates)
        print("\n🚀 Loading fine-tuned YOLO model...")
        self.model = YOLO('./models/yolov8n_license_plate.pt')
        
        if self.use_gpu:
            self.model.to(self.device)
        
        # Define classes to detect: 2=car, 3=motorcycle, 5=bus, 7=truck, 80=license_plate
        self.vehicle_classes = [2, 3, 5, 7]
        self.plate_class = 80
        
        # Initialize tracker based on config
        self.tracker_type = TRACKER_TYPE
        if self.tracker_type == "botsort":
            # BoT-SORT: Use YOLOv8 built-in tracking (no separate tracker needed)
            self.mot_tracker = None
            self.frame_count = 0  # For persist tracking
        else:
            # SORT: Use traditional SORT tracker
            self.mot_tracker = Sort()
        
        # Results and statistics
        self.results = {}
        self.frame_count = 0
        self.plates_detected = 0
        self.valid_spanish_plates = 0
        
        # NEW: Temporal voting - store ALL reads per car for majority voting
        self.car_plate_reads = {}  # car_id -> {'reads': [all_texts], 'max_confidence': float}
        
        # Ground truth plates
        self.ground_truth_plates = []
        if self.ground_truth_file:
            self.ground_truth_plates = self.load_ground_truth(self.ground_truth_file)
            print(f"✅ Loaded {len(self.ground_truth_plates)} ground truth plates from {self.ground_truth_file}")
        
        # Performance tracking
        self.frame_times = []
        self.detection_times = []
        self.ocr_times = []
        
        # Load settings from configuration file
        self.plate_confidence_threshold = PLATE_CONF_THRESHOLD
        self.ocr_confidence_threshold = self.min_confidence  # Can be overridden via CLI
        self.min_plate_size = MIN_PLATE_SIZE
        self.yolo_imgsz = YOLO_IMGSZ
        self.ocr_upscale_factor = OCR_UPSCALE_FACTOR
        self.temporal_voting_enabled = TEMPORAL_VOTING_ENABLED
        self.ocr_correction_enabled = OCR_CORRECTION_ENABLED
        
        # Multiple OCR attempts per plate with different preprocessing
        self.ocr_attempts = 3
        
        print(f"⚙️  Configuration:")
        print(f"   - YOLO inference size: {self.yolo_imgsz}")
        print(f"   - Vehicle conf threshold: {VEHICLE_CONF_THRESHOLD:.2f}")
        print(f"   - Plate conf threshold: {self.plate_confidence_threshold:.2f}")
        print(f"   - OCR confidence threshold: {self.ocr_confidence_threshold:.2f}")
        print(f"   - OCR upscale factor: {self.ocr_upscale_factor}x")
        print(f"   - Min plate size: {self.min_plate_size}")
        print(f"   - Tracker type: {self.tracker_type}")
        print(f"   - Temporal voting: {self.temporal_voting_enabled}")
        print(f"   - OCR correction: {self.ocr_correction_enabled}")
        print(f"   - OCR attempts per plate: {self.ocr_attempts}")
        
        print("✅ System initialized successfully!")
        
        # Display settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2
        self.colors = {
            'car': (0, 255, 0),           # Green for vehicle
            'track_id': (255, 0, 0),       # Blue for car ID
            'plate': (0, 165, 255),        # Orange for plate bbox
            'plate_text': (0, 255, 255),   # Yellow for plate text
            'info': (255, 255, 255),      # White for info
            'success': (0, 255, 0),        # Green for success
        }
    
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
    
    def create_visualization(self, frame, vehicle_detections, plate_detections, tracked_cars, frame_results):
        """
        Create visualization with bounding boxes and information.
        """
        viz_frame = frame.copy()
        height, width = viz_frame.shape[:2]
        
        # Draw vehicle bounding boxes
        for vehicle in vehicle_detections:
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
        for plate in plate_detections:
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
            cv2.rectangle(viz_frame,
                         (x1, text_bg_y1),
                         (x1 + text_width + 10, y1),
                         self.colors['plate_text'], -1)
            
            # Text
            cv2.putText(viz_frame, text_label,
                       (x1 + 5, y1 - 5),
                       self.font, self.font_scale, (0, 0, 0),
                       self.font_thickness)
        
        # Add information overlay
        info_y = 30
        info_lines = [
            f"HEADLESS LPR v{self.version} | GPU: {'ON' if self.use_gpu else 'OFF'}",
            f"Frame: {self.frame_count} | FPS: {self.current_fps:.1f}" if hasattr(self, 'current_fps') else f"Frame: {self.frame_count}",
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
    
    def track_plate_read(self, car_id: int, plate_text: str, confidence: float):
        """
        Track ALL OCR reads for each car (for temporal voting).
        Store each read in a list for majority voting.
        """
        # Store original for display, clean version for voting
        original_text = plate_text.upper().strip()
        clean_plate = re.sub(r'[^A-Z0-9]', '', original_text)
        
        if car_id not in self.car_plate_reads:
            # First time seeing this car
            self.car_plate_reads[car_id] = {
                'reads': [clean_plate],
                'original_reads': [original_text],  # Keep original formatting
                'max_confidence': confidence,
                'first_frame': self.frame_count,
                'last_frame': self.frame_count,
            }
        else:
            # Append read and update tracking info
            self.car_plate_reads[car_id]['reads'].append(clean_plate)
            self.car_plate_reads[car_id]['original_reads'].append(original_text)
            if confidence > self.car_plate_reads[car_id]['max_confidence']:
                self.car_plate_reads[car_id]['max_confidence'] = confidence
            self.car_plate_reads[car_id]['last_frame'] = self.frame_count

    def format_spanish_plate(self, clean_text: str) -> str:
        """
        Format a cleaned plate text into proper Spanish plate format.
        """
        if not clean_text:
            return ""
        
        # Modern format: 4 digits + 3 letters (e.g., 8314JSP -> 8314-JSP)
        if re.match(r'^\d{4}[B-DF-HJ-NP-TV-Z]{3}$', clean_text):
            return f"{clean_text[:4]}-{clean_text[4:]}"
        
        # Provincial format: 1-2 letters + 4 digits + 1-2 letters
        # e.g., M1485ZX -> M-1485-ZX, MU1234AB -> MU-1234-AB
        if re.match(r'^[A-Z]{1,2}\d{4}[A-Z]{1,2}$', clean_text):
            # Find where the letters end and where they start after the numbers
            prefix_len = 0
            for i, c in enumerate(clean_text):
                if c.isdigit():
                    prefix_len = i
                    break
            suffix_len = len(clean_text) - prefix_len - 4
            return f"{clean_text[:prefix_len]}-{clean_text[prefix_len:prefix_len+4]}-{clean_text[prefix_len+4:]}"
        
        # Very old format: 1-2 letters + 4-6 digits (e.g., AB1234 -> AB-1234)
        if re.match(r'^[A-Z]{1,2}\d{4,6}$', clean_text):
            prefix_len = 0
            for i, c in enumerate(clean_text):
                if c.isdigit():
                    prefix_len = i
                    break
            return f"{clean_text[:prefix_len]}-{clean_text[prefix_len:]}"
        
        # If no match, return cleaned version with hyphen guess
        if len(clean_text) == 7 and clean_text[:4].isdigit():
            return f"{clean_text[:4]}-{clean_text[4:]}"
        
        return clean_text
    
    def correct_ocr_order(self, text: str) -> str:
        """
        Attempt to correct common OCR character order errors.
        Only applies corrections if the original text is NOT already valid.
        Spanish plates follow specific patterns, so we can detect misreads.
        """
        if not text or len(text) < 5:
            return text
        
        # First, check if the original text is already valid - if so, don't change it
        if self.is_valid_spanish_format(text):
            return text
            
        # Only attempt corrections if original is invalid
        
        # Pattern 1: 4 digits + 3 letters could be provincial (LNNNNLL) read incorrectly
        # e.g., 1485MZX -> M1485ZX
        if len(text) == 7 and text[:4].isdigit() and text[4:].isalpha():
            # Try moving first letter to front (LNNNNLL from NNNNLLL)
            potential_correct = text[4] + text[:4] + text[5:]
            if self.is_valid_spanish_format(potential_correct):
                return potential_correct
            
            # Try moving first 2 letters to front (LLNNNNLL from NNNNLLLL)
            if len(text[4:]) >= 2:
                potential_correct2 = text[4:6] + text[:4] + text[6:]
                if self.is_valid_spanish_format(potential_correct2):
                    return potential_correct2
        
        # Pattern 2: 3 letters + 4 digits could be reversed (unlikely but handle it)
        if len(text) == 7 and text[:3].isalpha() and text[3:].isdigit():
            potential_correct = text[3:] + text[:3]
            if self.is_valid_spanish_format(potential_correct):
                return potential_correct
        
        return text

    def get_voted_plate(self, car_id: int) -> str:
        """
        Get the plate text using temporal majority voting.
        Returns the most common read across all frames for this car.
        Applies proper Spanish plate formatting.
        """
        if car_id not in self.car_plate_reads:
            return ""
            
        reads = self.car_plate_reads[car_id]['reads']
        if not reads:
            return ""
        
        # Use temporal voting if enabled, otherwise return first read
        if self.temporal_voting_enabled:
            most_common_clean = Counter(reads).most_common(1)[0][0]
        else:
            most_common_clean = reads[0]
        
        # Attempt to correct OCR order errors if enabled
        if self.ocr_correction_enabled:
            corrected_clean = self.correct_ocr_order(most_common_clean)
        else:
            corrected_clean = most_common_clean
        
        # Apply proper Spanish formatting
        return self.format_spanish_plate(corrected_clean)
    
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
    
    def apply_perspective_transform(self, plate_crop: np.ndarray) -> np.ndarray:
        """
        Apply perspective transformation to deskew the license plate.
        Detects contours and flattens the plate into a rectangle.
        """
        if plate_crop is None or plate_crop.size == 0:
            return plate_crop
        
        # Convert to grayscale
        if len(plate_crop.shape) == 3:
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_crop.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return plate_crop
        
        # Find the largest contour with 4 corners (likely the plate border)
        largest_contour = None
        max_area = 0
        epsilon_factor = 0.02
        
        for contour in contours:
            # Approximate contour to polygon
            epsilon = epsilon_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 4:
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
                    largest_contour = approx
        
        if largest_contour is None or len(largest_contour) != 4:
            # Try with larger epsilon if no 4-corner contour found
            for contour in contours:
                epsilon = 0.04 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) >= 4:
                    area = cv2.contourArea(contour)
                    if area > max_area:
                        max_area = area
                        largest_contour = approx
                        break
        
        if largest_contour is None or len(largest_contour) < 4:
            return plate_crop
        
        # Get the corners and take only the first 4
        corners = largest_contour.reshape(-1, 2)[:4]
        
        if len(corners) != 4:
            return plate_crop
        
        # Sort corners by sum (top-left has smallest sum, bottom-right has largest)
        corners = corners[np.argsort(corners.sum(axis=1))]
        top_left, top_right = corners[:2]
        bottom_left, bottom_right = corners[2:]
        
        # Sort by x to get top-left, top-right
        top_left, top_right = sorted([top_left, top_right], key=lambda x: x[0])
        bottom_left, bottom_right = sorted([bottom_left, bottom_right], key=lambda x: x[0])
        
        # Calculate widths
        top_width = np.sqrt((top_right[0] - top_left[0])**2 + (top_right[1] - top_left[1])**2)
        bottom_width = np.sqrt((bottom_right[0] - bottom_left[0])**2 + (bottom_right[1] - bottom_left[1])**2)
        left_height = np.sqrt((top_left[0] - bottom_left[0])**2 + (top_left[1] - bottom_left[1])**2)
        right_height = np.sqrt((top_right[0] - bottom_right[0])**2 + (top_right[1] - bottom_right[1])**2)
        
        # Output rectangle dimensions (use maximum of each pair)
        output_width = int(max(top_width, bottom_width))
        output_height = int(max(left_height, right_height))
        
        # Ensure minimum size
        output_width = max(output_width, 100)
        output_height = max(output_height, 30)
        
        # Source points (detected corners)
        src_pts = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
        
        # Destination points (rectangle)
        dst_pts = np.array([
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1]
        ], dtype=np.float32)
        
        # Get perspective transform matrix
        transform_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # Apply perspective transformation
        try:
            warped = cv2.warpPerspective(plate_crop, transform_matrix, (output_width, output_height))
            return warped
        except:
            return plate_crop
    
    def preprocess_plate_improved(self, plate_crop: np.ndarray) -> list:
        """
        Improved preprocessing for Spanish plates.
        Returns list of (processed_image, method_name)
        For PaddleOCR, we need 3-channel images (RGB/BGR).
        """
        methods = []
        
        # Ensure we have a 3-channel image for PaddleOCR
        if len(plate_crop.shape) == 2:
            # Grayscale -> BGR
            plate_crop = cv2.cvtColor(plate_crop, cv2.COLOR_GRAY2BGR)
        elif plate_crop.shape[2] == 1:
            # Single channel -> BGR
            plate_crop = cv2.cvtColor(plate_crop, cv2.COLOR_GRAY2BGR)
        
        # Method 1: Original color
        methods.append((plate_crop.copy(), 'original_color'))
        
        # Method 2: Convert to grayscale and back to color (simulates some preprocessing)
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        color_from_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        methods.append((color_from_gray, 'gray_to_color'))
        
        # Method 3: Brightness/contrast adjustment
        alpha = 1.5  # Contrast control
        beta = 20    # Brightness control
        adjusted = cv2.convertScaleAbs(plate_crop, alpha=alpha, beta=beta)
        methods.append((adjusted, 'contrast_brightness'))
        
        # Method 4: Sharpen
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(plate_crop, -1, kernel)
        methods.append((sharpened, 'sharpened'))
        
        # Method 5: Histogram equalization (color)
        # Convert to YUV, equalize Y, convert back
        yuv = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        hist_eq = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        methods.append((hist_eq, 'hist_eq_color'))
        
        # Method 6: Denoised
        denoised = cv2.fastNlMeansDenoisingColored(plate_crop, None, 10, 10, 7, 21)
        methods.append((denoised, 'denoised_color'))
        
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
            
            # Check format validity using regex validation
            is_valid_spanish = self.is_valid_spanish_format(text)
            
            if text and score >= best_score:
                best_text = text
                best_score = score
                best_method = method_name
        
        return best_text, best_score, best_method
    
    def is_valid_spanish_format(self, text: str) -> bool:
        """
        Check if text matches Spanish plate format using Regex.
        Strips hyphens and spaces before checking to prevent format rejection.
        """
        if not text:
            return False
            
        # Clean the text to just alphanumeric characters
        clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # Format 1: Modern (e.g., 8314JSP) - 4 numbers, 3 letters
        # Note: Spain excludes vowels in modern plates
        if re.match(r'^\d{4}[B-DF-HJ-NP-TV-Z]{3}$', clean_text):
            return True
            
        # Format 2: Old Provincial (e.g., M1485ZX or AL1234A) 
        # 1-2 letters, 4 numbers, 1-2 letters
        if re.match(r'^[A-Z]{1,2}\d{4}[A-Z]{1,2}$', clean_text):
            return True
            
        # Format 3: Very Old Provincial (e.g., AB1234)
        # 1-2 letters, 4-6 numbers
        if re.match(r'^[A-Z]{1,2}\d{4,6}$', clean_text):
            return True
            
        return False
    
    def _convert_botsort_to_sort_format(self, detections, track_ids):
        """
        Convert BoT-SORT format to SORT format for compatibility with get_car function.
        BoT-SORT returns track_ids separately, SORT returns [x1,y1,x2,y2,score,track_id]
        """
        if not detections:
            return []
        
        sort_format = []
        for i, det in enumerate(detections):
            if i < len(track_ids):
                tid = track_ids[i]
            else:
                tid = -1
            # get_car expects: [x1, y1, x2, y2, track_id] - 5 values
            sort_format.append([det[0], det[1], det[2], det[3], float(tid)])
        
        return np.array(sort_format) if sort_format else np.array([])
    
    def detect_vehicles(self, frame):
        """Detect vehicles in frame using fine-tuned model. Returns detections and track_ids."""
        det_start = time.time()
        
        if self.tracker_type == "botsort":
            # Use YOLO's track method with BoT-SORT
            results = self.model.track(
                frame, 
                persist=True, 
                tracker="botsort.yaml", 
                imgsz=self.yolo_imgsz,
                verbose=False,
                device=self.device
            )[0]
            
            detections = []
            track_ids = []
            
            # Use explicit attributes instead of raw data tensor to avoid unpacking errors
            if results.boxes is not None and len(results.boxes) > 0:
                # Safely extract boxes, confidences, and classes
                boxes = results.boxes.xyxy.cpu().tolist()
                confs = results.boxes.conf.cpu().tolist()
                clsses = results.boxes.cls.cpu().tolist()
                
                # Check if tracking IDs are available
                if results.boxes.id is not None:
                    t_ids = results.boxes.id.cpu().tolist()
                    for box, conf, cls, t_id in zip(boxes, confs, clsses, t_ids):
                        x1, y1, x2, y2 = box
                        if int(cls) in self.vehicle_classes and conf > VEHICLE_CONF_THRESHOLD:
                            detections.append([x1, y1, x2, y2, conf])
                            track_ids.append(int(t_id))
                else:
                    # Fallback if no tracking IDs (first frame or no detections)
                    for box, conf, cls in zip(boxes, confs, clsses):
                        x1, y1, x2, y2 = box
                        if int(cls) in self.vehicle_classes and conf > VEHICLE_CONF_THRESHOLD:
                            detections.append([x1, y1, x2, y2, conf])
                            track_ids.append(-1)  # No track ID available
            
            self.detection_times.append(time.time() - det_start)
            return detections, track_ids
        else:
            # Use traditional SORT tracking
            results = self.model(frame, imgsz=self.yolo_imgsz, verbose=False, device=self.device)[0]
            
            detections = []
            if results.boxes is not None and len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().tolist()
                confs = results.boxes.conf.cpu().tolist()
                clsses = results.boxes.cls.cpu().tolist()
                
                for box, conf, cls in zip(boxes, confs, clsses):
                    x1, y1, x2, y2 = box
                    # Use config threshold
                    if int(cls) in self.vehicle_classes and conf > VEHICLE_CONF_THRESHOLD:
                        detections.append([x1, y1, x2, y2, conf])
            
            self.detection_times.append(time.time() - det_start)
            return detections
    
    def detect_license_plates(self, frame):
        """Detect license plates in frame using fine-tuned model with tracking."""
        plate_start = time.time()
        
        # Use track() to get tracking IDs for plates
        results = self.model.track(
            frame, 
            persist=True, 
            tracker="botsort.yaml",
            imgsz=self.yolo_imgsz, 
            verbose=False, 
            device=self.device
        )[0]
        
        plates = []
        plate_track_ids = []
        
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().tolist()
            confs = results.boxes.conf.cpu().tolist()
            clsses = results.boxes.cls.cpu().tolist()
            
            # Get track IDs if available
            if results.boxes.id is not None:
                track_ids = results.boxes.id.cpu().tolist()
            else:
                track_ids = [-1] * len(boxes)
            
            for box, conf, cls, track_id in zip(boxes, confs, clsses, track_ids):
                x1, y1, x2, y2 = box
                cls_int = int(cls)
                conf_val = float(conf)
                
                # Filter for license plate class (80) and apply confidence threshold
                if cls_int == self.plate_class and conf_val >= self.plate_confidence_threshold:
                    # Check minimum size
                    plate_width = x2 - x1
                    plate_height = y2 - y1
                    
                    # Size filter
                    if plate_width >= self.min_plate_size[0] and plate_height >= self.min_plate_size[1]:
                        plates.append([x1, y1, x2, y2, conf, cls])
                        plate_track_ids.append(int(track_id) if track_id is not None else -1)
        
        self.detection_times.append(time.time() - plate_start)
        return plates, plate_track_ids
    
    def process_frame(self, frame):
        """Process a single frame."""
        frame_start = time.time()
        
        # Detect vehicles (returns detections and track_ids for botsort)
        vehicle_result = self.detect_vehicles(frame)
        
        # Handle both tracker types
        if self.tracker_type == "botsort":
            vehicle_detections, track_ids = vehicle_result
            # Convert to SORT-like format for get_car function
            # track_ids format: [det1_track_id, det2_track_id, ...]
            track_ids = self._convert_botsort_to_sort_format(vehicle_detections, track_ids)
        else:
            vehicle_detections = vehicle_result
            # Track vehicles with SORT
            if vehicle_detections:
                track_ids = self.mot_tracker.update(np.asarray(vehicle_detections))
            else:
                track_ids = []
        
        # Detect license plates (now returns both detections and track IDs)
        plate_detections, plate_track_ids = self.detect_license_plates(frame)
        
        frame_results = {}
        
        # Process each detected plate directly using plate's own track ID for temporal voting
        for i, plate in enumerate(plate_detections):
            x1, y1, x2, y2, score, class_id = plate
            
            # Use the plate's own track ID from YOLO tracking
            # This allows proper temporal voting across frames
            if i < len(plate_track_ids) and plate_track_ids[i] != -1:
                car_id = plate_track_ids[i]
            else:
                # Fallback: use position-based ID if no track ID available
                car_id = int(self.frame_count * 1000 + x1)
            
            # Crop license plate with padding
            pad = 5
            y1_pad = max(0, int(y1) - pad)
            y2_pad = min(frame.shape[0], int(y2) + pad)
            x1_pad = max(0, int(x1) - pad)
            x2_pad = min(frame.shape[1], int(x2) + pad)
            
            plate_crop = frame[y1_pad:y2_pad, x1_pad:x2_pad, :]
            
            if plate_crop.size == 0 or plate_crop.shape[0] < 20 or plate_crop.shape[1] < 60:
                continue
            
            # Apply upscale factor from config if > 1
            if self.ocr_upscale_factor > 1.0:
                plate_crop = cv2.resize(plate_crop, None, 
                                        fx=self.ocr_upscale_factor, 
                                        fy=self.ocr_upscale_factor, 
                                        interpolation=cv2.INTER_CUBIC)
            
            # Display cropped plate if enabled (before OCR)
            if self.enable_display_plate:
                # Create window once
                if not hasattr(self, '_plate_window_created'):
                    cv2.namedWindow('License Plate Crop', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('License Plate Crop', 600, 200)
                    self._plate_window_created = True
                
                # Update window title with frame number
                cv2.setWindowTitle('License Plate Crop', f'License Plate Crop - Frame {self.frame_count}')
                
                cv2.imshow('License Plate Crop', plate_crop)
                cv2.waitKey(1)
            
            # Read Spanish license plate with multiple preprocessing methods (ALWAYS run this)
            plate_text, plate_score, ocr_method = self.read_license_plate_multi(plate_crop)
            
            # Only track if confidence is above threshold
            if plate_text and plate_score >= self.ocr_confidence_threshold:
                is_valid = self.is_valid_spanish_format(plate_text)
                
                # IMPROVED: Check Spanish format before accepting
                if is_valid:
                    self.plates_detected += 1
                    self.valid_spanish_plates += 1
                    
                    # Track plate read for temporal voting
                    self.track_plate_read(car_id, plate_text, plate_score)
                    
                    # Store frame results (use plate bbox as car bbox since no car matching)
                    frame_results[car_id] = {
                        'car': {'bbox': [x1, y1, x2, y2]},
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
        
        # Return all data needed for visualization
        return {
            'frame_results': frame_results,
            'vehicle_detections': vehicle_detections,
            'track_ids': track_ids,
            'plate_detections': plate_detections
        }
    
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
        
        # Create display window if enabled
        if self.enable_display:
            cv2.namedWindow('Spanish LPR - Headless System', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Spanish LPR - Headless System', 1280, 720)
        
        print("\n🎬 Starting headless processing...")
        if self.enable_display:
            print("   Controls: Q=Quit, P=Pause, S=Screenshot")
        print("   " + "-"*50)
        
        paused = False
        frame = None  # Initialize for screenshot functionality
        
        while True:
            if not paused:
                # Read frame
                ret, frame = cap.read()
                if not ret or (max_frames and self.frame_count >= max_frames):
                    break
                
                # Process frame and get detection data
                frame_data = self.process_frame(frame)
                
                # Increment frame counter
                self.frame_count += 1
                
                # Display frame if enabled
                if self.enable_display and frame_data:
                    # Use the create_visualization method
                    viz_frame = self.create_visualization(
                        frame=frame,
                        vehicle_detections=frame_data['vehicle_detections'],
                        plate_detections=frame_data['plate_detections'],
                        tracked_cars=frame_data['track_ids'],
                        frame_results=frame_data['frame_results']
                    )
                    
                    # Calculate current FPS
                    elapsed = time.time() - start_time
                    self.current_fps = self.frame_count / elapsed if elapsed > 0 else 0
                    
                    # Calculate current FPS
                    elapsed = time.time() - start_time
                    self.current_fps = self.frame_count / elapsed if elapsed > 0 else 0
                    
                    cv2.imshow('Spanish LPR - Headless System', viz_frame)
                
                # Print progress every 50 frames
                if self.frame_count % 50 == 0:
                    elapsed = time.time() - start_time
                    current_fps = self.frame_count / elapsed if elapsed > 0 else 0
                    progress = self.frame_count / total_frames * 100
                    
                    unique_cars = len(self.car_plate_reads)
                    
                    print(f"   📈 Frame {self.frame_count}/{total_frames} ({progress:.1f}%) | "
                          f"FPS: {current_fps:.1f} | "
                          f"Plates: {self.plates_detected} | "
                          f"Unique Cars: {unique_cars} | "
                          f"Valid: {self.valid_spanish_plates}")
            
            # Handle keyboard input if display enabled
            if self.enable_display:
                key = cv2.waitKey(1 if not paused else 0) & 0xFF
                
                if key == ord('q'):  # Quit
                    print("\n🛑 Quitting...")
                    break
                elif key == ord('p'):  # Pause/Resume
                    paused = not paused
                    print(f"   {'⏸️ Paused' if paused else '▶️ Resumed'}")
                elif key == ord('s'):  # Screenshot
                    screenshot_path = f"screenshot_{self.frame_count}.jpg"
                    cv2.imwrite(screenshot_path, frame)
                    print(f"   📸 Screenshot saved: {screenshot_path}")
        
        # Clean up
        cap.release()
        if self.enable_display:
            cv2.destroyAllWindows()
        
        # Calculate final statistics
        elapsed_time = time.time() - start_time
        avg_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        return avg_fps
    
    def compare_plates(self) -> dict:
        """
        Compare unique OCR results with ground truth plates using temporal voting.
        """
        if not self.ground_truth_plates:
            print("⚠️  No ground truth provided, skipping accuracy comparison")
            return None
        
        # Get voted plate for each car
        ocr_plates_with_votes = {}
        for car_id in self.car_plate_reads:
            voted_plate = self.get_voted_plate(car_id)
            if voted_plate:
                ocr_plates_with_votes[car_id] = voted_plate
        
        ocr_plates = list(ocr_plates_with_votes.values())
        
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
        for car_id, ocr_plate in ocr_plates_with_votes.items():
            matched = False
            match_type = None
            char_accuracy = 0.0
            max_confidence = self.car_plate_reads[car_id]['max_confidence']
            
            # Try exact match first
            for gt_plate in self.ground_truth_plates:
                if ocr_plate == gt_plate:
                    comparison['exact_matches'].append({
                        'ocr_plate': ocr_plate,
                        'ground_truth': gt_plate,
                        'confidence': max_confidence
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
                            'confidence': max_confidence,
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
                    'confidence': max_confidence
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
        print(f"   Unique cars detected: {len(self.car_plate_reads)}")
        print(f"   Total processing time: {time.strftime('%H:%M:%S', time.gmtime(self.frame_count/avg_fps)) if avg_fps > 0 else 'N/A'}")
        print(f"   Average FPS: {avg_fps:.2f}")
        
        if self.plates_detected > 0:
            print(f"   Plates per frame: {self.plates_detected/self.frame_count:.3f}")
            print(f"   Spanish format validation rate: {self.valid_spanish_plates/self.plates_detected*100:.1f}%")
        
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
            # Show sample results with temporal voting
            print(f"\n📋 Sample results (first 5 unique plates per car with temporal voting):")
            count = 0
            for car_id, reads_data in sorted(self.car_plate_reads.items(), 
                                              key=lambda x: x[1]['max_confidence'], reverse=True)[:5]:
                voted_plate = self.get_voted_plate(car_id)
                if voted_plate:
                    print(f"   Car {car_id}: {voted_plate} (reads: {len(reads_data['reads'])}, max_confidence: {reads_data['max_confidence']:.3f})")
                    count += 1
        
        # Generate JSON report
        report = {
            'timestamp': datetime.now().isoformat(),
            'video_file': os.path.basename(self.video_path),
            'frames_processed': self.frame_count,
            'plates_detected': self.plates_detected,
            'valid_spanish_plates': self.valid_spanish_plates,
            'unique_cars_detected': len(self.car_plate_reads),
            'average_fps': avg_fps,
            'gpu_acceleration': self.use_gpu,
            'gpu_device': torch.cuda.get_device_name(0) if self.use_gpu else None,
            'output_file': self.output_csv,
            'processing_time_seconds': self.frame_count/avg_fps if avg_fps > 0 else 0,
            'plates_per_frame': self.plates_detected/self.frame_count if self.frame_count > 0 else 0,
            'spanish_format_validation_rate': self.valid_spanish_plates/self.plates_detected if self.plates_detected > 0 else 0,
            'version': self.version
        }
        
        report_file = f'spanish_lpr_headless_v{self.version.replace(".", "")}_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Full report saved to: {report_file}")
        
        return report
    
    def write_unique_plates_file(self):
        """
        Write unique plates to text file with ground truth comparison.
        Format: Ground truth | car_id | license_nmb | license_nmb_score | Accuracy
        FIXED: Now outputs exactly one entry per ground truth plate (max 30 entries)
        IMPROVED: Uses Hungarian algorithm style matching to prevent duplicate assignments
        """
        print(f"\n💾 Writing unique plates to: {self.output_text_file}")
        
        # Get all voted plates with their car_id and confidence
        voted_plates_data = []  # List of (car_id, voted_plate, confidence)
        for car_id in self.car_plate_reads:
            voted_plate = self.get_voted_plate(car_id)
            if voted_plate:
                confidence = self.car_plate_reads[car_id]['max_confidence']
                voted_plates_data.append((car_id, voted_plate, confidence))
        
        # Normalize ground truth for comparison (remove spaces/hyphens)
        gt_normalized = {}
        for gt in self.ground_truth_plates:
            gt_clean = re.sub(r'[^A-Z0-9]', '', gt.upper())
            gt_normalized[gt_clean] = gt
        
        # Create a list of all possible matches with their scores
        match_candidates = []  # (gt_clean, car_id, voted_plate, confidence, accuracy)
        
        for gt_clean, gt_original in gt_normalized.items():
            for car_id, voted_plate, confidence in voted_plates_data:
                ocr_clean = re.sub(r'[^A-Z0-9]', '', voted_plate.upper())
                
                # Calculate character-level accuracy
                matches = 0
                min_len = min(len(ocr_clean), len(gt_clean))
                for i in range(min_len):
                    if ocr_clean[i] == gt_clean[i]:
                        matches += 1
                accuracy = matches / max(len(ocr_clean), len(gt_clean)) if max(len(ocr_clean), len(gt_clean)) > 0 else 0
                
                # Only consider matches with reasonable accuracy (> 0.5 or exact match for short plates)
                if accuracy > 0.5 or (len(gt_clean) <= 4 and accuracy > 0.25):
                    match_candidates.append((gt_clean, car_id, voted_plate, confidence, accuracy, gt_original))
        
        # Sort matches by accuracy (descending), then confidence (descending)
        match_candidates.sort(key=lambda x: (x[4], x[3]), reverse=True)
        
        # Greedy assignment: assign each ground truth to the best available OCR result
        matched_results = {}  # gt_clean -> (car_id, voted_plate, confidence, accuracy)
        used_car_ids = set()
        used_ocr_texts = set()
        
        for gt_clean, car_id, voted_plate, confidence, accuracy, gt_original in match_candidates:
            # Skip if this ground truth already matched
            if gt_clean in matched_results:
                continue
            
            # Skip if this OCR result already used (prevent duplicates)
            ocr_clean = re.sub(r'[^A-Z0-9]', '', voted_plate.upper())
            if ocr_clean in used_ocr_texts:
                continue
            
            # Assign this match
            matched_results[gt_clean] = (car_id, voted_plate, confidence, accuracy)
            used_car_ids.add(car_id)
            used_ocr_texts.add(ocr_clean)
        
        with open(self.output_text_file, 'w') as f:
            f.write("Ground truth\tcar_id\tlicense_nmb\tlicense_nmb_score\tAccuracy\n")
            
            total_correct_digits = 0
            total_digits_compared = 0
            exact_matches = 0
            detected_count = 0
            
            # Write results sorted by ground truth plate
            for gt_clean, gt_original in sorted(gt_normalized.items()):
                if gt_clean in matched_results:
                    car_id, voted_plate, confidence, accuracy = matched_results[gt_clean]
                    detected_count += 1
                    
                    # Calculate digit accuracy (7 characters for Spanish plates)
                    correct_digits = int(accuracy * 7)
                    total_correct_digits += correct_digits
                    total_digits_compared += 7
                    
                    if accuracy == 1.0:
                        exact_matches += 1
                        accuracy_str = "100%"
                    else:
                        accuracy_str = f"{int(accuracy*100)}% ({correct_digits}/7)"
                    
                    f.write(f"{gt_original}\t{car_id:.0f}\t{voted_plate}\t{confidence:.5f}\t\t{accuracy_str}\n")
                else:
                    # No match found for this ground truth plate
                    f.write(f"{gt_original}\t\tNA\t\t\t\t0%\n")
            
            # Write accuracy summary
            f.write(f"\nAccuracy Summary:\n")
            f.write(f"Total detected: {detected_count} out of {len(gt_normalized)}\n")
            f.write(f"Exact matches: {exact_matches}\n")
            if len(gt_normalized) > 0:
                detection_rate = detected_count / len(gt_normalized) * 100
                f.write(f"Detection rate: {detected_count}/{len(gt_normalized)} = {detection_rate:.1f}%\n")
            if total_digits_compared > 0:
                digit_accuracy = total_correct_digits / total_digits_compared * 100
                f.write(f"Digit accuracy: {total_correct_digits}/{total_digits_compared} = {digit_accuracy:.1f}%\n")
        
        print(f"   Wrote {len(matched_results)} unique plates (one per ground truth plate)")


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Headless Spanish License Plate Recognition System v1.1 - FIXED DUPLICATE REPORTING'
    )
    parser.add_argument('--video', '-v',
                        type=str,
                        required=True,
                        help='Path to input video file')
    parser.add_argument('--output', '-o',
                        type=str,
                        default='results_spanish_headless.csv',
                        help='Output CSV file path')
    parser.add_argument('--plates-file', '-pf',
                        type=str,
                        default='unique_plates.txt',
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
    parser.add_argument('--display', '-d',
                        action='store_true',
                        help='Enable display window with cv2.imshow')
    parser.add_argument('--display-plate', '-dp',
                        action='store_true',
                        help='Display cropped license plate before OCR')
    return parser.parse_args()


def main():
    """
    Main function for headless Spanish LPR system v1.1.
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
    enable_display = args.display
    enable_display_plate = args.display_plate
    
    print("\n" + "=" * 70)
    print(f"🚀 HEADLESS SPANISH LPR - v1.1 (FIXED DUPLICATE REPORTING)")
    print("=" * 70)
    print(f"Video: {video_path}")
    if ground_truth_file:
        print(f"Ground Truth: {ground_truth_file}")
    print(f"Min Confidence: {min_confidence}")
    print(f"Display: {'Enabled' if enable_display else 'Disabled'}")
    print(f"Display Plate: {'Enabled' if enable_display_plate else 'Disabled'}")
    print(f"Plates File: {output_text_file}")
    print("=" * 70)
    
    # Create and run system
    lpr_system = HeadlessSpanishLPR_v1_1(
        video_path=video_path,
        output_csv=output_csv,
        output_text_file=output_text_file,
        ground_truth_file=ground_truth_file,
        min_confidence=min_confidence,
        enable_display=enable_display,
        enable_display_plate=enable_display_plate
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
    print(f"   • Version: v{report.get('version', '1.1')}")
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
