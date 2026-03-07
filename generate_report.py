#!/usr/bin/env python3
"""
Spanish LPR System v1.1 - Final Testing Report Generator
Generates a PDF report with all parameter testing results
"""

from fpdf import FPDF
import os

class LPRReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Spanish License Plate Recognition System v1.1', 0, 1, 'C')
        self.ln(5)
    
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(4)
    
    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, body)
        self.ln()

# Create PDF
pdf = LPRReport()
pdf.add_page()

# Title
pdf.set_font('Arial', 'B', 16)
pdf.cell(0, 10, 'Parameter Testing Report', 0, 1, 'C')
pdf.ln(5)

pdf.set_font('Arial', 'I', 10)
pdf.cell(0, 8, 'Version: v1.1', 0, 1, 'C')
pdf.cell(0, 8, 'Date: March 2026', 0, 1, 'C')
pdf.ln(10)

# Executive Summary
pdf.chapter_title('1. Executive Summary')
summary = """This report presents the results of comprehensive parameter testing for the Spanish License Plate Recognition (LPR) System v1.1. A total of 55 tests were conducted across 5 key parameters to identify optimal configuration settings.

Test Environment:
- Video: out.mp4 (2337 frames, 1280x720 @ 30 fps)
- Ground Truth: 30 Spanish license plates
- GPU: NVIDIA GeForce RTX 3080 Ti
- Tracker: BoSort
- OCR Correction: Enabled

Key Findings:
- YOLO_IMGSZ = 640 provides optimal balance (70% detection, 87.1% digit accuracy)
- PLATE_CONF_THRESHOLD has the most dramatic impact on detection rate
- VEHICLE_CONF_THRESHOLD and OCR_MIN_CONFIDENCE have minimal impact
- OCR_UPSCALE_FACTOR = 2.0 is optimal (default)
"""
pdf.chapter_body(summary)

# Test Results Summary Table
pdf.chapter_title('2. Test Results Summary')

# YOLO_IMGSZ Results
pdf.set_font('Arial', 'B', 11)
pdf.cell(0, 8, '2.1 YOLO_IMGSZ Test Results', 0, 1)
pdf.ln(2)

# Table header
pdf.set_font('Arial', 'B', 8)
col_width = 25
headers = ['Value', 'Detection', 'Digit Acc', 'Cars', 'FPS']
for h in headers:
    pdf.cell(col_width, 7, h, 1, 0, 'C')
pdf.ln()

# YOLO_IMGSZ data
pdf.set_font('Arial', '', 8)
yolo_data = [
    (320, '40.0%', '82.1%', 12, 15.2),
    (384, '43.3%', '87.9%', 13, 13.8),
    (448, '46.7%', '82.7%', 14, 14.3),
    (512, '53.3%', '80.4%', 16, 12.9),
    (576, '60.0%', '88.9%', 18, 10.6),
    (640, '70.0%', '87.1%', 21, 10.1),
    (704, '53.3%', '90.2%', 16, 9.2),
    (768, '73.3%', '81.2%', 22, 8.5),
    (832, '70.0%', '82.3%', 21, 8.0),
    (960, '70.0%', '78.9%', 21, 8.0),
    (1280, '73.3%', '72.7%', 22, 7.8),
]
for row in yolo_data:
    for val in row:
        pdf.cell(col_width, 7, str(val), 1, 0, 'C')
    pdf.ln()

pdf.ln(5)
pdf.set_font('Arial', 'B', 11)
pdf.cell(0, 8, 'Key Finding: YOLO_IMGSZ = 640 provides optimal balance with 70% detection rate and 87.1% digit accuracy.', 0, 1)
pdf.ln(10)

# VEHICLE_CONF_THRESHOLD Results
pdf.chapter_title('2.2 VEHICLE_CONF_THRESHOLD Test Results')
pdf.set_font('Arial', '', 10)
pdf.cell(0, 7, 'Values tested: 0.10, 0.18, 0.26, 0.34, 0.42, 0.50, 0.58, 0.66, 0.74, 0.82, 0.90', 0, 1)
pdf.ln(3)
pdf.cell(0, 7, 'Result: Minimal impact on detection. Detection rate remained stable at 70% for values 0.10-0.74.', 0, 1)
pdf.cell(0, 7, 'Recommended: 0.50 (default)', 0, 1)
pdf.ln(10)

# PLATE_CONF_THRESHOLD Results
pdf.chapter_title('2.3 PLATE_CONF_THRESHOLD Test Results')
pdf.set_font('Arial', '', 10)
pdf.cell(0, 7, 'Values tested: 0.10, 0.18, 0.26, 0.34, 0.42, 0.50, 0.58, 0.66, 0.74, 0.82, 0.90', 0, 1)
pdf.ln(3)
pdf.cell(0, 7, 'Result: DRAMATIC impact on detection rate!', 0, 1)
pdf.ln(3)
pdf.cell(0, 7, '- Values 0.10-0.42: Stable 70% detection rate', 0, 1)
pdf.cell(0, 7, '- Values >0.50: Drastic reduction in detection', 0, 1)
pdf.cell(0, 7, '- Values 0.82-0.90: Near-zero detection (only 1 plate)', 0, 1)
pdf.ln(3)
pdf.cell(0, 7, 'Recommended: 0.40 (default) - Critical parameter!', 0, 1)
pdf.ln(10)

# OCR_UPSCALE_FACTOR Results
pdf.chapter_title('2.4 OCR_UPSCALE_FACTOR Test Results')
pdf.set_font('Arial', '', 10)
pdf.cell(0, 7, 'Values tested: 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0', 0, 1)
pdf.ln(3)
pdf.cell(0, 7, 'Result: Moderate impact on accuracy and detection', 0, 1)
pdf.ln(3)
pdf.cell(0, 7, '- Lower values (1.0-1.2): Higher detection but lower digit accuracy', 0, 1)
pdf.cell(0, 7, '- Higher values (2.4-3.0): Lower detection but higher digit accuracy', 0, 1)
pdf.ln(3)
pdf.cell(0, 7, 'Best overall: 2.0 (default) with 70% detection and 87.1% digit accuracy', 0, 1)
pdf.ln(10)

# OCR_MIN_CONFIDENCE Results
pdf.chapter_title('2.5 OCR_MIN_CONFIDENCE Test Results')
pdf.set_font('Arial', '', 10)
pdf.cell(0, 7, 'Values tested: 0.10, 0.18, 0.26, 0.34, 0.42, 0.50, 0.58, 0.66, 0.74, 0.82, 0.90', 0, 1)
pdf.ln(3)
pdf.cell(0, 7, 'Result: NO significant impact on results. All values produced identical results.', 0, 1)
pdf.cell(0, 7, 'Recommended: 0.40 (default)', 0, 1)
pdf.ln(10)

# Parameter Impact Analysis
pdf.chapter_title('3. Parameter Impact Analysis')

impact = """Parameter Impact Ranking (Most to Least Impactful):

1. PLATE_CONF_THRESHOLD (HIGHEST IMPACT)
   - Range tested: 0.10 - 0.90
   - Impact: Dramatic reduction in detection above 0.50
   - Recommendation: Keep at 0.40 (default)

2. YOLO_IMGSZ (HIGH IMPACT)
   - Range tested: 320 - 1280
   - Impact: Significant trade-off between detection and accuracy
   - Recommendation: 640 (optimal balance)

3. OCR_UPSCALE_FACTOR (MODERATE IMPACT)
   - Range tested: 1.0 - 3.0
   - Impact: Moderate effect on detection vs accuracy
   - Recommendation: 2.0 (default)

4. VEHICLE_CONF_THRESHOLD (LOW IMPACT)
   - Range tested: 0.10 - 0.90
   - Impact: Minimal across wide range
   - Recommendation: 0.50 (default)

5. OCR_MIN_CONFIDENCE (NO IMPACT)
   - Range tested: 0.10 - 0.90
   - Impact: No measurable effect
   - Recommendation: 0.40 (default)
"""
pdf.chapter_body(impact)

# Recommended Configuration
pdf.chapter_title('4. Recommended Optimal Configuration')

config = """Based on comprehensive testing, the following configuration is recommended:

  Parameter                  Value       Notes
  -------------------------  ----------  ----------------------------------------
  YOLO_IMGSZ                640         Optimal balance of detection & accuracy
  VEHICLE_CONF_THRESHOLD    0.50        Default - works well across all tests
  PLATE_CONF_THRESHOLD      0.40        CRITICAL - keep at default
  OCR_UPSCALE_FACTOR        2.0         Default - optimal OCR performance
  OCR_MIN_CONFIDENCE        0.40         Default - no impact on results
  TRACKER_TYPE              botsort     Fixed throughout testing
  OCR_CORRECTION_ENABLED    True        Fixed throughout testing

Expected Performance with Optimal Settings:
  - Detection Rate: 70% (21/30 plates)
  - Digit Accuracy: 87.1%
  - Processing Speed: ~10 FPS
  - Exact Matches: 12/21 detected plates
"""
pdf.chapter_body(config)

# Conclusions
pdf.chapter_title('5. Conclusions')

conclusions = """1. YOLO_IMGSZ = 640 provides the best balance between detection rate (70%) 
   and digit accuracy (87.1%) at approximately 10 FPS.

2. PLATE_CONF_THRESHOLD is the most critical parameter - values above 0.50 
   drastically reduce detection capability.

3. VEHICLE_CONF_THRESHOLD has minimal impact on results across a wide range 
   (0.10-0.74).

4. OCR_UPSCALE_FACTOR = 2.0 (default) provides optimal performance.

5. OCR_MIN_CONFIDENCE has no measurable impact on results - the temporal 
   voting system appears to handle confidence filtering internally.

6. The system achieves 70% detection rate and 87.1% digit accuracy with the 
   default configuration, which represents a good balance for Spanish license 
   plate recognition.
"""
pdf.chapter_body(conclusions)

# Appendix
pdf.chapter_title('6. Appendix: Complete Test Data')

appendix = """Total Tests: 55
Test Duration: Approximately 3-4 minutes per test
Total Testing Time: ~3 hours

All test results have been saved to: metrics_testing.csv

The CSV file contains the following columns:
- test_number: Sequential test identifier
- parameter_tested: Name of the parameter being tested
- parameter_value: Value used for the parameter
- yolo_imgsz: YOLO inference resolution
- vehicle_conf_threshold: Vehicle detection confidence threshold
- plate_conf_threshold: Plate detection confidence threshold  
- ocr_upscale_factor: OCR image upscale factor
- ocr_min_confidence: OCR minimum confidence threshold
- tracker_type: Object tracker used (botsort)
- ocr_correction_enabled: OCR correction feature (True)
- avg_confidence: Average confidence score
- detection_rate: Percentage of ground truth plates detected
- digit_accuracy: Percentage of correctly recognized digits
- num_cars_detected: Number of unique vehicles detected
- num_plates_detected: Total number of plate detections across all frames
"""
pdf.chapter_body(appendix)

# Footer
pdf.ln(20)
pdf.set_font('Arial', 'I', 8)
pdf.cell(0, 5, 'Report generated for Spanish LPR System v1.1', 0, 1, 'C')

# Save PDF
output_path = '/home/alex/apli/numberplate/numberplate_yolo8/LPR_Testing_Report_v1.1.pdf'
pdf.output(output_path)
print(f"PDF report saved to: {output_path}")
