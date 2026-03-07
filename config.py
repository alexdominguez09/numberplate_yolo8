"""
Spanish LPR System v1.1 - Configuration File
============================================
This file contains all adjustable settings for the license plate recognition system.
Copy this file to config.py or modify this directly to change settings.

Version: 1.1
"""

# =============================================================================
# YOLO DETECTION SETTINGS
# =============================================================================

# YOLO Inference Resolution
# -------------------------
# Higher values (1280, 1920) can detect smaller plates better but are slower.
# Recommended: 640 (default), 1280, 1920
YOLO_IMGSZ = 640

# Vehicle Detection Confidence Threshold
# --------------------------------------
# Lower values detect more vehicles but may include false positives.
# Range: 0.0 - 1.0
# Recommended: 0.25 for fine-tuned models that detect both vehicles and plates
VEHICLE_CONF_THRESHOLD = 0.25

# License Plate Detection Confidence Threshold
# --------------------------------------------
# Lower values detect more plates but may include false positives.
# Range: 0.0 - 1.0
# Recommended: 0.4 (default)
PLATE_CONF_THRESHOLD = 0.4

# =============================================================================
# OCR SETTINGS
# =============================================================================

# Plate Image Upscale Factor
# ---------------------------
# Upscale plate crops before OCR to improve accuracy.
# 1.0 = no upscale (default)
# 2.0 = 2x upscale
# 3.0 = 3x upscale (can improve OCR but slower)
# Recommended: 1.0 (default), 2.0, or 3.0
OCR_UPSCALE_FACTOR = 1.5

# OCR Confidence Threshold
# ------------------------
# Minimum confidence required for OCR result to be accepted.
# Range: 0.0 - 1.0
# Recommended: 0.40 (default)
OCR_MIN_CONFIDENCE = 0.42

# Minimum Plate Size (width, height) in pixels
# --------------------------------------------
# Filters out very small detections that are likely false positives
# Recommended: (30, 15) - width, height
MIN_PLATE_SIZE = (30, 15)

# =============================================================================
# TRACKING SETTINGS
# =============================================================================

# Tracker Type
# ------------
# Options: "sort" (default), "botsort"
# BoT-SORT provides better tracking for occluded vehicles but requires
# additional configuration files.
TRACKER_TYPE = "botsort"

# =============================================================================
# PLATE VALIDATION SETTINGS
# =============================================================================

# Enable Temporal Voting
# ---------------------
# When True, stores all OCR reads per car and uses majority voting.
# When False, uses highest confidence read only.
# Recommended: True (default) - improves accuracy
TEMPORAL_VOTING_ENABLED = True

# Enable OCR Order Correction
# ---------------------------
# Attempts to fix common OCR misreads (e.g., 1485MZX -> M1485ZX).
# Only applies correction if original read is not valid Spanish format.
# Recommended: True (default)
OCR_CORRECTION_ENABLED = True
