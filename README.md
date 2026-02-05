# Spanish License Plate Recognition with YOLOv8 & GPU Acceleration

**Production-ready Spanish License Plate Recognition system** with real-time visualization, GPU acceleration, and specialized Spanish plate validation.

## ✨ Features

- **Spanish Plate Validation**: Specialized processing for Spanish plate formats (####-LLL, LL-####)
- **GPU Acceleration**: CUDA-enabled processing for real-time performance
- **Real-time Visualization**: Professional overlay with bounding boxes, track IDs, and performance metrics
- **Dual Operation Modes**: 
  - Production system with visualization (5.85 FPS)
  - Headless mode for maximum performance (6.2 FPS)
- **SORT Tracking**: Multi-object tracking for vehicle continuity

## 📊 Performance Metrics (NVIDIA GeForce 940M, 2GB)

| Metric | Production (Visualization) | Headless (Max Performance) |
|--------|----------------------------|----------------------------|
| **Frames Processed** | 300 | 300 |
| **Average FPS** | 5.85 | 6.15 |
| **Plates Detected** | 37 | 37 |
| **Valid Spanish Plates** | 21 | 21 |
| **Spanish Plate Accuracy** | 56.8% | 56.8% |
| **Processing Time** | 51.26 seconds | 48.75 seconds |

## 🚀 Quick Start

### 1. Installation
```bash
# Clone repository
git clone https://github.com/alexdominguez09/numberplate_yolo8.git
cd numberplate_yolo8

# Install dependencies
pip install -r requirements-dev.txt

# Download required models
./download_models.sh  # See Model Download section
```

### 2. Model Download
Create `download_models.sh`:
```bash
#!/bin/bash
# Download YOLOv8n model
wget https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8n.pt

# Create models directory
mkdir -p models

# Download license plate detector (original from Video-ANPR)
# Note: You need to download this from the original repository
echo "Please download license_plate_detector.pt from:"
echo "https://github.com/sveyek/Video-ANPR/tree/main/models"
echo "and place it in the models/ directory"
```

### 3. Usage Examples
```python
# Production system with visualization
python main_spanish_production.py

# Headless mode (maximum performance)
python main_spanish_headless.py

# Test with Spanish plate images
python test_spanish_images.py
```

## 🛠️ System Architecture

1. **Vehicle Detection**: YOLOv8n (COCO-trained) for vehicle detection
2. **License Plate Detection**: Fine-tuned YOLO model for Spanish plates
3. **Spanish Plate Validation**: Custom preprocessing and character correction
4. **OCR Processing**: EasyOCR with Spanish-specific optimizations
5. **Tracking**: SORT algorithm for vehicle continuity
6. **Visualization**: OpenCV-based real-time overlay

## 📁 File Structure
```
numberplate_yolo8/
├── main_spanish_production.py    # Production system with visualization
├── main_spanish_headless.py      # Headless version (max performance)
├── utils_spanish_fixed.py        # Spanish plate validation & preprocessing
├── main_spanish_realtime.py      # Alternative realtime implementation
├── test_spanish_images.py        # Testing utility
├── requirements-dev.txt          # Dependencies
├── LICENSE                       # MIT License
└── README.md                     # This file
```

## 🔧 Configuration
Key configuration parameters in `main_spanish_production.py`:
- `plate_confidence_threshold = 0.3`
- `ocr_confidence_threshold = 0.15`
- `min_plate_size = (20, 20)`
- `max_frames = 300` (set to `None` for full video)

## 📝 Spanish Plate Formats Supported
- **Current (2000+)**: `####-LLL` (e.g., `1234-ABC`)
- **Old (pre-2000)**: `LL-####` (e.g., `AB-1234`)
- **Character Restrictions**: No vowels (AEIOU), no Q

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments
- Original Video-ANPR project: [sveyek/Video-ANPR](https://github.com/sveyek/Video-ANPR)
- YOLOv8 by Ultralytics
- SORT tracking algorithm
- EasyOCR for text recognition