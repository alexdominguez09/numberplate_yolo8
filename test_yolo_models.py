"""
Quick YOLO Model Test Script
Run individual YOLO models on out.mp4 to test detection performance.
"""
import cv2
import numpy as np
import argparse
from ultralytics import YOLO
import time
from pathlib import Path

# COCO class names
COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
    39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
    44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
    49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
    54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
    59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
    64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
    69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
    74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
    79: 'toothbrush'
}

# Vehicle classes we care about
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck


def run_detection(model_path: str, video_path: str, conf_threshold: float = 0.25,
                  display: bool = True, max_frames: int = None, filter_classes: list = None):
    """
    Run YOLO detection on video.
    
    Args:
        model_path: Path to YOLO model
        video_path: Path to input video
        conf_threshold: Confidence threshold
        display: Show cv2.imshow window
        max_frames: Max frames to process
        filter_classes: List of class IDs to filter (None = all classes)
    """
    print(f"\n{'='*60}")
    print(f"🔍 YOLO Model Test - Fine-tuned Model")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Video: {video_path}")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"Filter classes: {filter_classes if filter_classes else 'All classes'}")
    print(f"{'='*60}\n")
    
    # Load model
    model = YOLO(model_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file: {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📹 Video: {width}x{height} @ {fps} fps, {total_frames} frames")
    
    if max_frames:
        total_frames = min(total_frames, max_frames)
        print(f"📦 Processing first {max_frames} frames\n")
    
    # Always use fine-tuned model now
    is_fine_tuned = True
    
    # Create window if display enabled
    if display:
        cv2.namedWindow('YOLO Detection Test', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('YOLO Detection Test', 1280, 720)
    
    # Processing
    frame_count = 0
    total_detections = 0
    start_time = time.time()
    
    print("🎬 Starting detection...")
    print("   Controls: Q=Quit, P=Pause, S=Screenshot\n")
    
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret or (max_frames and frame_count >= max_frames):
                break
            
            # Run detection
            results = model(frame, conf=conf_threshold, verbose=False)[0]
            
            # Get detections
            if results.boxes is not None and len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy()
                
                # Filter by selected classes if specified
                if filter_classes is not None:
                    filtered_boxes = []
                    filtered_confs = []
                    filtered_classes = []
                    for box, conf, cls in zip(boxes, confs, classes):
                        if int(cls) in filter_classes:
                            filtered_boxes.append(box)
                            filtered_confs.append(conf)
                            filtered_classes.append(int(cls))
                    boxes = filtered_boxes
                    confs = filtered_confs
                    classes = filtered_classes
                
                # Draw boxes
                for box, conf, cls in zip(boxes, confs, classes):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Get class name from COCO classes (fine-tuned model has COCO classes)
                    class_name = COCO_CLASSES.get(int(cls), f"class_{int(cls)}")
                    
                    # Fine-tuned model - blue for plates, green for others
                    if int(cls) == 80 or 'plate' in class_name.lower():
                        # License plate - blue box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        color = (255, 0, 0)
                    else:
                        # Other classes - green box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        color = (0, 255, 0)
                    
                    label = f"{class_name} [{int(cls)}]: {conf:.2f}"
                    
                    # Label background
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                frame_detections = len(boxes)
            else:
                frame_detections = 0
            
            total_detections += frame_detections
            frame_count += 1
            
            # Info overlay
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            
            info_text = f"Frame: {frame_count}/{total_frames} | FPS: {current_fps:.1f} | Detections: {frame_detections}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show model type and filter info
            if filter_classes:
                model_text = f"Fine-tuned | Classes: {filter_classes}"
            else:
                model_text = "Fine-tuned | All classes"
            cv2.putText(frame, model_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            if display:
                cv2.imshow('YOLO Detection Test', frame)
            
            # Print progress every 30 frames
            if frame_count % 30 == 0:
                print(f"   📈 Frame {frame_count}/{total_frames} | FPS: {current_fps:.1f} | Detections: {frame_detections}")
        
        # Handle keys
        if display:
            key = cv2.waitKey(1 if not paused else 0) & 0xFF
            
            if key == ord('q'):
                print("\n🛑 Quitting...")
                break
            elif key == ord('p'):
                paused = not paused
                print(f"   {'⏸️ Paused' if paused else '▶️ Resumed'}")
            elif key == ord('s'):
                cv2.imwrite(f"screenshot_{frame_count}.jpg", frame)
                print(f"   📸 Screenshot saved")
    
    # Cleanup
    cap.release()
    if display:
        cv2.destroyAllWindows()
    
    # Summary
    elapsed = time.time() - start_time
    avg_fps = frame_count / elapsed if elapsed > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"📊 DETECTION RESULTS")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Frames processed: {frame_count}")
    print(f"Total detections: {total_detections}")
    print(f"Avg detections/frame: {total_detections/frame_count:.2f}" if frame_count > 0 else "N/A")
    print(f"Processing time: {elapsed:.1f}s")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Quick YOLO Fine-tuned Model Test')
    parser.add_argument('--video', '-v', type=str, default='out.mp4',
                        help='Input video path (default: out.mp4)')
    parser.add_argument('--conf', '-c', type=float, default=0.25,
                        help='Confidence threshold (default: 0.25)')
    parser.add_argument('--classes', '-cls', type=str, default=None,
                        help='Comma-separated class IDs to detect (e.g., 2,3,80). If not specified, all 81 classes are detected')
    parser.add_argument('--display', '-d', action='store_true',
                        help='Enable display window')
    parser.add_argument('--max-frames', '-f', type=int, default=None,
                        help='Max frames to process')
    
    args = parser.parse_args()
    
    # Parse class filter
    filter_classes = None
    if args.classes:
        try:
            filter_classes = [int(c.strip()) for c in args.classes.split(',')]
            print(f"🎯 Filtering for classes: {filter_classes}")
        except ValueError:
            print(f"❌ Error: Invalid class IDs. Use format: 2,3,80")
            return
    
    # Check video exists
    if not Path(args.video).exists():
        print(f"❌ Video file not found: {args.video}")
        return
    
    # Run detection with fine-tuned model
    run_detection(
        model_path='./models/yolov8n_license_plate.pt',
        video_path=args.video,
        conf_threshold=args.conf,
        display=args.display,
        max_frames=args.max_frames,
        filter_classes=filter_classes
    )


if __name__ == '__main__':
    main()
