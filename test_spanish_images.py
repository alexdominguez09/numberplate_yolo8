"""
Test Spanish license plate recognition on sample images.
"""
import cv2
import os
import numpy as np
from utils_spanish import read_spanish_license_plate, clean_spanish_plate_text, validate_spanish_plate

def test_spanish_images():
    """Test Spanish plate recognition on actual image files."""
    test_dir = "test_spanish_plates"
    
    if not os.path.exists(test_dir):
        print(f"Test directory not found: {test_dir}")
        return
    
    print(f"Testing Spanish plate recognition in: {test_dir}")
    print("="*60)
    
    # Get all image files
    image_files = []
    for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
        image_files.extend([f for f in os.listdir(test_dir) if f.lower().endswith(ext)])
    
    if not image_files:
        print("No test images found")
        return
    
    print(f"Found {len(image_files)} test images")
    
    results = []
    
    for img_file in sorted(image_files):
        img_path = os.path.join(test_dir, img_file)
        
        print(f"\n{'='*40}")
        print(f"Image: {img_file}")
        print('='*40)
        
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            print(f"  Could not read image")
            continue
        
        # Display image info
        height, width = img.shape[:2]
        print(f"  Size: {width}x{height}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Test with preprocessing
        plate_text, plate_score = read_spanish_license_plate(gray, use_preprocessing=True)
        
        if plate_text:
            print(f"  Detected: {plate_text}")
            print(f"  Confidence: {plate_score:.3f}")
            
            # Validate format
            is_valid, formatted, format_type = validate_spanish_plate(plate_text)
            if is_valid:
                print(f"  Format: {format_type}")
                print(f"  Formatted: {formatted}")
            else:
                print(f"  WARNING: Not a valid Spanish format")
            
            results.append({
                'image': img_file,
                'detected': plate_text,
                'confidence': plate_score,
                'valid': is_valid,
                'formatted': formatted,
                'format': format_type
            })
            
            # Display image with result
            display = cv2.resize(img, (min(600, width), min(200, height)))
            
            # Add text overlay
            text_y = 30
            texts = [
                f"Plate: {plate_text}",
                f"Confidence: {plate_score:.3f}",
                f"Format: {format_type if is_valid else 'Invalid'}"
            ]
            
            for text in texts:
                cv2.putText(display, text, (10, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                text_y += 30
            
            # Show image
            cv2.imshow(f"Spanish Plate: {img_file}", display)
            cv2.waitKey(1000)  # Show for 1 second
            cv2.destroyAllWindows()
            
        else:
            print(f"  No plate detected")
            results.append({
                'image': img_file,
                'detected': None,
                'confidence': 0.0,
                'valid': False,
                'formatted': None,
                'format': None
            })
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    
    detected = [r for r in results if r['detected']]
    valid = [r for r in results if r['valid']]
    
    print(f"Total images: {len(results)}")
    print(f"Plates detected: {len(detected)} ({len(detected)/len(results)*100:.1f}%)")
    print(f"Valid Spanish plates: {len(valid)} ({len(valid)/len(results)*100:.1f}%)")
    
    if detected:
        print(f"\nDetected plates:")
        for result in detected:
            status = "✓" if result['valid'] else "✗"
            print(f"  {status} {result['image']}: {result['detected']} "
                  f"(conf: {result['confidence']:.3f}, format: {result['format']})")
    
    # Save results
    import json
    with open('spanish_plate_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: spanish_plate_test_results.json")
    
    return results

def analyze_ocr_issues():
    """Analyze why Spanish plates might not be detected correctly."""
    print(f"\n{'='*60}")
    print("OCR ISSUE ANALYSIS")
    print('='*60)
    
    # Common Spanish plate OCR issues
    print("Common Spanish plate OCR challenges:")
    print("1. Blue EU strip on left side can confuse OCR")
    print("2. Hyphen (-) between numbers and letters")
    print("3. Specific character set (no vowels AEIOU)")
    print("4. Font variations in Spanish plates")
    print("5. Reflection and lighting issues")
    
    print("\nSuggested improvements:")
    print("1. Preprocess to remove blue EU strip")
    print("2. Train OCR specifically on Spanish plates")
    print("3. Adjust character recognition for Spanish font")
    print("4. Use multiple OCR engines and combine results")
    print("5. Implement post-processing with Spanish rules")

if __name__ == "__main__":
    print("SPANISH LICENSE PLATE TESTING")
    print("="*60)
    
    results = test_spanish_images()
    analyze_ocr_issues()