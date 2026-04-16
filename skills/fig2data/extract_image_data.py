"""
Extract experimental data from images in the Final Report using EasyOCR.
"""
import os
import easyocr
from PIL import Image
import json

# Initialize EasyOCR reader (English)
print("Initializing EasyOCR reader...")
reader = easyocr.Reader(['en'], gpu=False)

image_dir = 'final_report_images'
output_file = 'extracted_data.json'

results = {}

# Process each image
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    print(f"\nProcessing {image_file}...")
    
    # Get image dimensions
    with Image.open(image_path) as img:
        width, height = img.size
        print(f"  Image size: {width}x{height}")
    
    # Run OCR
    try:
        ocr_results = reader.readtext(image_path, detail=0)
        text = '\n'.join(ocr_results)
        results[image_file] = {
            'text': text,
            'lines': len(ocr_results),
            'size': f'{width}x{height}'
        }
        print(f"  Extracted {len(ocr_results)} text elements")
        
        # Print first part of text for verification
        if text:
            print(f"  Preview: {text[:500]}...")
    except Exception as e:
        results[image_file] = {'error': str(e)}
        print(f"  Error: {e}")

# Save results
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n\nResults saved to {output_file}")
