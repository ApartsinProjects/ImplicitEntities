#!/usr/bin/env python3
"""
Fig2Data - Extract Experimental Data from Chart Images
======================================================
A comprehensive tool for extracting tabular data from images of charts, plots, and figures.

Usage:
    python main.py <docx_file> [--output OUTPUT_DIR] [--format markdown|json]

Example:
    python main.py "Final Report.docx" --output extracted_data --format markdown
"""

import os
import sys
import json
import zipfile
import argparse
import cv2
import numpy as np
from PIL import Image
from collections import defaultdict

# Try to import easyocr (optional)
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("Warning: easyocr not installed. Run: pip install easyocr")


class Fig2Data:
    """Main class for extracting experimental data from chart images."""
    
    def __init__(self, output_dir="extracted_data"):
        self.output_dir = output_dir
        self.image_dir = os.path.join(output_dir, "images")
        self.results = {}
        os.makedirs(self.image_dir, exist_ok=True)
    
    def extract_images_from_docx(self, docx_path):
        """Extract embedded images from a DOCX file."""
        print(f"Extracting images from {docx_path}...")
        
        with zipfile.ZipFile(docx_path, 'r') as z:
            media_files = [f for f in z.namelist() if f.startswith('word/media/')]
            
            for f in media_files:
                filename = os.path.basename(f)
                filepath = os.path.join(self.image_dir, filename)
                with open(filepath, 'wb') as out:
                    out.write(z.read(f))
        
        print(f"Extracted {len(media_files)} images to {self.image_dir}")
        return sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])
    
    def extract_text_ocr(self, image_path):
        """Extract text from image using EasyOCR."""
        if not EASYOCR_AVAILABLE:
            return ""
        
        reader = easyocr.Reader(['en'], gpu=False)
        results = reader.readtext(image_path, detail=0)
        return '\n'.join(results)
    
    def analyze_bar_chart(self, image_path):
        """Analyze bar chart and extract data."""
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        horizontal_projection = binary.sum(axis=0)
        
        bars = []
        in_bar = False
        bar_start = 0
        
        for i, val in enumerate(horizontal_projection):
            if val > 500 and not in_bar:
                in_bar = True
                bar_start = i
            elif val < 100 and in_bar:
                in_bar = False
                bar_width = i - bar_start
                if bar_width > 5:
                    bars.append({
                        'position': bar_start + bar_width // 2,
                        'width': bar_width,
                        'normalized_position': (bar_start + bar_width // 2) / width
                    })
        
        return {'type': 'bar_chart', 'bars': bars[:20]}
    
    def detect_scatter_points(self, image_path):
        """Detect scatter plot points."""
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        height, width = img.shape[:2]
        
        colors = {
            'blue': ([100, 80, 80], [140, 255, 255]),
            'red': ([0, 80, 80], [10, 255, 255]),
            'green': ([40, 80, 80], [80, 255, 255])
        }
        
        chart_left = int(width * 0.12)
        chart_right = int(width * 0.95)
        chart_top = int(height * 0.1)
        chart_bottom = int(height * 0.9)
        
        points = []
        for color_name, (lower, upper) in colors.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            if color_name == 'red':
                mask2 = cv2.inRange(hsv, np.array([170, 80, 80]), np.array([180, 255, 255]))
                mask = cv2.bitwise_or(mask, mask2)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 30 < area < 300:
                    M = cv2.moments(cnt)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        if chart_left < cx < chart_right and chart_top < cy < chart_bottom:
                            points.append({
                                'x': cx, 'y': cy,
                                'x_pct': round(cx / width, 3),
                                'y_pct': round(1 - cy / height, 3),
                                'color': color_name
                            })
        
        return {'type': 'scatter_plot', 'points': points[:50]}
    
    def extract_heatmap_grid(self, image_path):
        """Extract correlation/heatmap values."""
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        height, width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        chart_left = int(width * 0.25)
        chart_top = int(height * 0.15)
        chart_right = int(width * 0.95)
        chart_bottom = int(height * 0.9)
        
        grid_area = img[chart_top:chart_bottom, chart_left:chart_right]
        grid_gray = gray[chart_top:chart_bottom, chart_left:chart_right]
        
        _, binary = cv2.threshold(grid_gray, 230, 255, cv2.THRESH_BINARY)
        colored = cv2.bitwise_not(binary)
        
        contours, _ = cv2.findContours(colored, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cells = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if 30 < w < 150 and 20 < h < 100:
                b, g, r = grid_area[y:y+h, x:x+w].mean(axis=(0, 1))
                value = (r - b) / 255 if r > b else -(b - r) / 255
                cells.append({
                    'col': x, 'row': y,
                    'rgb': {'r': round(r, 1), 'g': round(g, 1), 'b': round(b, 1)},
                    'value': round(value, 2)
                })
        
        return {'type': 'heatmap', 'cells': sorted(cells, key=lambda c: (c['row'], c['col']))}
    
    def classify_image(self, image_path):
        """Classify image type."""
        img = cv2.imread(image_path)
        if img is None:
            return 'unknown'
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        non_white = np.sum(gray < 250) / (height * width)
        
        if non_white < 0.05:
            return 'empty'
        
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rectangular = sum(1 for c in contours if cv2.contourArea(c) > 1000 and 
                         0.5 < cv2.boundingRect(c)[2]/max(cv2.boundingRect(c)[3], 1) < 10)
        
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 10, 
                                   param1=50, param2=10, minRadius=3, maxRadius=15)
        
        if circles is not None and len(circles[0]) > 10:
            return 'scatter'
        elif rectangular > 5:
            return 'heatmap'
        elif rectangular > 2:
            return 'bar_chart'
        
        return 'other'
    
    def extract_numbers_from_text(self, text):
        """Extract all numbers from OCR text."""
        import re
        pattern = r'-?\d+\.?\d*'
        return [float(n) for n in re.findall(pattern, text)]
    
    def process_all_images(self):
        """Process all images in the output directory."""
        image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])
        
        for filename in image_files:
            filepath = os.path.join(self.image_dir, filename)
            print(f"Processing {filename}...")
            
            img_type = self.classify_image(filepath)
            
            result = {'filename': filename, 'type': img_type}
            
            # Extract text
            if EASYOCR_AVAILABLE:
                text = self.extract_text_ocr(filepath)
                result['text'] = text
                result['numbers'] = self.extract_numbers_from_text(text)
            
            # Extract chart data
            if img_type == 'bar_chart':
                result['chart_data'] = self.analyze_bar_chart(filepath)
            elif img_type == 'scatter':
                result['chart_data'] = self.detect_scatter_points(filepath)
            elif img_type == 'heatmap':
                result['chart_data'] = self.extract_heatmap_grid(filepath)
            
            self.results[filename] = result
        
        return self.results
    
    def save_json(self, output_file="extracted_data.json"):
        """Save results as JSON."""
        output_path = os.path.join(self.output_dir, output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"Saved JSON to {output_path}")
    
    def save_markdown(self, output_file="extracted_data.md"):
        """Save results as Markdown."""
        output_path = os.path.join(self.output_dir, output_file)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Experimental Data Extracted from Figures\n\n")
            f.write("This document contains experimental data extracted from figure images.\n\n")
            f.write("---\n\n")
            
            for filename, data in self.results.items():
                f.write(f"## {filename}\n\n")
                f.write(f"**Type:** {data.get('type', 'unknown').upper()}\n\n")
                
                if 'text' in data and data['text']:
                    f.write("### Extracted Text\n\n")
                    f.write(f"```\n{data['text'][:500]}...\n```\n\n")
                
                if 'numbers' in data and data['numbers']:
                    f.write("### Extracted Numbers\n\n")
                    f.write(f"```\n{data['numbers'][:20]}\n```\n\n")
                
                if 'chart_data' in data:
                    f.write("### Chart Data\n\n")
                    f.write(f"```json\n{json.dumps(data['chart_data'], indent=2)}\n```\n\n")
                
                f.write("---\n\n")
        
        print(f"Saved Markdown to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract experimental data from chart images")
    parser.add_argument("docx_file", help="Path to DOCX file with embedded images")
    parser.add_argument("--output", default="extracted_data", help="Output directory")
    parser.add_argument("--format", default="markdown", choices=["markdown", "json"], help="Output format")
    
    args = parser.parse_args()
    
    extractor = Fig2Data(args.output)
    extractor.extract_images_from_docx(args.docx_file)
    extractor.process_all_images()
    
    if args.format == "json":
        extractor.save_json()
    else:
        extractor.save_markdown()
    
    print("Done!")


if __name__ == "__main__":
    main()
