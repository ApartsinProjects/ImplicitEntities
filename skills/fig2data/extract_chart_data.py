"""
Comprehensive chart data extraction from images.
Extracts data from bar charts, scatter plots, and heatmaps using OpenCV and image processing.
"""
import os
import cv2
import numpy as np
from PIL import Image
import json
import re

def extract_bar_chart_data(image_path, title=""):
    """Extract data from bar charts by detecting bar heights."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold to find bars (dark bars on light background)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bars = []
    height, width = gray.shape
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter out small contours (noise)
        if w > 10 and h > 10:
            bars.append({
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h),
                'bottom_y': height - y,
                'normalized_height': h / height
            })
    
    # Sort by x position
    bars = sorted(bars, key=lambda b: b['x'])
    
    return {
        'type': 'bar_chart',
        'bar_count': len(bars),
        'bars': bars[:20]  # Limit to 20 bars
    }

def extract_scatter_plot_data(image_path):
    """Extract data points from scatter plots by detecting colored dots."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    height, width = img.shape[:2]
    
    # Convert to different color spaces for detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Detect colored points (excluding axis labels)
    # Common colors in scatter plots: blue, red, green
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    
    # Create masks for each color
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    # Find contours for each color
    def get_points(mask, color_name):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        points = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 20 < area < 500:  # Filter noise
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    # Skip points near edges (likely axis labels)
                    if 50 < cx < width - 50 and 50 < cy < height - 50:
                        points.append({
                            'x': cx,
                            'y': cy,
                            'x_normalized': cx / width,
                            'y_normalized': cy / height,
                            'color': color_name
                        })
        return points
    
    all_points = []
    all_points.extend(get_points(mask_blue, 'blue'))
    all_points.extend(get_points(mask_red, 'red'))
    all_points.extend(get_points(mask_green, 'green'))
    
    return {
        'type': 'scatter_plot',
        'total_points': len(all_points),
        'points': all_points[:100]
    }

def extract_heatmap_data(image_path):
    """Extract correlation values from heatmap images."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect colored cells in the heatmap
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Blue to red gradient detection
    # Lower values (negative) = blue, Higher values = red
    h, s, v = cv2.split(hsv)
    
    # Create a mask for colored cells (excluding white/black text)
    _, text_mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    _, dark_mask = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    
    # Invert to find colored regions
    colored_regions = cv2.bitwise_and(s, v)
    
    # Find contours of colored regions
    contours, _ = cv2.findContours(colored_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cells = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 20 and h > 20:  # Valid cell size
            # Get average color to determine value
            roi = img[y:y+h, x:x+w]
            avg_color = roi.mean(axis=(0,1))
            
            # Calculate normalized value based on color
            # Blue (negative) to Red (positive)
            b, g, r = avg_color
            if r > b:
                value = (r - b) / 255 * 1.0  # Positive correlation
            else:
                value = -(b - r) / 255 * 1.0  # Negative correlation
            
            cells.append({
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h),
                'avg_color': {'r': float(r), 'g': float(g), 'b': float(b)},
                'estimated_value': round(value, 2)
            })
    
    # Sort by position (top-left to bottom-right)
    cells = sorted(cells, key=lambda c: (c['y'], c['x']))
    
    return {
        'type': 'heatmap',
        'cell_count': len(cells),
        'cells': cells[:50]
    }

def classify_image(image_path):
    """Classify the image type based on visual features."""
    img = cv2.imread(image_path)
    if img is None:
        return 'unknown'
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    # Check if image is mostly empty
    non_white = np.sum(gray < 250) / (height * width)
    
    if non_white < 0.05:
        return 'empty'
    
    # Check for rectangular regions (bar chart or heatmap)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rectangular = sum(1 for c in contours if cv2.contourArea(c) > 1000 and 
                      0.5 < cv2.boundingRect(c)[2]/max(cv2.boundingRect(c)[3], 1) < 10)
    
    # Check for small circular points (scatter plot)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 10, param1=50, param2=10, minRadius=3, maxRadius=15)
    
    if circles is not None and len(circles[0]) > 10:
        return 'scatter'
    elif rectangular > 5:
        return 'heatmap'
    elif rectangular > 2:
        return 'bar_chart'
    
    return 'other'

# Process all images
image_dir = 'final_report_images'
results = {}

for filename in sorted(os.listdir(image_dir)):
    if not filename.endswith('.png'):
        continue
    
    filepath = os.path.join(image_dir, filename)
    print(f"\nProcessing {filename}...")
    
    # First classify the image
    img_type = classify_image(filepath)
    print(f"  Detected type: {img_type}")
    
    # Extract appropriate data
    if img_type == 'bar_chart':
        data = extract_bar_chart_data(filepath)
    elif img_type == 'scatter':
        data = extract_scatter_plot_data(filepath)
    elif img_type == 'heatmap':
        data = extract_heatmap_data(filepath)
    else:
        data = {'type': img_type}
    
    if data:
        results[filename] = data

# Save results
output_path = 'chart_data_extraction.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n\nResults saved to {output_path}")
