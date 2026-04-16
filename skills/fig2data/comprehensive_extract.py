"""
Comprehensive data extraction by combining OCR text with image analysis.
Uses color detection and pattern matching to extract numerical data from charts.
"""
import os
import cv2
import numpy as np
from PIL import Image
import json
import re

def get_color_at_position(img, x, y, radius=5):
    """Get the average color around a position."""
    h, w = img.shape[:2]
    y = min(max(y, radius), h - radius)
    x = min(max(x, radius), w - radius)
    roi = img[y-radius:y+radius, x-radius:x+radius]
    return roi.mean(axis=(0)).tolist() if roi.size > 0 else [0, 0, 0]

def analyze_bar_chart_by_color(img_path, num_bars=5):
    """Analyze bar chart by detecting colored regions and their heights."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find bars by detecting vertical regions of dark pixels
    # Focus on the middle portion of the image where bars typically are
    left_margin = int(width * 0.15)
    right_margin = int(width * 0.9)
    top_margin = int(height * 0.15)
    bottom_margin = int(height * 0.9)
    
    # Get the chart area
    chart_area = gray[top_margin:bottom_margin, left_margin:right_margin]
    
    # Find horizontal slices to estimate number of bars
    vertical_profile = chart_area.mean(axis=1)
    
    # Detect bar regions by looking for dark vertical bands
    _, binary = cv2.threshold(chart_area, 180, 255, cv2.THRESH_BINARY_INV)
    
    # Project horizontally to find bar positions
    horizontal_projection = binary.sum(axis=0)
    
    # Find peaks in the projection (these are the bars)
    bars = []
    in_bar = False
    bar_start = 0
    
    for i, val in enumerate(horizontal_projection):
        if val > 500 and not in_bar:
            in_bar = True
            bar_start = i
        elif val < 100 and in_bar:
            in_bar = False
            bar_center = (bar_start + i) // 2
            bar_width = i - bar_start
            
            # Get bar height by looking at vertical projection
            bar_region = binary[:, bar_start:i]
            if bar_region.size > 0:
                bar_height = bar_region.sum(axis=1).max()
                normalized_height = bar_height / (height * 255) if height > 0 else 0
                
                bars.append({
                    'position': bar_center + left_margin,
                    'width': bar_width,
                    'height_pixels': int(bar_height),
                    'normalized_height': round(normalized_height, 3)
                })
    
    return {
        'chart_area': f'{left_margin},{top_margin} to {right_margin},{bottom_margin}',
        'detected_bars': len(bars),
        'bars': bars
    }

def detect_scatter_points(img_path):
    """Detect scatter plot points by color clustering."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    height, width = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for common plot colors
    colors = {
        'blue': ([100, 80, 80], [140, 255, 255]),
        'red': ([0, 80, 80], [10, 255, 255]),
        'green': ([40, 80, 80], [80, 255, 255]),
        'orange': ([10, 100, 100], [25, 255, 255]),
        'purple': ([120, 50, 50], [150, 255, 255])
    }
    
    # Chart area (approximate - exclude axes)
    chart_left = int(width * 0.12)
    chart_right = int(width * 0.95)
    chart_top = int(height * 0.1)
    chart_bottom = int(height * 0.9)
    
    all_points = []
    
    for color_name, (lower, upper) in colors.items():
        lower_np = np.array(lower)
        upper_np = np.array(upper)
        mask = cv2.inRange(hsv, lower_np, upper_np)
        
        # Also check second red range
        if color_name == 'red':
            mask2 = cv2.inRange(hsv, np.array([170, 80, 80]), np.array([180, 255, 255]))
            mask = cv2.bitwise_or(mask, mask2)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 30 < area < 300:  # Filter too small/large
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Only include points in chart area
                    if chart_left < cx < chart_right and chart_top < cy < chart_bottom:
                        all_points.append({
                            'x': cx,
                            'y': cy,
                            'x_pct': round((cx - chart_left) / (chart_right - chart_left), 3),
                            'y_pct': round((chart_bottom - cy) / (chart_bottom - chart_top), 3),
                            'color': color_name,
                            'area': int(area)
                        })
    
    return {
        'chart_bounds': {'left': chart_left, 'right': chart_right, 'top': chart_top, 'bottom': chart_bottom},
        'total_points': len(all_points),
        'points': all_points[:50]
    }

def detect_heatmap_grid(img_path):
    """Detect heatmap by finding grid cells and their colors."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find grid cells by detecting non-text regions
    # First, find the grid area
    chart_left = int(width * 0.25)
    chart_top = int(height * 0.15)
    chart_right = int(width * 0.95)
    chart_bottom = int(height * 0.9)
    
    # Get the grid area
    grid_area = img[chart_top:chart_bottom, chart_left:chart_right]
    grid_gray = gray[chart_top:chart_bottom, chart_left:chart_right]
    
    # Threshold to find colored cells
    _, binary = cv2.threshold(grid_gray, 230, 255, cv2.THRESH_BINARY)
    
    # Invert to find colored regions
    colored = cv2.bitwise_not(binary)
    
    # Find contours
    contours, _ = cv2.findContours(colored, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cells = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 30 < w < 150 and 20 < h < 100:  # Reasonable cell size
            # Get color
            roi = grid_area[y:y+h, x:x+w]
            b, g, r = roi.mean(axis=(0, 1))
            
            # Determine correlation value from color
            # Blue (negative) to white (zero) to red (positive)
            if r > b:
                value = (r - b) / 255
            else:
                value = -(b - r) / 255
            
            cells.append({
                'col': x,
                'row': y,
                'width': w,
                'height': h,
                'rgb': {'r': round(r, 1), 'g': round(g, 1), 'b': round(b, 1)},
                'correlation': round(value, 2)
            })
    
    # Sort cells by position
    cells = sorted(cells, key=lambda c: (c['row'], c['col']))
    
    return {
        'grid_bounds': {'left': chart_left, 'top': chart_top, 'right': chart_right, 'bottom': chart_bottom},
        'cell_count': len(cells),
        'cells': cells
    }

# Process all images
image_dir = 'final_report_images'
results = {}

# Image descriptions based on OCR
image_descriptions = {
    'image1.png': 'Bar chart - Accuracy per skill with different k values (K=0, K=[0,0,0,1], etc.)',
    'image2.png': 'Logo - Academic College of AFEKA',
    'image3.png': 'Bar chart - Mean prediction score by model (Grade 4)',
    'image4.png': 'Scatter plot - Expected vs actual retained accuracy (Grade 4)',
    'image5.png': 'Scatter plot - Expected vs actual retained accuracy (Grade 5)',
    'image6.png': 'Architecture diagram - Controllable Generative Student',
    'image7.png': 'Bar chart - Retained vs Forgotten comparisons (grades 4 & 5)',
    'image8.png': 'Bar chart - Imperfect Student accuracy on forgotten skills (Grade 5)',
    'image9.png': 'Bar chart - Mean prediction score by model (Grade 5)',
    'image10.png': 'Bar chart - RMSE by prompt strategy (Grade 4)',
    'image12.png': 'Empty/placeholder',
    'image13.png': 'Bar chart - Imperfect Student accuracy on forgotten skills (Grade 4)',
    'image14.png': 'Heatmap - Skill correlation matrix (Grade 4)',
    'image15.png': 'Heatmap - Skill correlation matrix (Grade 5)',
    'image16.png': 'Bar chart - RMSE by prompt strategy (Grade 5)'
}

for filename in sorted(os.listdir(image_dir)):
    if not filename.endswith('.png'):
        continue
    
    filepath = os.path.join(image_dir, filename)
    print(f"\nProcessing {filename}...")
    print(f"  Description: {image_descriptions.get(filename, 'Unknown')}")
    
    result = {
        'filename': filename,
        'description': image_descriptions.get(filename, 'Unknown')
    }
    
    # Try different extraction methods based on image type
    if 'bar chart' in image_descriptions.get(filename, '').lower():
        bar_data = analyze_bar_chart_by_color(filepath)
        if bar_data:
            result['bar_analysis'] = bar_data
    
    if 'scatter' in image_descriptions.get(filename, '').lower():
        scatter_data = detect_scatter_points(filepath)
        if scatter_data:
            result['scatter_analysis'] = scatter_data
    
    if 'heatmap' in image_descriptions.get(filename, '').lower():
        heatmap_data = detect_heatmap_grid(filepath)
        if heatmap_data:
            result['heatmap_analysis'] = heatmap_data
    
    results[filename] = result

# Save results
output_path = 'comprehensive_chart_data.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n\nResults saved to {output_path}")
