"""
Final comprehensive data extraction combining OCR text with image analysis.
This parses the OCR results and extracts tabular data from charts.
"""
import os
import json
import re
import cv2
import numpy as np
from collections import defaultdict

# Load the OCR data we already have
with open('extracted_data.json', 'r', encoding='utf-8') as f:
    ocr_data = json.load(f)

def extract_numbers_from_text(text):
    """Extract all numbers (floats and integers) from text."""
    pattern = r'-?\d+\.?\d*'
    return [float(n) for n in re.findall(pattern, text)]

def parse_bar_chart_ocr(text, image_name):
    """Parse bar chart data from OCR text."""
    data = {
        'type': 'bar_chart',
        'image': image_name,
        'data_points': []
    }
    
    # Extract numbers
    numbers = extract_numbers_from_text(text)
    
    # Common chart labels to identify data categories
    skill_labels = [
        'Measurement & Data', 'Number & Operation', 'Operations & Algebraic',
        'Chance', 'Number', 'Fraction', 'Algebraic', 'Operations',
        'Think:', 'Base.', 'Data'
    ]
    
    strategy_labels = ['Combined', 'Few-shot', 'Rule-based']
    model_labels = ['claude', 'deepseek', 'gpt-4o']
    
    # Determine chart type based on content
    if any(s in text.lower() for s in ['rmse', 'strategy']):
        data['chart_type'] = 'RMSE by Strategy'
        # For RMSE charts, typically 3 bars per group
        if len(numbers) >= 3:
            data['data_points'] = [{'strategy': s, 'value': v} 
                                   for s, v in zip(strategy_labels[:len(numbers)//3+1], numbers[:3])]
    
    elif any(s in text.lower() for s in ['mean prediction', 'prediction score']):
        data['chart_type'] = 'Prediction Score by Model'
        if numbers:
            data['data_points'] = [{'model': m, 'value': v} 
                                   for m, v in zip(model_labels, numbers[:3])]
    
    elif any(s in text.lower() for s in ['accuracy', 'skill', 'grade']):
        data['chart_type'] = 'Accuracy per Skill'
        # This is more complex - extract key values
        data['all_values'] = numbers
    
    return data

def parse_scatter_data(text, image_name):
    """Parse scatter plot data."""
    numbers = extract_numbers_from_text(text)
    
    data = {
        'type': 'scatter_plot',
        'image': image_name,
        'description': '',
        'values': numbers[:20]  # First 20 numbers likely represent data
    }
    
    # Determine what's being plotted
    if 'expected' in text.lower() and 'actual' in text.lower():
        data['description'] = 'Expected vs Actual Retained Accuracy'
        # Values typically pairs: (expected, actual) or single points
        data['data_points'] = numbers
    
    return data

def parse_heatmap_ocr(text, image_name):
    """Parse heatmap/correlation matrix from OCR text."""
    data = {
        'type': 'heatmap',
        'image': image_name,
        'description': 'Skill Correlation Matrix'
    }
    
    # Extract correlation values (typically between -1 and 1)
    numbers = extract_numbers_from_text(text)
    correlations = [n for n in numbers if -1.1 <= n <= 1.1]
    
    # Extract skill labels
    skills = []
    for line in text.split('\n'):
        for skill in ['Measurement & Data', 'Number & Operations', 'Operations & Algebraic', 
                      'Number & Operation', 'Operations & Algeb']:
            if skill.lower() in line.lower():
                skills.append(skill)
    
    data['skills'] = list(set(skills))[:4]  # Unique skills
    data['correlation_values'] = correlations[:16]  # Up to 4x4 matrix
    
    return data

def process_all_images():
    """Process all images and extract data."""
    results = {}
    
    for image_name, ocr_result in ocr_data.items():
        text = ocr_result.get('text', '')
        
        result = {
            'filename': image_name,
            'size': ocr_result.get('size', ''),
            'text_length': len(text),
            'preview': text[:200] + '...' if len(text) > 200 else text
        }
        
        # Determine chart type and parse
        text_lower = text.lower()
        
        if 'correlation' in text_lower:
            result.update(parse_heatmap_ocr(text, image_name))
        elif 'scatter' in text_lower or ('expected' in text_lower and 'actual' in text_lower):
            result.update(parse_scatter_data(text, image_name))
        elif any(k in text_lower for k in ['rmse', 'accuracy', 'score', 'strategy']):
            result.update(parse_bar_chart_ocr(text, image_name))
        elif image_name == 'image2.png':
            result['type'] = 'logo'
            result['description'] = 'Academic College of AFEKA logo'
        elif image_name == 'image6.png':
            result['type'] = 'diagram'
            result['description'] = 'Architecture diagram'
        else:
            result['type'] = 'unknown'
            result['extracted_numbers'] = extract_numbers_from_text(text)[:20]
        
        results[image_name] = result
    
    return results

# Process and save
results = process_all_images()

output_file = 'extracted_experimental_data.md'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("# Experimental Data Extracted from Final Report Images\n\n")
    f.write("This document contains experimental data extracted from figures in the Final Report.\n\n")
    f.write("---\n\n")
    
    for img_name, data in results.items():
        f.write(f"## {img_name}\n\n")
        f.write(f"**Type:** {data.get('type', 'unknown').upper()}\n")
        f.write(f"**Size:** {data.get('size', 'N/A')}\n\n")
        
        if 'description' in data:
            f.write(f"**Description:** {data['description']}\n\n")
        
        if 'chart_type' in data:
            f.write(f"**Chart Type:** {data['chart_type']}\n\n")
        
        if 'data_points' in data:
            f.write("### Data Points\n\n")
            f.write("| Item | Value |\n")
            f.write("|------|-------|\n")
            for point in data['data_points']:
                if isinstance(point, dict):
                    for k, v in point.items():
                        f.write(f"| {k} | {v} |\n")
                else:
                    f.write(f"| {point} | |\n")
            f.write("\n")
        
        if 'correlation_values' in data:
            f.write("### Correlation Values\n\n")
            f.write(f"Values: {data['correlation_values']}\n\n")
            if 'skills' in data:
                f.write(f"Skills: {', '.join(data['skills'])}\n\n")
        
        if 'values' in data:
            f.write("### Extracted Values\n\n")
            f.write(f"```\n{data['values']}\n```\n\n")
        
        if 'all_values' in data:
            f.write("### All Numerical Values\n\n")
            f.write(f"```\n{data['all_values']}\n```\n\n")
        
        if 'extracted_numbers' in data:
            f.write("### Extracted Numbers\n\n")
            f.write(f"```\n{data['extracted_numbers']}\n```\n\n")
        
        f.write("---\n\n")

print(f"Data extraction complete. Results saved to {output_file}")
