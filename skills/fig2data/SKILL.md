# Fig2Data Skill - Extract Experimental Data from Chart Images

## Purpose
Extract tabular experimental data from images of charts, plots, and figures in research documents (DOCX, PDF, etc.).

## When to Use
- User asks to "extract data from images", "digitize charts", "extract figures to tables"
- Converting visual data (bar charts, heatmaps, scatter plots) to tabular format
- Working with figures in reports/papers that contain experimental results
- "trace", "debug with tracing", "run pytrace autopsy"

---

## Tools Used

### 1. EasyOCR
- **Purpose**: Optical Character Recognition to extract text from images
- **Install**: `pip install easyocr`
- **Dependencies**: PyTorch (torch, torchvision)
- **Usage**: 
  ```python
  import easyocr
  reader = easyocr.Reader(['en'], gpu=False)
  results = reader.readtext(image_path, detail=0)
  ```
- **Pros**: Pure Python, no external dependencies, good text extraction
- **Cons**: Slow on CPU, doesn't extract numerical data from bars/heatmaps directly

### 2. OpenCV (cv2)
- **Purpose**: Image processing for detecting bars, points, colors
- **Install**: `pip install opencv-python` or `pip install opencv-python-headless`
- **Usage**:
  ```python
  import cv2
  # Detect colored regions
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  # Find contours
  contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  ```
- **Pros**: Powerful for shape detection, color analysis
- **Cons**: Requires manual tuning for each chart type

### 3. PlotDigitizer
- **Purpose**: Python library for digitizing plots
- **Install**: `pip install plotdigitizer`
- **Functions**: `process_image()`, `find_trajectory()`, `axis_transformation()`
- **Cons**: Requires interactive point selection (not fully automated)

### 4. Pillow (PIL)
- **Purpose**: Basic image loading and manipulation
- **Install**: `pip install pillow`
- **Usage**:
  ```python
  from PIL import Image
  img = Image.open(image_path)
  width, height = img.size
  ```

### 5. Scikit-image
- **Purpose**: Advanced image processing
- **Install**: `pip install scikit-image`
- **Usage**: For edge detection, morphology, etc.

### 6. Matplotlib
- **Purpose**: Visualization and plotting
- **Install**: `pip install matplotlib`
- **Usage**: For debugging, visualizing detection results

### 7. Seaborn
- **Purpose**: Statistical data visualization
- **Install**: `pip install seaborn`
- **Usage**: Creating publication-quality charts from extracted data

### 8. Plotly
- **Purpose**: Interactive visualization
- **Install**: `pip install plotly`
- **Usage**: For creating interactive charts from extracted data

### 8. ImageIO
- **Purpose**: Image I/O operations
- **Install**: `pip install imageio`
- **Usage**: Reading/writing various image formats

### 9. pdf2image
- **Purpose**: Convert PDF pages to images
- **Install**: `pip install pdf2image`
- **Usage**: For extracting figures from PDF documents

### 10. PyTorch (torch, torchvision)
- **Purpose**: Deep learning backend for EasyOCR
- **Install**: `pip install torch torchvision`
- **Note**: Automatically installed as EasyOCR dependency

---

## Workflow

### Step 1: Extract Images from Document
```python
import zipfile
import os

def extract_images_from_docx(docx_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with zipfile.ZipFile(docx_path, 'r') as z:
        for f in z.namelist():
            if f.startswith('word/media/'):
                filename = os.path.basename(f)
                filepath = os.path.join(output_dir, filename)
                with open(filepath, 'wb') as out:
                    out.write(z.read(f))
```

### Step 2: OCR Text Extraction
```python
import easyocr
import json

def extract_text_from_images(image_dir, output_file):
    reader = easyocr.Reader(['en'], gpu=False)
    results = {}
    
    for filename in sorted(os.listdir(image_dir)):
        if filename.endswith('.png'):
            filepath = os.path.join(image_dir, filename)
            text = reader.readtext(filepath, detail=0)
            results[filename] = {'text': '\n'.join(text), 'lines': len(text)}
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
```

### Step 3: Chart-Specific Extraction

#### Bar Chart Analysis
```python
import cv2
import numpy as np

def analyze_bar_chart(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    # Threshold to find dark bars
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Project horizontally to find bar positions
    horizontal_projection = binary.sum(axis=0)
    
    # Find bars and estimate heights
    bars = []
    # ... (detection logic)
    
    return {'bars': bars, 'chart_area': f'{width * 0.15},{height * 0.15} to {width * 0.9},{height * 0.9}'}
```

#### Scatter Plot Point Detection
```python
def detect_scatter_points(image_path):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    height, width = img.shape[:2]
    
    # Define color ranges for common plot colors
    colors = {
        'blue': ([100, 80, 80], [140, 255, 255]),
        'red': ([0, 80, 80], [10, 255, 255]),
        'green': ([40, 80, 80], [80, 255, 255])
    }
    
    # Chart area (exclude axes)
    chart_left = int(width * 0.12)
    chart_right = int(width * 0.95)
    chart_top = int(height * 0.1)
    chart_bottom = int(height * 0.9)
    
    points = []
    for color_name, (lower, upper) in colors.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if 30 < cv2.contourArea(cnt) < 300:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    if chart_left < cx < chart_right and chart_top < cy < chart_bottom:
                        points.append({'x': cx, 'y': cy, 'color': color_name})
    
    return {'points': points}
```

#### Heatmap Color Extraction
```python
def extract_heatmap_values(image_path):
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Chart area
    chart_left = int(width * 0.25)
    chart_top = int(height * 0.15)
    
    grid_area = img[chart_top:, chart_left:]
    grid_gray = gray[chart_top:, chart_left:]
    
    _, binary = cv2.threshold(grid_gray, 230, 255, cv2.THRESH_BINARY)
    colored = cv2.bitwise_not(binary)
    
    contours, _ = cv2.findContours(colored, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cells = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 30 < w < 150 and 20 < h < 100:
            b, g, r = grid_area[y:y+h, x:x+w].mean(axis=(0,1))
            # Determine value from color (blue=negative, red=positive)
            value = (r - b) / 255 if r > b else -(b - r) / 255
            cells.append({'col': x, 'row': y, 'correlation': round(value, 2)})
    
    return {'cells': sorted(cells, key=lambda c: (c['row'], c['col']))}
```

---

## Scripts in This Skill

### Main Scripts
1. **`extract_image_data.py`** - EasyOCR text extraction from all images
2. **`extract_chart_data.py`** - OpenCV-based chart analysis
3. **`comprehensive_extract.py`** - Combined OCR + image analysis
4. **`final_data_extraction.py`** - Parse OCR results into structured data

---

## Output Format

The skill produces `extracted_experimental_data.md` with:

```markdown
# Experimental Data Extracted from Figures

## Figure X: [Title]

**Source:** imageN.png  
**Type:** [bar_chart|scatter_plot|heatmap|etc]

### Data

| Column1 | Column2 |
|---------|---------|
| value   | value   |

### Extracted Values
```
[values]
```
```

---

## Tips

1. **Start with OCR**: Extract all text first to identify chart types
2. **Classify images**: Use image size and content to determine chart type
3. **Manual verification**: Always verify extracted values against original images
4. **Color calibration**: Adjust HSV ranges for different chart color schemes
5. **Reference axes**: Always define chart area to exclude axis labels

---

## Limitations

- Scatter plots without labeled points cannot be fully digitized
- Approximate values from bar heights need manual verification
- Complex multi-panel figures require custom extraction logic
- OCR errors in text extraction require manual correction
