# HTML to Academic DOCX Converter

Converts KaTeX-encoded HTML papers to Microsoft Word documents with native editable equations, scientific formatting, and proper typesetting.

## Overview

This tool transforms HTML academic papers (with KaTeX math) into polished Microsoft Word documents featuring:

- **Native Word equations** (OMML) - Fully editable in Word
- **Scientific paper formatting** - Times New Roman, proper spacing, headers
- **Professional tables** - Full-width, borders, formatted headers
- **Centered figures** - Images properly aligned
- **Justified text** - Body text alignment

## Prerequisites

### Required Tools

1. **Python 3.8+** - Core runtime
2. **Node.js** - For KaTeX CLI
3. **Pandoc** - Document conversion (installed automatically via pypandoc)

### Python Packages

```bash
pip install pypandoc python-docx
```

### Node.js Packages

```bash
npm install katex
```

## Pipeline

The conversion happens in 3 stages:

```
HTML (KaTeX) → HTML (MathML) → DOCX (OMML) → Formatted DOCX
```

### Stage 1: KaTeX → MathML HTML

Converts LaTeX math (`$$...$$`, `$...$`) to MathML using KaTeX:

```bash
node scripts/katex_to_mathml.js --input paper.html --output paper_mathml.html
```

### Stage 2: MathML → Native Word DOCX

Converts MathML to Word's native OMML equations:

```bash
python scripts/convert_to_docx.py --input paper_mathml.html --output paper.docx
```

### Stage 3: Apply Academic Formatting

Applies scientific paper styling:

```bash
python scripts/apply_academic_style.py --input paper.docx --output paper_formatted.docx
```

## Quick Start

### One-Command Conversion

Run the full pipeline:

```bash
python html2doc.py --input paper_v5_final.html --output paper_final.docx
```

### Or Step by Step

```bash
# Stage 1: Convert math to MathML
node scripts/katex_to_mathml.js

# Stage 2: Convert to DOCX with native math
python scripts/convert_to_docx.py

# Stage 3: Apply formatting
python scripts/apply_academic_style.py
```

## Configuration

### Custom Styles

Edit `scripts/academic_styles.py` to customize:

- Font family (default: Times New Roman)
- Font sizes (body: 11pt, headings: various)
- Line spacing (default: 1.5)
- Margins (default: 1 inch)
- Table styling
- Heading formats

### Equation Detection

The script automatically detects:
- Display math: `$$...$$` or `<div class="equation">`
- Inline math: `$...$`

## Troubleshooting

### Equations Not Converting

1. Ensure KaTeX is properly installed: `npm list katex`
2. Check that math uses `$...$` or `$$...$$` format
3. Verify HTML has no syntax errors in math expressions

### Images Not Showing

1. Ensure figures are in relative path from HTML
2. SVG images need to be PNG/JPG for Word compatibility

### Tables Not Full-Width

Tables inherit width from source. Use `--table-width 100` flag:

```bash
python scripts/apply_academic_style.py --input paper.docx --output paper.docx --table-width 100
```

## Output

The final DOCX includes:

| Feature | Status |
|---------|--------|
| Native equations | ✅ Editable in Word |
| Tables | ✅ Full-width, bordered |
| Images | ✅ Centered |
| Typography | ✅ Times New Roman |
| Formatting | ✅ Justified, proper spacing |

## Files

```
html2doc/
├── README.md                 # This file
├── html2doc.py               # Main entry point
└── scripts/
    ├── katex_to_mathml.js    # Stage 1: KaTeX → MathML
    ├── convert_to_docx.py    # Stage 2: MathML → DOCX
    ├── apply_academic_style.py  # Stage 3: Formatting
    └── academic_styles.py    # Style configurations
```
