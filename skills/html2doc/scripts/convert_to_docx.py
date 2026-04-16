#!/usr/bin/env python3
"""
Convert HTML with MathML to DOCX with native Word equations

Usage: python convert_to_docx.py [--input input.html] [--output output.docx]
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

# Try to import pypandoc, install if missing
try:
    import pypandoc
except ImportError:
    print("Installing pypandoc...")
    os.system("pip install pypandoc")
    import pypandoc

from docx import Document
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT


VALID_PROFILES = ("camera-ready-generic", "review-manuscript")


def strip_html_title_tag(input_file):
    """Create a temporary HTML copy without the head <title> tag.

    Pandoc maps the HTML title metadata into a DOCX Title paragraph, which
    duplicates the manuscript's visible body title (<h1>). Removing the head
    title for conversion keeps the visible manuscript title only once.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        html = f.read()

    sanitized = re.sub(r'<title\b[^>]*>.*?</title>', '', html, flags=re.IGNORECASE | re.DOTALL)

    handle = tempfile.NamedTemporaryFile('w', suffix='.html', delete=False, encoding='utf-8')
    handle.write(sanitized)
    handle.close()
    return handle.name


def ensure_reference_doc(profile):
    """Create or refresh the reference DOCX used by Pandoc."""
    repo_root = Path(__file__).resolve().parents[1]
    ref_script = repo_root / "scripts" / "create_reference_doc.py"
    ref_doc = repo_root / "reference.docx"
    subprocess.run(
        [sys.executable, str(ref_script), "--output", str(ref_doc), "--profile", profile],
        check=True,
    )
    return str(ref_doc)


def convert_html_to_docx(input_file, output_file, profile="camera-ready-generic"):
    """Convert HTML with MathML to DOCX with native Word equations."""
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    
    print(f"Converting: {input_file} -> {output_file}")
    
    reference_doc = ensure_reference_doc(profile)

    # Convert HTML to DOCX using MathML (this creates native OMML equations)
    temp_input = strip_html_title_tag(input_file)
    try:
        output = pypandoc.convert_file(
            temp_input,
            'docx',
            outputfile=output_file,
            extra_args=['--mathml', f'--reference-doc={reference_doc}']
        )
        print("Conversion complete")
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)
    finally:
        if os.path.exists(temp_input):
            os.remove(temp_input)
    
    # Verify conversion
    with zipfile.ZipFile(output_file, 'r') as z:
        doc_xml = z.read('word/document.xml').decode('utf-8')
        has_omml = '<m:oMath' in doc_xml
        eq_count = doc_xml.count('<m:oMath')
        dollar_count = doc_xml.count('\\$')
        
        print(f"\n=== Conversion Results ===")
        print(f"Native Word equations: {eq_count}")
        print(f"Unconverted $ signs: {dollar_count}")
        
        if has_omml:
            print("Status: SUCCESS - Equations are native Word OMML")
        else:
            print("Warning: No native equations found")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Convert HTML with MathML to DOCX with native Word equations'
    )
    parser.add_argument(
        '--input', '-i',
        default='paper_with_mathml.html',
        help='Input HTML file (default: paper_with_mathml.html)'
    )
    parser.add_argument(
        '--output', '-o',
        default='paper_converted.docx',
        help='Output DOCX file (default: paper_converted.docx)'
    )
    parser.add_argument(
        '--profile',
        default='camera-ready-generic',
        choices=VALID_PROFILES,
        help='Formatting profile for the reference DOCX'
    )
    
    args = parser.parse_args()
    
    convert_html_to_docx(args.input, args.output, args.profile)
    
    print("\nNext step: python scripts/apply_academic_style.py")


if __name__ == '__main__':
    main()
