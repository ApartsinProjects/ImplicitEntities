#!/usr/bin/env python3
"""
HTML to Academic DOCX Converter - Main Entry Point

Converts KaTeX-encoded HTML papers to Microsoft Word documents
with native editable equations and scientific formatting.

Usage: python html2doc.py --input paper.html [--output paper.docx]

Requirements:
    pip install pypandoc python-docx
    npm install katex
"""

import argparse
import os
import sys
import subprocess

# Check dependencies
def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []
    
    # Check Python packages
    try:
        import pypandoc
    except ImportError:
        missing.append('pypandoc')
    
    try:
        from docx import Document
    except ImportError:
        missing.append('python-docx')
    
    # Check Node.js package
    if not os.path.exists('node_modules/katex'):
        missing.append('katex (run: npm install katex)')
    
    if missing:
        print("Missing dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall with:")
        print("  pip install pypandoc python-docx")
        print("  npm install katex")
        return False
    
    return True


def run_command(cmd, description):
    """Run a shell command and report status."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=False
    )
    
    if result.returncode != 0:
        print(f"Error: {description} failed")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Convert HTML with KaTeX math to academic DOCX'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input HTML file with KaTeX math'
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Output DOCX file (default: <input>_academic.docx)'
    )
    parser.add_argument(
        '--keep-temp',
        action='store_true',
        help='Keep intermediate files'
    )
    parser.add_argument(
        '--profile',
        default='camera-ready-generic',
        choices=['camera-ready-generic', 'review-manuscript'],
        help='Formatting profile for DOCX conversion'
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        sys.exit(1)
    
    # Set defaults
    input_basename = os.path.splitext(os.path.basename(args.input))[0]
    output_file = args.output or f"{input_basename}_academic.docx"
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    print(f"\nConverting: {args.input}")
    print(f"Output: {output_file}")
    
    # Stage 1: KaTeX to MathML
    mathml_file = f"{input_basename}_mathml.html"
    if not run_command(
        f'node html2doc/scripts/katex_to_mathml.js --input "{args.input}" --output "{mathml_file}"',
        "Stage 1: Converting KaTeX to MathML"
    ):
        sys.exit(1)
    
    # Stage 2: MathML to DOCX
    docx_file = f"{input_basename}_converted.docx"
    if not run_command(
        f'python html2doc/scripts/convert_to_docx.py --input "{mathml_file}" --output "{docx_file}" --profile "{args.profile}"',
        "Stage 2: Converting to DOCX with native equations"
    ):
        sys.exit(1)
    
    # Stage 3: Apply academic formatting
    if not run_command(
        f'python html2doc/scripts/apply_academic_style.py --input "{docx_file}" --output "{output_file}" --profile "{args.profile}"',
        "Stage 3: Applying academic formatting"
    ):
        sys.exit(1)
    
    # Cleanup intermediate files
    if not args.keep_temp:
        print("\nCleaning up intermediate files...")
        for f in [mathml_file, docx_file]:
            if os.path.exists(f):
                os.remove(f)
        # Remove node_modules if created in current dir
        if os.path.exists('node_modules') and not os.path.exists('../node_modules'):
            # Don't remove - might be needed
            pass
    
    print(f"\n{'='*60}")
    print("CONVERSION COMPLETE")
    print(f"{'='*60}")
    print(f"Output: {output_file}")
    print("\nFeatures:")
    print("  - Native Word equations (OMML) - fully editable")
    print("  - Academic formatting (Times New Roman, 1.5 spacing)")
    print("  - Full-width tables with borders")
    print("  - Centered images")
    print("  - Justified text")


if __name__ == '__main__':
    main()
