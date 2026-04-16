#!/usr/bin/env python3
"""
Create a Pandoc reference DOCX tuned for academic manuscript conversion.

Usage:
    python create_reference_doc.py --output html2doc/reference.docx
"""

import argparse
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt


PROFILES = {
    "camera-ready-generic": {
        "font_family": "Times New Roman",
        "body_font_size": 11,
        "title_font_size": 18,
        "author_font_size": 12,
        "affiliation_font_size": 10,
        "heading1_size": 14,
        "heading2_size": 12,
        "heading3_size": 11,
        "caption_font_size": 10,
        "reference_font_size": 10,
        "line_spacing": 1.15,
        "margin_inches": 1.0,
    },
    "review-manuscript": {
        "font_family": "Times New Roman",
        "body_font_size": 11,
        "title_font_size": 18,
        "author_font_size": 12,
        "affiliation_font_size": 10,
        "heading1_size": 14,
        "heading2_size": 12,
        "heading3_size": 11,
        "caption_font_size": 10,
        "reference_font_size": 10,
        "line_spacing": 1.5,
        "margin_inches": 1.0,
    },
}


def ensure_style(styles, name, style_type, base=None):
    try:
        style = styles[name]
    except KeyError:
        style = styles.add_style(name, style_type)
    if base is not None:
        style.base_style = styles[base]
    return style


def configure_paragraph_style(style, font_name, font_size, *, bold=False, italic=False,
                              alignment=None, first_line_indent=None,
                              left_indent=None, line_spacing=None,
                              space_before=0, space_after=0,
                              keep_with_next=False):
    style.font.name = font_name
    style.font.size = Pt(font_size)
    style.font.bold = bold
    style.font.italic = italic

    pf = style.paragraph_format
    if alignment is not None:
        pf.alignment = alignment
    if first_line_indent is not None:
        pf.first_line_indent = first_line_indent
    if left_indent is not None:
        pf.left_indent = left_indent
    if line_spacing is not None:
        pf.line_spacing = line_spacing
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    pf.keep_with_next = keep_with_next


def build_reference_doc(output_path: Path, profile_name: str) -> None:
    config = PROFILES[profile_name]
    doc = Document()

    for section in doc.sections:
        section.top_margin = Inches(config["margin_inches"])
        section.bottom_margin = Inches(config["margin_inches"])
        section.left_margin = Inches(config["margin_inches"])
        section.right_margin = Inches(config["margin_inches"])

    styles = doc.styles

    configure_paragraph_style(
        styles["Normal"],
        config["font_family"],
        config["body_font_size"],
        line_spacing=config["line_spacing"],
        space_after=0,
    )

    configure_paragraph_style(
        styles["Title"],
        config["font_family"],
        config["title_font_size"],
        bold=True,
        alignment=WD_ALIGN_PARAGRAPH.CENTER,
        space_after=8,
    )

    configure_paragraph_style(
        styles["Subtitle"],
        config["font_family"],
        config["author_font_size"],
        alignment=WD_ALIGN_PARAGRAPH.CENTER,
        space_after=4,
    )

    configure_paragraph_style(
        styles["Heading 1"],
        config["font_family"],
        config["heading1_size"],
        bold=True,
        space_before=12,
        space_after=6,
        keep_with_next=True,
    )
    configure_paragraph_style(
        styles["Heading 2"],
        config["font_family"],
        config["heading2_size"],
        bold=True,
        space_before=10,
        space_after=4,
        keep_with_next=True,
    )
    configure_paragraph_style(
        styles["Heading 3"],
        config["font_family"],
        config["heading3_size"],
        bold=True,
        italic=False,
        space_before=8,
        space_after=4,
        keep_with_next=True,
    )

    body_text = ensure_style(styles, "Body Text", WD_STYLE_TYPE.PARAGRAPH, base="Normal")
    configure_paragraph_style(
        body_text,
        config["font_family"],
        config["body_font_size"],
        alignment=WD_ALIGN_PARAGRAPH.JUSTIFY,
        line_spacing=config["line_spacing"],
        space_after=0,
    )

    first_paragraph = ensure_style(styles, "First Paragraph", WD_STYLE_TYPE.PARAGRAPH, base="Body Text")
    configure_paragraph_style(
        first_paragraph,
        config["font_family"],
        config["body_font_size"],
        alignment=WD_ALIGN_PARAGRAPH.JUSTIFY,
        line_spacing=config["line_spacing"],
        space_after=0,
    )

    compact = ensure_style(styles, "Compact", WD_STYLE_TYPE.PARAGRAPH, base="Body Text")
    configure_paragraph_style(
        compact,
        config["font_family"],
        config["body_font_size"],
        alignment=WD_ALIGN_PARAGRAPH.JUSTIFY,
        left_indent=Inches(0.22),
        first_line_indent=Inches(-0.18),
        line_spacing=1.05,
        space_after=2,
    )

    abstract_label = ensure_style(styles, "Abstract Label", WD_STYLE_TYPE.PARAGRAPH, base="Body Text")
    configure_paragraph_style(
        abstract_label,
        config["font_family"],
        config["body_font_size"],
        bold=True,
        alignment=WD_ALIGN_PARAGRAPH.CENTER,
        space_before=4,
        space_after=4,
        keep_with_next=True,
    )

    keywords_style = ensure_style(styles, "Keywords", WD_STYLE_TYPE.PARAGRAPH, base="Body Text")
    configure_paragraph_style(
        keywords_style,
        config["font_family"],
        config["body_font_size"],
        italic=False,
        alignment=WD_ALIGN_PARAGRAPH.JUSTIFY,
        space_after=4,
    )

    image_caption = ensure_style(styles, "Image Caption", WD_STYLE_TYPE.PARAGRAPH, base="Body Text")
    configure_paragraph_style(
        image_caption,
        config["font_family"],
        config["caption_font_size"],
        italic=True,
        alignment=WD_ALIGN_PARAGRAPH.CENTER,
        line_spacing=1.0,
        space_before=4,
        space_after=6,
    )

    table_caption = ensure_style(styles, "Table Caption", WD_STYLE_TYPE.PARAGRAPH, base="Body Text")
    configure_paragraph_style(
        table_caption,
        config["font_family"],
        config["caption_font_size"],
        italic=True,
        alignment=WD_ALIGN_PARAGRAPH.CENTER,
        line_spacing=1.0,
        space_before=8,
        space_after=3,
        keep_with_next=True,
    )

    references_heading = ensure_style(styles, "References Heading", WD_STYLE_TYPE.PARAGRAPH, base="Heading 1")
    configure_paragraph_style(
        references_heading,
        config["font_family"],
        config["heading1_size"],
        bold=True,
        space_before=12,
        space_after=6,
        keep_with_next=True,
    )

    reference_entry = ensure_style(styles, "Reference Entry", WD_STYLE_TYPE.PARAGRAPH, base="Body Text")
    configure_paragraph_style(
        reference_entry,
        config["font_family"],
        config["reference_font_size"],
        alignment=WD_ALIGN_PARAGRAPH.JUSTIFY,
        line_spacing=1.0,
        space_after=2,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))


def main():
    parser = argparse.ArgumentParser(description="Create a Pandoc reference DOCX for academic papers")
    parser.add_argument("--output", "-o", required=True, help="Output DOCX path")
    parser.add_argument(
        "--profile",
        default="camera-ready-generic",
        choices=sorted(PROFILES.keys()),
        help="Formatting profile to embed in the reference DOCX",
    )
    args = parser.parse_args()

    build_reference_doc(Path(args.output), args.profile)
    print(f"Saved reference DOCX: {args.output}")


if __name__ == "__main__":
    main()
