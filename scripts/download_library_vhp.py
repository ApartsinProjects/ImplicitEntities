"""
Download VHP transcripts from public library collections.

Sources:
  1. Niles-Maine District Library (~70 interviews)
  2. Indian Prairie Public Library (~12 interviews)

Usage:
  python download_library_vhp.py                # Download all
  python download_library_vhp.py --source niles  # Only Niles
  python download_library_vhp.py --source ippl   # Only IPPL
  python download_library_vhp.py --dry-run       # List URLs only
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import requests

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "source_data" / "new_transcripts"
METADATA_PATH = OUTPUT_DIR / "library_vhp_metadata.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}

DELAY = 2.0  # seconds between requests

# ═══════════════════════════════════════════════════════════════
#  KNOWN TRANSCRIPT URLS
# ═══════════════════════════════════════════════════════════════

NILES_BASE = "https://www.nileslibrary.org/vhp/Searchable/"
NILES_NAMES = [
    "Sol J Schatz",
    "Arthur Shapiro",
    "Donald Spitzer",
    "Chuck Jacobs",
    "Jack Weinberg",
    "Don Lewan",
    "David Besser",
    "Robert Crandall",
    "Matthew Potoczek",
    "John Bugajsky",
    "Albert Dominick",
    "Kenneth Radnitzer",
    # Additional names to try (common VHP participants at Niles)
    "Frank Mariani",
    "George Covent",
    "Harold Berman",
    "Henry Matusiak",
    "Howard Dolan",
    "James Kozak",
    "Jerome Podgers",
    "Joseph Airdo",
    "Larry Weisman",
    "Leonard Biel",
    "Marvin Berman",
    "Michael Harrington",
    "Norman Padnos",
    "Paul Quarnstrom",
    "Ralph Hintz",
    "Richard Rintz",
    "Robert McAllister",
    "Robert Wordel",
    "Ronald Bielski",
    "Sam Harris",
    "Stanley Mroz",
    "Theodore Harczak",
    "Thomas Tarpey",
    "Vernon Schwegel",
    "Walter Blix",
    "Warren Bornhoeft",
    "William Erickson",
    "William Schmidt",
]

IPPL_BASE = "https://ippl.info/sitemedia/documents/learning-research/veterans-history-project/transcripts/"
IPPL_FILES = [
    "vhp_army_bailey.pdf",
    "vhp_army_naughton.pdf",
    "vhp_army_johnson.pdf",
    "vhp_army_alexander.pdf",
    "vhp_army_bing.pdf",
    "vhp_army_hemzy.pdf",
    "vhp_army_kalisik.pdf",
    "vhp_army_radice.pdf",
    "vhp_army_trout.pdf",
    "vhp_army_vandermeer.pdf",
    "vhp_marines_novak.pdf",
    "vhp_navy_sakala.pdf",
]


def download_pdf(url: str, output_path: Path) -> bool:
    """Download a PDF file."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code == 200 and len(resp.content) > 1000:
            output_path.write_bytes(resp.content)
            return True
        else:
            print(f"    HTTP {resp.status_code}, size={len(resp.content)}")
            return False
    except Exception as e:
        print(f"    Error: {e}")
        return False


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from PDF using available libraries."""
    # Try PyPDF2 first
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(str(pdf_path))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        if text.strip():
            return text
    except ImportError:
        pass
    except Exception as e:
        print(f"    PyPDF2 error: {e}")

    # Try pdfminer
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract
        return pdfminer_extract(str(pdf_path))
    except ImportError:
        pass
    except Exception as e:
        print(f"    pdfminer error: {e}")

    # Try pdfplumber
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
        return text
    except ImportError:
        pass

    print(f"    No PDF library available. Install: pip install PyPDF2 pdfminer.six pdfplumber")
    return ""


def download_niles(dry_run: bool = False) -> list[dict]:
    """Download from Niles-Maine District Library."""
    print(f"\n{'='*60}")
    print(f"  NILES-MAINE DISTRICT LIBRARY")
    print(f"  {len(NILES_NAMES)} candidates to try")
    print(f"{'='*60}")

    results = []
    pdf_dir = OUTPUT_DIR / "_pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    for name in NILES_NAMES:
        url = NILES_BASE + name.replace(" ", "%20") + "%20searchable.pdf"
        safe_name = re.sub(r"[^\w]", "_", name)
        txt_name = f"niles_{safe_name}.txt"
        txt_path = OUTPUT_DIR / txt_name

        if txt_path.exists() and txt_path.stat().st_size > 500:
            print(f"  [SKIP] {name} (already exists)")
            results.append({"name": name, "source": "niles", "file": txt_name, "status": "exists"})
            continue

        print(f"  [TRY]  {name}...", end=" ")

        if dry_run:
            print(f"URL: {url}")
            continue

        pdf_path = pdf_dir / f"niles_{safe_name}.pdf"

        if download_pdf(url, pdf_path):
            # Extract text
            text = extract_text_from_pdf(pdf_path)
            if text and len(text.strip()) > 200:
                txt_path.write_text(text, encoding="utf-8")
                print(f"OK ({len(text):,} chars)")
                results.append({
                    "name": name,
                    "source": "niles",
                    "source_url": url,
                    "file": txt_name,
                    "chars": len(text),
                    "status": "downloaded",
                })
            else:
                print(f"PDF extracted but too short ({len(text.strip())} chars)")
                results.append({"name": name, "source": "niles", "status": "empty_pdf"})
        else:
            print(f"FAILED")
            results.append({"name": name, "source": "niles", "status": "not_found"})

        time.sleep(DELAY)

    return results


def download_ippl(dry_run: bool = False) -> list[dict]:
    """Download from Indian Prairie Public Library."""
    print(f"\n{'='*60}")
    print(f"  INDIAN PRAIRIE PUBLIC LIBRARY")
    print(f"  {len(IPPL_FILES)} files")
    print(f"{'='*60}")

    results = []
    pdf_dir = OUTPUT_DIR / "_pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    for fname in IPPL_FILES:
        url = IPPL_BASE + fname
        safe_name = fname.replace(".pdf", "")
        txt_name = f"ippl_{safe_name}.txt"
        txt_path = OUTPUT_DIR / txt_name

        if txt_path.exists() and txt_path.stat().st_size > 500:
            print(f"  [SKIP] {fname} (already exists)")
            results.append({"name": safe_name, "source": "ippl", "file": txt_name, "status": "exists"})
            continue

        print(f"  [TRY]  {fname}...", end=" ")

        if dry_run:
            print(f"URL: {url}")
            continue

        pdf_path = pdf_dir / f"ippl_{fname}"

        if download_pdf(url, pdf_path):
            text = extract_text_from_pdf(pdf_path)
            if text and len(text.strip()) > 200:
                txt_path.write_text(text, encoding="utf-8")
                print(f"OK ({len(text):,} chars)")
                results.append({
                    "name": safe_name,
                    "source": "ippl",
                    "source_url": url,
                    "file": txt_name,
                    "chars": len(text),
                    "status": "downloaded",
                })
            else:
                print(f"PDF extracted but too short")
                results.append({"name": safe_name, "source": "ippl", "status": "empty_pdf"})
        else:
            print(f"FAILED")
            results.append({"name": safe_name, "source": "ippl", "status": "not_found"})

        time.sleep(DELAY)

    # Also try the alternate URL pattern
    IPPL_ALT = "https://www.ippl.info/~bdiwtwtv/sitemedia/documents/learning-research/veterans-history-project/transcripts/"
    for fname in IPPL_FILES:
        safe_name = fname.replace(".pdf", "")
        txt_name = f"ippl_{safe_name}.txt"
        txt_path = OUTPUT_DIR / txt_name
        if txt_path.exists():
            continue

        url = IPPL_ALT + fname
        print(f"  [ALT]  {fname}...", end=" ")
        if dry_run:
            print(f"URL: {url}")
            continue

        pdf_path = pdf_dir / f"ippl_{fname}"
        if download_pdf(url, pdf_path):
            text = extract_text_from_pdf(pdf_path)
            if text and len(text.strip()) > 200:
                txt_path.write_text(text, encoding="utf-8")
                print(f"OK ({len(text):,} chars)")
                results.append({
                    "name": safe_name,
                    "source": "ippl_alt",
                    "source_url": url,
                    "file": txt_name,
                    "chars": len(text),
                    "status": "downloaded",
                })
            else:
                print(f"empty")
        else:
            print(f"FAILED")
        time.sleep(DELAY)

    return results


def main():
    parser = argparse.ArgumentParser(description="Download library VHP transcripts")
    parser.add_argument("--source", choices=["niles", "ippl", "all"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    if args.source in ("niles", "all"):
        all_results.extend(download_niles(dry_run=args.dry_run))

    if args.source in ("ippl", "all"):
        all_results.extend(download_ippl(dry_run=args.dry_run))

    # Save metadata
    if not args.dry_run and all_results:
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Summary
    downloaded = [r for r in all_results if r.get("status") == "downloaded"]
    existed = [r for r in all_results if r.get("status") == "exists"]
    failed = [r for r in all_results if r.get("status") in ("not_found", "empty_pdf")]

    print(f"\n{'='*60}")
    print(f"  DOWNLOAD SUMMARY")
    print(f"  New downloads: {len(downloaded)}")
    print(f"  Already existed: {len(existed)}")
    print(f"  Failed/not found: {len(failed)}")
    if downloaded:
        total_chars = sum(r.get("chars", 0) for r in downloaded)
        print(f"  Total new text: {total_chars:,} chars")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*60}")

    if failed:
        print(f"\n  Failed names (may not exist at this library):")
        for r in failed[:10]:
            print(f"    {r['name']}")


if __name__ == "__main__":
    main()
