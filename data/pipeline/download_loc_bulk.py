"""
LOC Veterans History Project Bulk Transcript Downloader
=======================================================
Downloads TEI XML transcripts from the Library of Congress VHP collection
using the JSON API (no Cloudflare issues) and tile.loc.gov file server.

Pipeline:
  1. JSON API: list items with subject_format "interview transcript" (3,142 available)
  2. Item JSON: for each item, find XML transcript files on tile.loc.gov
  3. Download XML, parse TEI format, extract speaker turns
  4. Save as clean text with speaker labels

Usage:
  python download_loc_bulk.py --max 50           # Download up to 50 transcripts
  python download_loc_bulk.py --max 200          # Download 200
  python download_loc_bulk.py --list-only        # Just list available items
  python download_loc_bulk.py --resume           # Skip already downloaded
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from xml.etree import ElementTree as ET

import requests

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "source_data" / "new_transcripts"
METADATA_PATH = OUTPUT_DIR / "loc_bulk_metadata.json"

LOC_BASE = "https://www.loc.gov"
COLLECTION = "veterans-history-project-collection"
TILE_BASE = "https://tile.loc.gov"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
}

DELAY = 1.0        # Between API calls
FILE_DELAY = 0.5   # Between file downloads
PAGE_SIZE = 25


# ═══════════════════════════════════════════════════════════════
#  STEP 1: LIST ITEMS WITH TRANSCRIPTS
# ═══════════════════════════════════════════════════════════════

def list_transcript_items(session: requests.Session, max_items: int = 50, start_page: int = 1) -> list[dict]:
    """
    Use LOC JSON API to find VHP items that have interview transcripts.
    Filter: subject_format = "interview transcript"
    """
    items = []
    page = start_page

    while len(items) < max_items:
        url = f"{LOC_BASE}/collections/{COLLECTION}/"
        params = {
            "fa": "subject_format:interview transcript",
            "fo": "json",
            "c": PAGE_SIZE,
            "sp": page,
            "at": "results,pagination",
        }

        print(f"  [API] Page {page}...", end=" ", flush=True)
        try:
            resp = session.get(url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"Error: {e}")
            break

        results = data.get("results", [])
        pagination = data.get("pagination", {})
        total = pagination.get("total", 0)

        if not results:
            print("no results")
            break

        for r in results:
            if len(items) >= max_items:
                break
            item_url = r.get("url", "")
            if not item_url:
                continue
            # Extract item ID from URL
            m = re.search(r"(afc\d+\.\d+)", item_url)
            item_id = m.group(1) if m else item_url.rstrip("/").split("/")[-1]

            items.append({
                "id": item_id,
                "title": r.get("title", ""),
                "url": item_url if item_url.startswith("http") else LOC_BASE + item_url,
                "date": r.get("date", ""),
                "online_format": r.get("online_format", []),
            })

        print(f"{len(results)} items (total available: {total})")

        if page * PAGE_SIZE >= total:
            break
        page += 1
        time.sleep(DELAY)

    print(f"  Listed {len(items)} items (out of {total} available)")
    return items


# ═══════════════════════════════════════════════════════════════
#  STEP 2: FIND XML TRANSCRIPT FILES
# ═══════════════════════════════════════════════════════════════

def find_transcript_files(session: requests.Session, item: dict) -> list[str]:
    """
    Get item JSON and find TEI XML transcript file URLs on tile.loc.gov.
    """
    item_url = item["url"]
    if not item_url.endswith("/"):
        item_url += "/"

    try:
        resp = session.get(item_url, params={"fo": "json"}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return []

    xml_urls = []

    for resource in data.get("resources", []):
        for file_group in resource.get("files", []):
            for f in file_group:
                url = f.get("url", "")
                mime = f.get("mimetype", "")

                # TEI XML transcripts
                if mime == "text/xml" and url:
                    xml_urls.append(url)
                # Also check for .txt files
                elif url.endswith(".txt"):
                    xml_urls.append(url)
                # PDF transcripts (pd000 pattern)
                elif re.search(r"pd\d{3}", url):
                    xml_urls.append(url)

    return xml_urls


# ═══════════════════════════════════════════════════════════════
#  STEP 3: DOWNLOAD AND PARSE TEI XML
# ═══════════════════════════════════════════════════════════════

def parse_tei_xml(xml_text: str) -> dict:
    """
    Parse TEI XML transcript into structured format with speaker turns.
    Returns: {title, speakers, turns: [{speaker, text}], raw_text}
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None

    # Extract title
    title_elem = root.find(".//title")
    title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""

    # Extract speaker turns
    turns = []
    turn_index = 0

    for sp in root.iter("sp"):
        who = sp.get("who", "").strip()
        # Collect all text from <p> elements within this <sp>
        paragraphs = []
        for p in sp.iter("p"):
            # Get all text including tail text from child elements
            text = "".join(p.itertext()).strip()
            if text:
                paragraphs.append(text)

        full_text = " ".join(paragraphs).strip()
        if not full_text:
            continue

        # Infer speaker role from position and content
        # First turn with a question is usually interviewer
        if not who:
            who = f"speaker_{turn_index % 2}"  # Alternate

        turns.append({
            "speaker": who,
            "text": full_text,
            "index": turn_index,
        })
        turn_index += 1

    # Infer roles: interviewer asks questions, veteran answers
    if turns:
        first_has_question = "?" in turns[0]["text"][:200] if turns else False
        for t in turns:
            if first_has_question:
                t["role"] = "interviewer" if t["index"] % 2 == 0 else "veteran"
            else:
                t["role"] = "veteran" if t["index"] % 2 == 0 else "interviewer"

    # Build raw text with speaker labels
    raw_lines = []
    for t in turns:
        label = "Interviewer" if t.get("role") == "interviewer" else "Veteran"
        raw_lines.append(f"{label}: {t['text']}")

    return {
        "title": title,
        "turns": turns,
        "raw_text": "\n\n".join(raw_lines),
        "n_turns": len(turns),
        "n_chars": sum(len(t["text"]) for t in turns),
    }


def download_transcript(session: requests.Session, item: dict, output_dir: Path) -> dict | None:
    """Download and parse a single transcript."""
    item_id = item["id"]
    safe_id = re.sub(r"[^\w.-]", "_", item_id)

    # Check if already downloaded
    txt_path = output_dir / f"loc_{safe_id}.txt"
    if txt_path.exists() and txt_path.stat().st_size > 500:
        return {"id": item_id, "status": "exists", "file": txt_path.name}

    # Find XML files
    xml_urls = find_transcript_files(session, item)
    time.sleep(FILE_DELAY)

    if not xml_urls:
        return {"id": item_id, "status": "no_files"}

    # Try each XML URL
    for url in xml_urls:
        if not url.endswith(".xml"):
            continue

        try:
            resp = session.get(url, timeout=30)
            if resp.status_code != 200:
                continue
        except Exception:
            continue

        time.sleep(FILE_DELAY)

        parsed = parse_tei_xml(resp.text)
        if not parsed or parsed["n_turns"] < 3:
            continue

        # Save as clean text with speaker labels
        txt_path.write_text(parsed["raw_text"], encoding="utf-8")

        return {
            "id": item_id,
            "title": parsed["title"] or item.get("title", ""),
            "source_url": item["url"],
            "xml_url": url,
            "file": txt_path.name,
            "n_turns": parsed["n_turns"],
            "n_chars": parsed["n_chars"],
            "status": "downloaded",
        }

    return {"id": item_id, "status": "no_valid_xml"}


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Bulk download LOC VHP transcripts")
    parser.add_argument("--max", type=int, default=50, help="Max items to process")
    parser.add_argument("--start-page", type=int, default=1, help="API page to start from (skip earlier pages)")
    parser.add_argument("--list-only", action="store_true", help="List items without downloading")
    parser.add_argument("--resume", action="store_true", help="Skip already downloaded")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(HEADERS)

    # Step 1: List items
    print(f"\n[Step 1] Listing VHP items with transcripts (max {args.max}, start page {args.start_page})...")
    items = list_transcript_items(session, max_items=args.max, start_page=args.start_page)

    if args.list_only:
        print(f"\n  {len(items)} items found:")
        for i, item in enumerate(items[:30]):
            print(f"  {i+1:3d}. {item['title'][:60]}")
            print(f"       {item['url']}")
        return

    # Step 2+3: Download each
    print(f"\n[Step 2] Downloading transcripts...")
    results = []
    downloaded = 0
    skipped = 0
    failed = 0

    for i, item in enumerate(items):
        print(f"  [{i+1}/{len(items)}] {item['title'][:50]}...", end=" ", flush=True)

        result = download_transcript(session, item, OUTPUT_DIR)
        results.append(result)

        if result["status"] == "downloaded":
            downloaded += 1
            print(f"OK ({result['n_turns']} turns, {result['n_chars']:,} chars)")
        elif result["status"] == "exists":
            skipped += 1
            print(f"SKIP (exists)")
        else:
            failed += 1
            print(f"{result['status']}")

        # Save metadata periodically
        if (i + 1) % 10 == 0:
            with open(METADATA_PATH, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    # Final save
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Summary
    total_chars = sum(r.get("n_chars", 0) for r in results if r.get("status") == "downloaded")
    total_turns = sum(r.get("n_turns", 0) for r in results if r.get("status") == "downloaded")

    print(f"\n{'='*60}")
    print(f"  LOC VHP BULK DOWNLOAD COMPLETE")
    print(f"  Downloaded: {downloaded} transcripts")
    print(f"  Skipped:    {skipped} (already existed)")
    print(f"  Failed:     {failed} (no XML/too short)")
    print(f"  Total turns: {total_turns:,}")
    print(f"  Total chars: {total_chars:,}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
