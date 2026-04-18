#!/usr/bin/env python3
"""
Download Veterans History Project transcripts from the Library of Congress
using the JSON API exclusively (no HTML scraping).

Strategy
--------
1. Use the LOC collection JSON API with fa=online-format:online+text to find
   items that have transcript resources.
2. For each item, fetch its JSON metadata to locate text/xml transcript files
   on tile.loc.gov.
3. Download the TEI-XML transcript and convert it to plain text.

This avoids the 403/Cloudflare blocks that occur when fetching HTML pages
from www.loc.gov.

Usage
-----
    python download_loc_vhp_v2.py --out vhp_transcripts --limit 50
    python download_loc_vhp_v2.py --dry-run --limit 10
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from html import unescape
from typing import Iterator, Optional
from urllib.parse import urlencode

import requests

BASE = "https://www.loc.gov"
COLLECTION = "veterans-history-project-collection"
COLLECTION_API = f"{BASE}/collections/{COLLECTION}/"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)

TIMEOUT = 90
MAX_RETRIES = 3
RETRY_BACKOFF = 5


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TranscriptItem:
    title: str
    item_url: str
    item_id: str
    xml_urls: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slugify(value: str, max_len: int = 120) -> str:
    value = unescape(value).strip()
    value = re.sub(r"[^\w\-. ]+", "", value, flags=re.UNICODE)
    value = re.sub(r"\s+", "_", value)
    value = value.strip("._")
    return (value or "transcript")[:max_len]


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def fetch_json(session: requests.Session, url: str) -> Optional[dict]:
    """Fetch JSON with retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, timeout=TIMEOUT)
            if r.status_code == 200:
                return r.json()
            print(f"  HTTP {r.status_code} for {url}", file=sys.stderr)
            return None
        except (requests.RequestException, ValueError) as e:
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF * (attempt + 1)
                print(f"  Retry {attempt+1}/{MAX_RETRIES} after {wait}s: {e}",
                      file=sys.stderr)
                time.sleep(wait)
            else:
                print(f"  Failed after {MAX_RETRIES} attempts: {e}",
                      file=sys.stderr)
    return None


def fetch_bytes(session: requests.Session, url: str) -> Optional[bytes]:
    """Fetch raw bytes with retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, timeout=TIMEOUT)
            if r.status_code == 200:
                return r.content
            print(f"  HTTP {r.status_code} for {url}", file=sys.stderr)
            return None
        except requests.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF * (attempt + 1)
                print(f"  Retry {attempt+1}/{MAX_RETRIES} after {wait}s: {e}",
                      file=sys.stderr)
                time.sleep(wait)
            else:
                print(f"  Failed after {MAX_RETRIES} attempts: {e}",
                      file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
# Collection enumeration (JSON API only)
# ---------------------------------------------------------------------------

def build_search_url(page: int, page_size: int) -> str:
    """Build URL to enumerate VHP items with online text."""
    params = {
        "fo": "json",
        "c": str(page_size),
        "sp": str(page),
        "fa": "online-format:online text",
        "at": "results,pagination",
    }
    return f"{COLLECTION_API}?{urlencode(params)}"


def extract_item_id(url: str) -> str:
    """Extract the afc... item ID from a LOC item URL."""
    # e.g. https://www.loc.gov/item/afc2001001.52884/
    m = re.search(r"(afc\d+\.\d+)", url)
    return m.group(1) if m else url.rstrip("/").split("/")[-1]


def iter_items(session: requests.Session, limit: int,
               page_size: int = 50, delay: float = 3.0
               ) -> Iterator[TranscriptItem]:
    """Iterate over VHP items that have online text, using JSON API."""
    page = 1
    yielded = 0

    while yielded < limit:
        url = build_search_url(page=page, page_size=page_size)
        data = fetch_json(session, url)
        if data is None:
            break

        results = data.get("results") or []
        if not results:
            break

        for result in results:
            if yielded >= limit:
                return

            item_url = result.get("url") or result.get("id") or ""
            if not item_url:
                continue

            title = result.get("title") or "Untitled"
            item_id = extract_item_id(item_url)

            yield TranscriptItem(
                title=title,
                item_url=item_url,
                item_id=item_id,
            )
            yielded += 1

        pagination = data.get("pagination") or {}
        total_pages = int(pagination.get("total", page))
        if page >= total_pages:
            break
        if not pagination.get("next"):
            break

        page += 1
        time.sleep(delay)


# ---------------------------------------------------------------------------
# Item inspection (find XML transcript URLs from JSON metadata)
# ---------------------------------------------------------------------------

def find_transcript_xml_urls(session: requests.Session,
                             item: TranscriptItem, delay: float
                             ) -> list[str]:
    """Fetch item JSON and extract text/xml transcript URLs from resources."""
    json_url = item.item_url.rstrip("/") + "/?fo=json"
    if "?" in item.item_url and "fo=" not in item.item_url:
        json_url = item.item_url + "&fo=json"

    data = fetch_json(session, json_url)
    time.sleep(delay)

    if data is None:
        return []

    xml_urls = []
    for resource in data.get("resources", []):
        label = (resource.get("resource_label") or "").lower()
        # Resources with transcripts typically have "transcript" in their label
        has_transcript_label = "transcript" in label

        for file_group in resource.get("files", []):
            if not isinstance(file_group, list):
                continue
            for f in file_group:
                if not isinstance(f, dict):
                    continue
                furl = f.get("url", "")
                mime = f.get("mimetype", "")
                # Look for text/xml files (TEI transcripts)
                if mime == "text/xml" or (
                    furl.endswith(".xml") and has_transcript_label
                ):
                    xml_urls.append(furl)
                # Also look for plain text or PDF transcript files
                elif mime in ("text/plain",) or furl.endswith(".txt"):
                    xml_urls.append(furl)

    return xml_urls


# ---------------------------------------------------------------------------
# TEI XML to plain text conversion
# ---------------------------------------------------------------------------

def _local(tag: str) -> str:
    """Strip XML namespace from a tag name."""
    return tag.split("}")[-1] if "}" in tag else tag


def _collect_text(elem: ET.Element) -> str:
    """Recursively collect all text content from an element."""
    return "".join(elem.itertext()).strip()


def _walk_tei_body(elem: ET.Element, lines: list[str]) -> None:
    """Recursively walk TEI body, producing one text block per logical unit.

    <sp> (speech turns) are processed as a unit: speaker name on its own
    line, followed by each <p> paragraph. Children of <sp> are NOT visited
    again by the outer loop, which prevents duplication.
    """
    tag = _local(elem.tag) if elem.tag else ""

    if tag == "head":
        t = _collect_text(elem)
        if t:
            lines.append(t)
        return  # don't recurse into head children

    if tag == "sp":
        # Emit speaker + paragraphs, then stop (don't recurse further)
        for child in elem:
            ctag = _local(child.tag) if child.tag else ""
            if ctag == "speaker":
                name = _collect_text(child)
                if name:
                    lines.append(name + ":")
            elif ctag == "p":
                t = _collect_text(child)
                if t:
                    lines.append(t)
        return

    if tag == "p":
        t = _collect_text(elem)
        if t:
            lines.append(t)
        return

    # For container elements (body, div1, div2, text, etc.), recurse into children
    for child in elem:
        _walk_tei_body(child, lines)


def tei_xml_to_text(xml_bytes: bytes) -> str:
    """Convert TEI-XML transcript to plain text.

    The VHP TEI format uses <sp> elements containing <speaker> + <p> children.
    We walk the tree structurally to avoid duplicating text.
    """
    try:
        xml_str = xml_bytes.decode("utf-8", errors="replace")
        xml_str = re.sub(r"<!DOCTYPE[^>]*>", "", xml_str)
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        # Fallback: strip XML tags with regex
        text = xml_bytes.decode("utf-8", errors="replace")
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # Find the body element
    body = None
    for tag_path in ["body", ".//body", ".//{*}body",
                      ".//text/body", ".//{*}text/{*}body"]:
        body = root.find(tag_path)
        if body is not None:
            break

    target = body if body is not None else root

    lines: list[str] = []
    _walk_tei_body(target, lines)

    if not lines:
        # Fallback: extract all text
        text = "".join(root.itertext())
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text

    return "\n\n".join(line for line in lines if line)


# ---------------------------------------------------------------------------
# Download pipeline
# ---------------------------------------------------------------------------

def download_item(session: requests.Session, item: TranscriptItem,
                  out_dir: pathlib.Path, delay: float, dry_run: bool
                  ) -> dict:
    """Download transcript for a single item."""
    result = {
        "title": item.title,
        "item_url": item.item_url,
        "item_id": item.item_id,
        "status": "no_transcript",
        "saved": None,
        "xml_url": None,
        "error": None,
    }

    try:
        # Find XML transcript URLs
        xml_urls = find_transcript_xml_urls(session, item, delay)
        if not xml_urls:
            return result

        xml_url = xml_urls[0]
        result["xml_url"] = xml_url

        filename = f"{item.item_id}_{slugify(item.title)}.txt"
        out_path = out_dir / filename

        if dry_run:
            result["status"] = "would_download"
            result["saved"] = str(out_path)
            return result

        # Download the XML
        xml_bytes = fetch_bytes(session, xml_url)
        if xml_bytes is None:
            result["status"] = "download_failed"
            result["error"] = "Could not fetch XML transcript"
            return result

        # Convert to plain text
        if xml_url.endswith(".xml"):
            text = tei_xml_to_text(xml_bytes)
        else:
            text = xml_bytes.decode("utf-8", errors="replace")

        if len(text.strip()) < 100:
            result["status"] = "too_short"
            result["error"] = f"Transcript text too short ({len(text)} chars)"
            return result

        # Save
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")

        result["status"] = "downloaded"
        result["saved"] = str(out_path)
        return result

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download LOC VHP transcripts via JSON API."
    )
    parser.add_argument(
        "--out",
        default="E:/Projects/ImplicitEntities/data/source_data/new_transcripts",
        help="Output directory",
    )
    parser.add_argument("--limit", type=int, default=25,
                        help="Max items to process")
    parser.add_argument("--page-size", type=int, default=50,
                        help="API page size")
    parser.add_argument("--delay", type=float, default=3.0,
                        help="Delay between requests (seconds)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be downloaded")
    parser.add_argument("--log-json", default=None,
                        help="JSONL log file path")
    parser.add_argument("--start-page", type=int, default=1,
                        help="Start from this search result page")
    args = parser.parse_args(argv)

    session = make_session()
    out_dir = pathlib.Path(args.out)

    stats = {"downloaded": 0, "no_transcript": 0, "error": 0,
             "would_download": 0, "too_short": 0, "download_failed": 0}
    log_path = pathlib.Path(args.log_json) if args.log_json else None

    for item in iter_items(session, limit=args.limit,
                           page_size=args.page_size, delay=args.delay):
        result = download_item(session, item, out_dir, args.delay, args.dry_run)
        status = result["status"]
        stats[status] = stats.get(status, 0) + 1

        marker = {
            "downloaded": "+",
            "would_download": "~",
            "no_transcript": "-",
            "error": "!",
            "too_short": "?",
            "download_failed": "X",
        }.get(status, "?")

        print(f"[{marker}] {item.item_id} | {item.title[:60]}")
        if result.get("saved"):
            print(f"    -> {result['saved']}")
        if result.get("error"):
            print(f"    !! {result['error']}")
        if result.get("xml_url"):
            print(f"    xml: {result['xml_url']}")

        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"\nSummary: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
