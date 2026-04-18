"""
Download oral history transcripts from three sources:
1. Internet Archive - UCLA Oral History Program
2. Kentucky Oral History (Nunn Center / SPOKEdb)
3. Densho Digital Repository (via browser-based approach)

Usage:
    python download_oral_histories.py --source ucla
    python download_oral_histories.py --source kentucky
    python download_oral_histories.py --source densho
    python download_oral_histories.py --source all
"""

import requests
from bs4 import BeautifulSoup
import json
import os
import re
import time
import sys
import argparse
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(r"E:\Projects\ImplicitEntities\data\source_data\new_transcripts")
BASE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

RATE_LIMIT = 1.5  # seconds between requests


def safe_filename(name, max_len=80):
    """Create a safe filename from a title."""
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = re.sub(r'\s+', '_', name.strip())
    name = re.sub(r'_+', '_', name)
    return name[:max_len]


def save_transcript(filepath, text, metadata):
    """Save transcript text and metadata."""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)

    meta_path = filepath.replace('.txt', '_meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return filepath, meta_path


# ============================================================
# SOURCE 1: Internet Archive - UCLA Oral History Program
# ============================================================

def search_archive_org(query, rows=50, page=1):
    """Search Internet Archive for items."""
    url = 'https://archive.org/advancedsearch.php'
    params = {
        'q': query,
        'fl[]': ['identifier', 'title', 'creator', 'date', 'description', 'subject'],
        'sort[]': 'titleSorter asc',
        'rows': rows,
        'page': page,
        'output': 'json'
    }
    r = requests.get(url, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def get_archive_text(identifier):
    """Download the DjVu text version of an Internet Archive item."""
    # The text file follows the pattern: identifier/identifier_djvu.txt
    text_url = f'https://archive.org/download/{identifier}/{identifier}_djvu.txt'
    r = requests.get(text_url, headers=HEADERS, timeout=60)
    if r.status_code == 200 and len(r.text) > 500:
        return r.text

    # Try alternate patterns
    for suffix in ['_djvu.txt', '.txt', '_text.txt']:
        alt_url = f'https://archive.org/download/{identifier}/{identifier}{suffix}'
        r = requests.get(alt_url, headers=HEADERS, timeout=60)
        if r.status_code == 200 and len(r.text) > 500:
            return r.text

    # Try the stream endpoint
    stream_url = f'https://archive.org/stream/{identifier}/{identifier}_djvu.txt'
    r = requests.get(stream_url, headers=HEADERS, timeout=60)
    if r.status_code == 200 and len(r.text) > 500:
        return r.text

    return None


def download_ucla_transcripts(max_items=100):
    """Download UCLA oral history transcripts from Internet Archive."""
    print("\n" + "=" * 60)
    print("DOWNLOADING UCLA ORAL HISTORY TRANSCRIPTS FROM ARCHIVE.ORG")
    print("=" * 60)

    output_dir = BASE_DIR / "ucla"
    output_dir.mkdir(exist_ok=True)

    # Search queries to find UCLA oral history transcripts
    queries = [
        'title:"oral history transcript" AND mediatype:texts AND (creator:"University of California" OR creator:"UCLA")',
        'title:"oral history transcript" AND mediatype:texts AND subject:"Oral histories"',
    ]

    all_items = {}
    for query in queries:
        page = 1
        while len(all_items) < max_items * 2:  # Collect extras since some won't have text
            data = search_archive_org(query, rows=100, page=page)
            docs = data['response']['docs']
            if not docs:
                break
            for doc in docs:
                ident = doc['identifier']
                if ident not in all_items:
                    all_items[ident] = doc
            page += 1
            time.sleep(RATE_LIMIT)
            if page > 15:  # Safety limit
                break

    print(f"Found {len(all_items)} unique items across all queries")

    downloaded = 0
    failed = 0
    metadata_list = []

    for ident, doc in sorted(all_items.items()):
        if downloaded >= max_items:
            break

        title = doc.get('title', ident)
        filename = f"ucla_{safe_filename(title)}.txt"
        filepath = output_dir / filename

        if filepath.exists():
            print(f"  SKIP (exists): {title[:60]}")
            downloaded += 1
            continue

        print(f"  Downloading: {title[:60]}...")
        try:
            text = get_archive_text(ident)
            if text and len(text) > 500:
                meta = {
                    'source': 'internet_archive',
                    'program': 'UCLA Oral History Program',
                    'identifier': ident,
                    'title': title,
                    'creator': doc.get('creator', ''),
                    'date': doc.get('date', ''),
                    'description': doc.get('description', ''),
                    'subject': doc.get('subject', ''),
                    'url': f'https://archive.org/details/{ident}',
                    'download_date': datetime.now().isoformat(),
                    'text_length': len(text)
                }
                save_transcript(str(filepath), text, meta)
                metadata_list.append(meta)
                downloaded += 1
                print(f"    OK: {len(text)} chars")
            else:
                print(f"    SKIP: no text available or too short")
                failed += 1
        except Exception as e:
            print(f"    ERROR: {e}")
            failed += 1

        time.sleep(RATE_LIMIT)

    # Save consolidated metadata
    meta_file = output_dir / "ucla_metadata.json"
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)

    print(f"\nUCLA Summary: {downloaded} downloaded, {failed} failed")
    print(f"Metadata saved to: {meta_file}")
    return downloaded


# ============================================================
# SOURCE 2: Kentucky Oral History (SPOKEdb / Nunn Center)
# ============================================================

def get_kentucky_online_items(max_pages=10):
    """Collect all items from SPOKEdb that have 'Online' access."""
    print("  Collecting Kentucky item listings...")

    all_arks = []  # (ark_path, title)
    online_arks = set()

    for page in range(1, max_pages + 1):
        url = f'https://kentuckyoralhistory.org/items/browse?page={page}'
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code != 200:
            break

        soup = BeautifulSoup(r.text, 'html.parser')

        for a in soup.find_all('a', href=True):
            href = a['href']
            text = a.get_text(strip=True)

            if '/ark:/' in href:
                if text == 'Online':
                    online_arks.add(href)
                elif text and text != 'Request' and len(text) > 10:
                    all_arks.append((href, text))

        print(f"    Page {page}: {len(all_arks)} items, {len(online_arks)} online so far")
        time.sleep(RATE_LIMIT)

    # Keep only online items
    online_items = []
    seen = set()
    for ark, title in all_arks:
        if ark in online_arks and ark not in seen:
            seen.add(ark)
            online_items.append((ark, title))

    print(f"  Total online items: {len(online_items)}")
    return online_items


def get_kentucky_transcript(ark_path):
    """Fetch transcript from a Kentucky oral history item via OHMS viewer."""
    item_url = f'https://kentuckyoralhistory.org{ark_path}'
    r = requests.get(item_url, headers=HEADERS, timeout=30)
    if r.status_code != 200:
        return None, None

    # Extract cachefile reference
    m = re.search(r'cachefile=([^\s"\'&<>]+\.xml)', r.text)
    if not m:
        return None, None

    cachefile = m.group(1)

    # Extract metadata from the item page
    soup = BeautifulSoup(r.text, 'html.parser')
    meta = {}
    for div in soup.find_all('div', class_='element-horizontal'):
        label_div = div.find('div', class_='divTableCell')
        if label_div:
            label = label_div.get_text(strip=True)
            # Get the next cell
            cells = div.find_all('div', class_='divTableCell')
            if len(cells) >= 2:
                value = cells[1].get_text(strip=True)
                if label and value:
                    meta[label.rstrip(':')] = value

    # Fetch the OHMS viewer page
    viewer_url = f'https://nunncenter.net/ohms-spokedb/render.php?cachefile={cachefile}'
    rv = requests.get(viewer_url, headers=HEADERS, timeout=30)
    if rv.status_code != 200:
        return None, meta

    soup2 = BeautifulSoup(rv.text, 'html.parser')

    # Extract transcript lines
    transcript_lines = soup2.find_all('span', class_='transcript-line')
    if not transcript_lines:
        return None, meta

    # Build clean transcript text
    parts = []
    for line in transcript_lines:
        # Remove timestamp links, keep text
        for a_tag in line.find_all('a', class_='jumpLink'):
            a_tag.decompose()
        text = line.get_text(strip=True)
        if text and text != 'No transcript.' and text != 'No transcript':
            parts.append(text)

    full_text = '\n\n'.join(parts)

    if len(full_text) < 100:
        return None, meta

    return full_text, meta


def download_kentucky_transcripts(max_items=80):
    """Download Kentucky oral history transcripts from SPOKEdb."""
    print("\n" + "=" * 60)
    print("DOWNLOADING KENTUCKY ORAL HISTORY TRANSCRIPTS (SPOKEdb)")
    print("=" * 60)

    output_dir = BASE_DIR / "kentucky"
    output_dir.mkdir(exist_ok=True)

    # Get all online items
    online_items = get_kentucky_online_items(max_pages=10)

    downloaded = 0
    failed = 0
    no_transcript = 0
    metadata_list = []

    for ark_path, title in online_items:
        if downloaded >= max_items:
            break

        filename = f"ky_{safe_filename(title)}.txt"
        filepath = output_dir / filename

        if filepath.exists():
            print(f"  SKIP (exists): {title[:60]}")
            downloaded += 1
            continue

        print(f"  Checking: {title[:60]}...")
        try:
            text, item_meta = get_kentucky_transcript(ark_path)

            if text:
                meta = {
                    'source': 'kentucky_oral_history',
                    'program': 'Louie B. Nunn Center for Oral History',
                    'ark': ark_path,
                    'title': title,
                    'url': f'https://kentuckyoralhistory.org{ark_path}',
                    'download_date': datetime.now().isoformat(),
                    'text_length': len(text),
                    'item_metadata': item_meta or {}
                }
                save_transcript(str(filepath), text, meta)
                metadata_list.append(meta)
                downloaded += 1
                print(f"    OK: {len(text)} chars")
            else:
                no_transcript += 1
                if no_transcript <= 5:
                    print(f"    SKIP: no transcript content")
        except Exception as e:
            print(f"    ERROR: {e}")
            failed += 1

        time.sleep(RATE_LIMIT)

    # Save consolidated metadata
    meta_file = output_dir / "kentucky_metadata.json"
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)

    print(f"\nKentucky Summary: {downloaded} downloaded, {no_transcript} no transcript, {failed} failed")
    print(f"Metadata saved to: {meta_file}")
    return downloaded


# ============================================================
# SOURCE 3: Densho Digital Repository
# ============================================================

def download_densho_transcripts(max_items=120):
    """
    Download Densho oral history transcripts.

    Densho blocks programmatic access (403 on all endpoints including API).
    Strategy: Use Internet Archive's cached copies of Densho transcripts,
    and also try the direct media URLs for PDFs.
    """
    print("\n" + "=" * 60)
    print("DOWNLOADING DENSHO ORAL HISTORY TRANSCRIPTS")
    print("=" * 60)

    output_dir = BASE_DIR / "densho"
    output_dir.mkdir(exist_ok=True)

    # Strategy 1: Search Internet Archive for Densho content
    print("  Searching Internet Archive for Densho transcripts...")

    queries = [
        'creator:"Densho" AND mediatype:texts',
        'title:"Densho" AND "oral history" AND mediatype:texts',
        '"Japanese American" AND "oral history transcript" AND mediatype:texts',
        '"Japanese American" AND "internment" AND "oral history" AND mediatype:texts',
    ]

    all_items = {}
    for query in queries:
        try:
            data = search_archive_org(query, rows=100, page=1)
            for doc in data['response']['docs']:
                ident = doc['identifier']
                if ident not in all_items:
                    all_items[ident] = doc
            time.sleep(RATE_LIMIT)
        except Exception as e:
            print(f"    Query error: {e}")

    print(f"  Found {len(all_items)} items on Internet Archive related to Densho/Japanese American oral history")

    # Strategy 2: Try known Densho collection IDs and their media URLs
    # These are the major Densho collections with oral histories
    known_collections = [
        'ddr-densho-1000',   # Densho Visual History Collection
        'ddr-densho-1007',   # Hawaii Nisei Veterans
        'ddr-densho-1012',   # Japanese American Citizens League
        'ddr-densho-1020',   # Office of Redress Administration
        'ddr-densho-1021',   # Manzanar National Historic Site
        'ddr-csujad-29',     # CSU Fullerton Oral History
        'ddr-csujad-30',     # CSU Children's Village
        'ddr-csujad-31',     # CSU Fullerton additional
        'ddr-chi-1',         # Japanese American Service Committee Chicago
        'ddr-manz-1',        # Manzanar
        'ddr-phljacl-1',     # Philadelphia JACL
        'ddr-janm-1',        # Japanese American National Museum
    ]

    # Strategy 3: Try with session and referer headers to access Densho
    session = requests.Session()
    session.headers.update(HEADERS)
    session.headers['Accept'] = 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
    session.headers['Accept-Language'] = 'en-US,en;q=0.5'
    session.headers['Referer'] = 'https://ddr.densho.org/'

    # Try to access a known transcript PDF URL
    print("  Testing Densho direct access...")
    test_url = 'https://ddr.densho.org/media/ddr-csujad-29/ddr-csujad-29-56-1-transcript-7a65479369.pdf'
    try:
        r = session.get(test_url, timeout=15)
        print(f"    Direct PDF access: {r.status_code}")
    except:
        print(f"    Direct PDF access: failed")

    # Download from Internet Archive
    downloaded = 0
    failed = 0
    metadata_list = []

    for ident, doc in sorted(all_items.items()):
        if downloaded >= max_items:
            break

        title = doc.get('title', ident)
        # Skip items that are clearly not oral history transcripts
        title_lower = title.lower()
        if not any(k in title_lower for k in ['oral history', 'interview', 'transcript', 'japanese american', 'internment', 'incarceration']):
            continue

        filename = f"densho_{safe_filename(title)}.txt"
        filepath = output_dir / filename

        if filepath.exists():
            print(f"  SKIP (exists): {title[:60]}")
            downloaded += 1
            continue

        print(f"  Downloading: {title[:60]}...")
        try:
            text = get_archive_text(ident)
            if text and len(text) > 500:
                meta = {
                    'source': 'internet_archive_densho',
                    'program': 'Densho / Japanese American Oral History',
                    'identifier': ident,
                    'title': title,
                    'creator': doc.get('creator', ''),
                    'date': doc.get('date', ''),
                    'description': doc.get('description', ''),
                    'url': f'https://archive.org/details/{ident}',
                    'download_date': datetime.now().isoformat(),
                    'text_length': len(text)
                }
                save_transcript(str(filepath), text, meta)
                metadata_list.append(meta)
                downloaded += 1
                print(f"    OK: {len(text)} chars")
            else:
                print(f"    SKIP: no text available")
                failed += 1
        except Exception as e:
            print(f"    ERROR: {e}")
            failed += 1

        time.sleep(RATE_LIMIT)

    # Save consolidated metadata
    meta_file = output_dir / "densho_metadata.json"
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)

    print(f"\nDensho Summary: {downloaded} downloaded, {failed} failed")
    print(f"Metadata saved to: {meta_file}")
    return downloaded


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Download oral history transcripts')
    parser.add_argument('--source', choices=['ucla', 'kentucky', 'densho', 'all'],
                       default='all', help='Which source to download from')
    parser.add_argument('--max-ucla', type=int, default=100, help='Max UCLA transcripts')
    parser.add_argument('--max-kentucky', type=int, default=80, help='Max Kentucky transcripts')
    parser.add_argument('--max-densho', type=int, default=120, help='Max Densho transcripts')
    args = parser.parse_args()

    total = 0

    if args.source in ('ucla', 'all'):
        total += download_ucla_transcripts(max_items=args.max_ucla)

    if args.source in ('kentucky', 'all'):
        total += download_kentucky_transcripts(max_items=args.max_kentucky)

    if args.source in ('densho', 'all'):
        total += download_densho_transcripts(max_items=args.max_densho)

    print(f"\n{'=' * 60}")
    print(f"TOTAL TRANSCRIPTS DOWNLOADED: {total}")
    print(f"Output directory: {BASE_DIR}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
