"""
Download Japanese American oral history transcripts from Internet Archive.
These include Densho-related content and other Japanese American oral histories.
"""

import requests
import json
import os
import re
import time
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(r"E:\Projects\ImplicitEntities\data\source_data\new_transcripts\densho")
BASE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

RATE_LIMIT = 1.5


def safe_filename(name, max_len=80):
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = re.sub(r'\s+', '_', name.strip())
    name = re.sub(r'_+', '_', name)
    return name[:max_len]


def search_ia(query, rows=200, page=1):
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


def get_text(identifier):
    """Try multiple URL patterns to get text content."""
    patterns = [
        f'https://archive.org/download/{identifier}/{identifier}_djvu.txt',
        f'https://archive.org/download/{identifier}/{identifier}.txt',
        f'https://archive.org/download/{identifier}/{identifier}_text.txt',
    ]

    for url in patterns:
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            if r.status_code == 200 and len(r.text) > 200:
                return r.text
        except:
            pass

    # Try listing files and finding any .txt file
    try:
        files_url = f'https://archive.org/metadata/{identifier}/files'
        r = requests.get(files_url, headers=HEADERS, timeout=30)
        if r.status_code == 200:
            data = r.json()
            for f in data.get('result', []):
                name = f.get('name', '')
                if name.endswith('_djvu.txt') or (name.endswith('.txt') and 'meta' not in name.lower()):
                    txt_url = f'https://archive.org/download/{identifier}/{name}'
                    r2 = requests.get(txt_url, headers=HEADERS, timeout=60)
                    if r2.status_code == 200 and len(r2.text) > 200:
                        return r2.text
    except:
        pass

    return None


def main():
    print("=" * 60)
    print("DOWNLOADING JAPANESE AMERICAN ORAL HISTORY TRANSCRIPTS")
    print("=" * 60)

    # Collect all relevant items
    queries = [
        'title:"oral history transcription of" AND mediatype:texts',
        'title:"oral history interview with" AND ("Japanese" OR "Nisei" OR "Issei") AND mediatype:texts',
        'title:"oral history transcript" AND ("Japanese American" OR "internment" OR "relocation") AND mediatype:texts',
        '("Densho" OR "ddr-") AND title:"transcript" AND mediatype:texts',
        'identifier:css_003* AND title:"oral history" AND mediatype:texts',
        'title:"oral history" AND "evacuation" AND "Japanese" AND mediatype:texts',
        'title:"oral history" AND "Manzanar" AND mediatype:texts',
        'title:"oral history" AND "Heart Mountain" AND mediatype:texts',
        'title:"oral history" AND "Tule Lake" AND mediatype:texts',
        'title:"oral history" AND ("Nisei" OR "Issei" OR "Sansei") AND mediatype:texts',
    ]

    all_items = {}
    for query in queries:
        for page in range(1, 6):
            try:
                data = search_ia(query, rows=200, page=page)
                docs = data['response']['docs']
                if not docs:
                    break
                for doc in docs:
                    ident = doc['identifier']
                    title = doc.get('title', '').lower()
                    # Filter: must look like an oral history transcript
                    if any(k in title for k in ['oral history', 'interview', 'transcript']):
                        all_items[ident] = doc
            except Exception as e:
                print(f"  Search error: {e}")
                break
        time.sleep(0.5)

    print(f"Found {len(all_items)} unique oral history transcript items")

    # Download
    downloaded = 0
    failed = 0
    skipped = 0
    metadata_list = []
    max_items = 150

    for ident, doc in sorted(all_items.items()):
        if downloaded >= max_items:
            break

        title = doc.get('title', ident)
        filename = f"densho_{safe_filename(title)}.txt"
        filepath = BASE_DIR / filename

        if filepath.exists():
            print(f"  SKIP (exists): {title[:60]}")
            downloaded += 1
            # Load existing metadata
            meta_path = str(filepath).replace('.txt', '_meta.json')
            if os.path.exists(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    metadata_list.append(json.load(f))
            continue

        print(f"  [{downloaded+1}/{max_items}] {title[:65]}...")
        try:
            text = get_text(ident)
            if text and len(text) > 200:
                meta = {
                    'source': 'internet_archive',
                    'program': 'Japanese American Oral History',
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

                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(text)
                meta_path = str(filepath).replace('.txt', '_meta.json')
                with open(meta_path, 'w', encoding='utf-8') as f:
                    json.dump(meta, f, indent=2, ensure_ascii=False)

                metadata_list.append(meta)
                downloaded += 1
                print(f"    OK: {len(text)} chars")
            else:
                failed += 1
                if failed <= 10:
                    print(f"    SKIP: no text available")
        except Exception as e:
            failed += 1
            print(f"    ERROR: {e}")

        time.sleep(RATE_LIMIT)

    # Save consolidated metadata
    meta_file = BASE_DIR / "densho_metadata.json"
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)

    print(f"\nDensho/JA Summary: {downloaded} downloaded, {failed} failed")
    print(f"Metadata saved to: {meta_file}")


if __name__ == '__main__':
    main()
