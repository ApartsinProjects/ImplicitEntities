"""Download additional oral history transcripts from Internet Archive (UCLA + other programs)."""

import requests
import json
import os
import re
import time
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(r"E:\Projects\ImplicitEntities\data\source_data\new_transcripts\ucla")
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


def get_text(identifier):
    patterns = [
        f'https://archive.org/download/{identifier}/{identifier}_djvu.txt',
        f'https://archive.org/download/{identifier}/{identifier}.txt',
    ]
    for url in patterns:
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            if r.status_code == 200 and len(r.text) > 500:
                return r.text
        except:
            pass

    # Try file listing
    try:
        r = requests.get(f'https://archive.org/metadata/{identifier}/files', headers=HEADERS, timeout=30)
        if r.status_code == 200:
            data = r.json()
            for f in data.get('result', []):
                name = f.get('name', '')
                if name.endswith('_djvu.txt') or (name.endswith('.txt') and 'meta' not in name.lower() and 'files' not in name.lower()):
                    r2 = requests.get(f'https://archive.org/download/{identifier}/{name}', headers=HEADERS, timeout=60)
                    if r2.status_code == 200 and len(r2.text) > 500:
                        return r2.text
    except:
        pass
    return None


def main():
    print("Collecting additional oral history transcripts from Internet Archive...")

    url = 'https://archive.org/advancedsearch.php'
    all_items = {}

    for page in range(1, 15):
        params = {
            'q': 'title:"oral history transcript" AND mediatype:texts',
            'fl[]': ['identifier', 'title', 'creator', 'date', 'description', 'subject'],
            'rows': 100,
            'page': page,
            'output': 'json'
        }
        r = requests.get(url, params=params, headers=HEADERS, timeout=30)
        data = r.json()
        docs = data['response']['docs']
        if not docs:
            break
        for doc in docs:
            all_items[doc['identifier']] = doc
        time.sleep(0.5)

    print(f"Found {len(all_items)} total items")

    # Check existing
    existing = set()
    for f in os.listdir(BASE_DIR):
        if f.endswith('.txt') and not f.endswith('_meta.json'):
            existing.add(f)
    print(f"Already have {len(existing)} files")

    # Download new ones
    downloaded = 0
    failed = 0
    target = 30  # get 30 more to reach ~114 UCLA
    metadata_list = []

    for ident, doc in sorted(all_items.items()):
        if downloaded >= target:
            break

        title = doc.get('title', ident)
        filename = f"ucla_{safe_filename(title)}.txt"

        if filename in existing:
            continue

        filepath = BASE_DIR / filename
        if filepath.exists():
            continue

        print(f"  [{downloaded+1}/{target}] {title[:65]}...")
        try:
            text = get_text(ident)
            if text and len(text) > 500:
                meta = {
                    'source': 'internet_archive',
                    'program': 'Oral History Program',
                    'identifier': ident,
                    'title': title,
                    'creator': doc.get('creator', ''),
                    'date': doc.get('date', ''),
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
        except Exception as e:
            failed += 1
            print(f"    ERROR: {e}")

        time.sleep(RATE_LIMIT)

    # Update consolidated metadata
    meta_file = BASE_DIR / "ucla_metadata.json"
    existing_meta = []
    if meta_file.exists():
        with open(meta_file, 'r', encoding='utf-8') as f:
            existing_meta = json.load(f)
    existing_meta.extend(metadata_list)
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(existing_meta, f, indent=2, ensure_ascii=False)

    final_count = len([f for f in os.listdir(BASE_DIR) if f.endswith('.txt') and not f.startswith('ucla_meta')])
    print(f"\nAdded {downloaded} new, {failed} failed. Total UCLA files: {final_count}")


if __name__ == '__main__':
    main()
