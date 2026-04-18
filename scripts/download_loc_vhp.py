#!/usr/bin/env python3
"""
Download publicly accessible Veterans History Project transcripts from the
Library of Congress.

What it does
------------
1. Uses the LOC JSON API to enumerate candidate items/resources in the
   Veterans History Project collection.
2. Keeps only pages that look like transcript resources/items.
3. Tries to find direct downloadable transcript files (preferred: text, then PDF).
4. Falls back to saving the page's visible transcript text when a direct file URL
   is not exposed but the page renders transcript text.

Notes
-----
- This only downloads material that is already publicly accessible online.
- Rights for publication/re-distribution are separate from technical access.
- LOC recommends rate limiting; default delay is conservative.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import time
from dataclasses import dataclass
from html import unescape
from typing import Iterable, Iterator, Optional
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

BASE = "https://www.loc.gov"
COLLECTION_API = f"{BASE}/collections/veterans-history-project-collection/"
USER_AGENT = "Mozilla/5.0 (compatible; VHPTranscriptDownloader/1.0; +research use)"


@dataclass
class Candidate:
    title: str
    url: str
    item_id: Optional[str] = None


@dataclass
class DownloadTarget:
    kind: str  # "text", "pdf", "text_from_page"
    url: str
    filename: str
    title: str


def slugify(value: str, max_len: int = 160) -> str:
    value = unescape(value).strip()
    value = re.sub(r"[^\w\-. ]+", "", value, flags=re.UNICODE)
    value = re.sub(r"\s+", "_", value)
    value = value.strip("._")
    return (value or "transcript")[:max_len]


def session_with_headers() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def api_url(page: int, page_size: int, query: str = "transcript") -> str:
    params = {
        "fo": "json",
        "c": str(page_size),
        "sp": str(page),
        "q": query,
        # partof is documented as a facet and the collection endpoint is already scoped,
        # but keeping the query narrowly transcript-focused reduces false positives.
        "at": "results,pagination",
    }
    return f"{COLLECTION_API}?{urlencode(params)}"


def fetch_json(session: requests.Session, url: str) -> dict:
    r = session.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def normalize_result_url(result: dict) -> Optional[str]:
    for key in ("url", "id", "item", "item_url"):
        val = result.get(key)
        if isinstance(val, str) and val.startswith("http"):
            return val
    return None


def result_looks_relevant(result: dict) -> bool:
    text_parts: list[str] = []
    for key in ("title", "description", "item_type"):
        val = result.get(key)
        if isinstance(val, str):
            text_parts.append(val)
        elif isinstance(val, list):
            text_parts.extend(str(x) for x in val)

    for key in ("subject", "partof", "original_format", "online_format"):
        val = result.get(key)
        if isinstance(val, list):
            text_parts.extend(str(x) for x in val)

    hay = " ".join(text_parts).lower()
    return (
        "transcript" in hay
        or "interview transcript" in hay
        or "transcription of audio recording" in hay
    )


def iter_candidates(session: requests.Session, limit: int, page_size: int, query: str) -> Iterator[Candidate]:
    page = 1
    yielded = 0
    while yielded < limit:
        data = fetch_json(session, api_url(page=page, page_size=page_size, query=query))
        results = data.get("results") or []
        if not results:
            break

        for result in results:
            if not result_looks_relevant(result):
                continue
            url = normalize_result_url(result)
            title = result.get("title") or "Untitled"
            if not url:
                continue
            yielded += 1
            yield Candidate(title=title, url=url)
            if yielded >= limit:
                return

        pagination = data.get("pagination") or {}
        if page >= int(pagination.get("total", page)):
            break
        if not pagination.get("next"):
            break
        page += 1


_DIRECT_FILE_RE = re.compile(r"https?://[^\"'\s<>]+\.(?:txt|pdf)(?:\?[^\"'\s<>]*)?", re.I)
_RESOURCE_TRANSCRIPT_RE = re.compile(
    r"https?://www\.loc\.gov/resource/[^\"'\s<>]+(?:pd\d+|pm\d+)/?(?:\?[^\"'\s<>]*)?",
    re.I,
)


def ensure_text_view(url: str) -> str:
    parsed = urlparse(url)
    q = parse_qs(parsed.query)
    q["st"] = ["text"]
    new_query = urlencode({k: v[0] if len(v) == 1 else v for k, v in q.items()}, doseq=True)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment))


def fetch_html(session: requests.Session, url: str) -> str:
    r = session.get(url, timeout=60)
    r.raise_for_status()
    return r.text


def looks_like_transcript_page(html: str) -> bool:
    plain = BeautifulSoup(html, "html.parser").get_text(" ", strip=True).lower()
    signals = [
        "interview transcript",
        "transcription of audio recording",
        "download: text",
        "transcript of interview",
        " transcript ",
    ]
    return any(sig in plain for sig in signals)


def extract_direct_file_links(html: str, page_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    found: list[str] = []

    for tag in soup.find_all(["a", "option"]):
        attr = tag.get("href") or tag.get("value")
        if not attr:
            continue
        attr = urljoin(page_url, attr)
        label = tag.get_text(" ", strip=True).lower()
        if attr.lower().endswith((".txt", ".pdf")):
            found.append(attr)
        elif any(k in label for k in ("text", "pdf", "complete")) and attr.startswith("http"):
            found.append(attr)

    for m in _DIRECT_FILE_RE.finditer(html):
        found.append(m.group(0))

    deduped = []
    seen = set()
    for url in found:
        if url not in seen:
            deduped.append(url)
            seen.add(url)
    return deduped


def extract_transcript_resource_links(html: str, page_url: str) -> list[str]:
    links = []
    for m in _RESOURCE_TRANSCRIPT_RE.finditer(html):
        links.append(urljoin(page_url, m.group(0)))

    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        href = urljoin(page_url, a["href"])
        text = a.get_text(" ", strip=True).lower()
        if "/resource/" in href and ("transcript" in text or re.search(r"(?:pd|pm)\d+", href)):
            links.append(href)

    deduped = []
    seen = set()
    for url in links:
        if url not in seen:
            deduped.append(url)
            seen.add(url)
    return deduped


def extract_visible_transcript_text(html: str) -> Optional[str]:
    soup = BeautifulSoup(html, "html.parser")

    # First try to isolate a transcript block if the page exposes one.
    text = soup.get_text("\n", strip=True)
    m = re.search(r"\bTranscript\b\s*(.*)", text, flags=re.I | re.S)
    if m:
        candidate = m.group(1).strip()
        if len(candidate) >= 200:
            return candidate

    # Fallback: return the visible page text only when it strongly looks like a transcript page.
    plain = re.sub(r"\n{3,}", "\n\n", text).strip()
    if looks_like_transcript_page(html) and len(plain) >= 400:
        return plain
    return None


def choose_download_targets(session: requests.Session, candidate: Candidate) -> list[DownloadTarget]:
    html = fetch_html(session, candidate.url)
    targets: list[DownloadTarget] = []

    direct_links = extract_direct_file_links(html, candidate.url)
    for link in direct_links:
        lower = link.lower()
        if ".txt" in lower:
            targets.append(
                DownloadTarget(
                    kind="text",
                    url=link,
                    filename=f"{slugify(candidate.title)}.txt",
                    title=candidate.title,
                )
            )
        elif ".pdf" in lower:
            targets.append(
                DownloadTarget(
                    kind="pdf",
                    url=link,
                    filename=f"{slugify(candidate.title)}.pdf",
                    title=candidate.title,
                )
            )

    # Transcript resources often live on related /resource/ pages.
    if not targets:
        for resource_url in extract_transcript_resource_links(html, candidate.url):
            try:
                r_html = fetch_html(session, resource_url)
            except requests.RequestException:
                continue

            for link in extract_direct_file_links(r_html, resource_url):
                lower = link.lower()
                if ".txt" in lower:
                    targets.append(
                        DownloadTarget("text", link, f"{slugify(candidate.title)}.txt", candidate.title)
                    )
                elif ".pdf" in lower:
                    targets.append(
                        DownloadTarget("pdf", link, f"{slugify(candidate.title)}.pdf", candidate.title)
                    )

            if targets:
                break

            # Final fallback: text view of the resource page.
            text_view_url = ensure_text_view(resource_url)
            try:
                text_view_html = fetch_html(session, text_view_url)
            except requests.RequestException:
                continue
            transcript_text = extract_visible_transcript_text(text_view_html)
            if transcript_text:
                targets.append(
                    DownloadTarget("text_from_page", text_view_url, f"{slugify(candidate.title)}.txt", candidate.title)
                )
                break

    if not targets:
        # Fallback directly on the candidate page.
        text_view_url = ensure_text_view(candidate.url)
        try:
            text_view_html = fetch_html(session, text_view_url)
        except requests.RequestException:
            text_view_html = ""
        transcript_text = extract_visible_transcript_text(text_view_html)
        if transcript_text:
            targets.append(
                DownloadTarget("text_from_page", text_view_url, f"{slugify(candidate.title)}.txt", candidate.title)
            )

    # Prefer text over PDF if both exist.
    unique: dict[tuple[str, str], DownloadTarget] = {}
    for t in targets:
        unique[(t.kind, t.url)] = t
    ordered = sorted(unique.values(), key=lambda t: {"text": 0, "text_from_page": 1, "pdf": 2}.get(t.kind, 9))
    return ordered


def save_response_bytes(session: requests.Session, url: str, path: pathlib.Path) -> None:
    with session.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def save_text_from_page(session: requests.Session, url: str, path: pathlib.Path) -> None:
    html = fetch_html(session, url)
    text = extract_visible_transcript_text(html)
    if not text:
        raise RuntimeError("Could not extract visible transcript text from page")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def download_candidate(session: requests.Session, candidate: Candidate, out_dir: pathlib.Path, delay: float, dry_run: bool) -> dict:
    result = {
        "title": candidate.title,
        "url": candidate.url,
        "status": "skipped",
        "saved": None,
        "kind": None,
        "error": None,
    }
    try:
        targets = choose_download_targets(session, candidate)
        if not targets:
            result["status"] = "no_transcript_found"
            return result

        target = targets[0]
        out_path = out_dir / target.filename
        if dry_run:
            result.update({"status": "would_download", "saved": str(out_path), "kind": target.kind})
            return result

        if target.kind in {"text", "pdf"}:
            save_response_bytes(session, target.url, out_path)
        elif target.kind == "text_from_page":
            save_text_from_page(session, target.url, out_path)
        else:
            raise RuntimeError(f"Unknown target kind: {target.kind}")

        result.update({"status": "downloaded", "saved": str(out_path), "kind": target.kind})
        return result
    except Exception as e:  # noqa: BLE001 - user-facing batch download script
        result.update({"status": "error", "error": str(e)})
        return result
    finally:
        time.sleep(delay)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Download LOC Veterans History Project transcripts.")
    parser.add_argument("--out", default="vhp_transcripts", help="Output directory")
    parser.add_argument("--limit", type=int, default=25, help="Maximum number of candidate items to inspect")
    parser.add_argument("--page-size", type=int, default=100, help="API page size")
    parser.add_argument("--delay", type=float, default=3.2, help="Delay between page requests in seconds")
    parser.add_argument("--query", default="transcript", help="Search query to seed transcript discovery")
    parser.add_argument("--dry-run", action="store_true", help="List what would be downloaded without writing files")
    parser.add_argument("--log-json", default=None, help="Optional JSONL log file path")
    args = parser.parse_args(argv)

    session = session_with_headers()
    out_dir = pathlib.Path(args.out)
    log_path = pathlib.Path(args.log_json) if args.log_json else None

    count = 0
    for candidate in iter_candidates(session, limit=args.limit, page_size=args.page_size, query=args.query):
        result = download_candidate(session, candidate, out_dir=out_dir, delay=args.delay, dry_run=args.dry_run)
        count += 1
        status = result["status"]
        print(f"[{status}] {candidate.title} :: {candidate.url}")
        if result.get("saved"):
            print(f"  -> {result['saved']} ({result.get('kind')})")
        if result.get("error"):
            print(f"  !! {result['error']}")

        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    if count == 0:
        print("No relevant candidates found.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
