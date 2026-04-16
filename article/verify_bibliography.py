"""
Bibliography verification script for article.html.
Extracts all 75 references, queries Crossref via habanero, and flags each as
VERIFIED, PARTIAL, NOT_FOUND, or PLACEHOLDER.
"""

import re
import sys
import os
import json
import time
import unicodedata
from pathlib import Path
from html.parser import HTMLParser
from difflib import SequenceMatcher
from habanero import Crossref

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Configuration ──────────────────────────────────────────────────────────
ARTICLE_PATH = Path(__file__).parent / "article.html"
OUTPUT_JSON = Path(__file__).parent / "bibliography_verification.json"
OUTPUT_MD = Path(__file__).parent / "bibliography_verification.md"
CONTACT_EMAIL = "bibliography-check@example.com"
DELAY_BETWEEN_REQUESTS = 1.2  # seconds, be polite


# ── HTML text extraction helper ────────────────────────────────────────────
class HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self._pieces = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            self._pieces.append(data)

    def get_text(self):
        return "".join(self._pieces)


def html_to_text(html_str: str) -> str:
    extractor = HTMLTextExtractor()
    extractor.feed(html_str)
    return extractor.get_text().strip()


# ── Reference parser ──────────────────────────────────────────────────────
def extract_references(html_path: Path) -> list[dict]:
    """Extract references from the <div class='references'> section."""
    text = html_path.read_text(encoding="utf-8")

    # Find the references div
    m = re.search(r'<div class="references">(.*?)</div>', text, re.DOTALL)
    if not m:
        raise ValueError("Could not find <div class='references'> in the HTML")

    refs_html = m.group(1)

    # Each reference is in a <p> tag
    ref_pattern = re.compile(r"<p>(.*?)</p>", re.DOTALL)
    refs = []
    for p_match in ref_pattern.finditer(refs_html):
        raw = p_match.group(1)
        plain = html_to_text(raw)
        # Extract the reference number
        num_m = re.match(r"\[(\d+)\]\s*", plain)
        if not num_m:
            continue
        ref_num = int(num_m.group(1))
        ref_text = plain[num_m.end():].strip()
        refs.append({"num": ref_num, "raw": ref_text})

    return refs


def parse_reference(ref_text: str) -> dict:
    """Parse a reference string into structured fields."""
    result = {"authors": "", "year": "", "title": "", "venue": ""}

    # Try to extract year
    year_m = re.search(r"\((\d{4})\)", ref_text)
    if year_m:
        result["year"] = year_m.group(1)

    # Split on year to get authors (before) and rest (after)
    if year_m:
        result["authors"] = ref_text[:year_m.start()].strip().rstrip(",").rstrip(".")
        rest = ref_text[year_m.end():].strip().lstrip(".").strip()
    else:
        rest = ref_text

    # Try to split title from venue on period followed by venue in italics
    # But since we have plain text, look for pattern: Title. Venue
    # The title often ends with a period, and the venue follows
    # Try splitting on ". " but be careful with initials
    parts = re.split(r"\.\s+", rest, maxsplit=1)
    if len(parts) >= 2:
        result["title"] = parts[0].strip().strip('"').strip()
        result["venue"] = parts[1].strip().rstrip(".")
    else:
        result["title"] = rest.strip().strip('"').strip()

    return result


# ── Similarity helpers ────────────────────────────────────────────────────
def normalize(s: str) -> str:
    """Normalize a string for fuzzy comparison."""
    s = unicodedata.normalize("NFKD", s)
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def check_placeholder(ref_text: str) -> bool:
    """Check if a reference looks like a placeholder."""
    placeholders = [
        r"et al\.\s*$",            # ends with just "et al."
        r"^\[?\s*TBD\s*\]?$",
        r"^\[?\s*placeholder\s*\]?$",
        r"forthcoming",
    ]
    lower = ref_text.lower().strip()
    # Very short references are suspicious
    if len(lower) < 20:
        return True
    for pat in placeholders:
        if re.search(pat, lower):
            return True
    return False


def is_non_crossref_type(ref_text: str) -> str:
    """Detect if reference is a type unlikely to be in Crossref."""
    lower = ref_text.lower()
    if "doctoral dissertation" in lower or "master's thesis" in lower or "phd thesis" in lower:
        return "thesis"
    if "arxiv" in lower:
        return "arxiv"
    if "pmc/nih" in lower:
        return "pmc"
    return ""


# ── Crossref verification ────────────────────────────────────────────────
def verify_with_crossref(cr: Crossref, parsed: dict, raw: str) -> dict:
    """Query Crossref and assess the match."""
    result = {
        "status": "NOT_FOUND",
        "crossref_title": None,
        "crossref_authors": None,
        "crossref_year": None,
        "crossref_venue": None,
        "crossref_doi": None,
        "title_similarity": 0.0,
        "author_match": False,
        "year_match": False,
        "venue_similarity": 0.0,
        "notes": [],
    }

    query_title = parsed["title"]
    if not query_title or len(query_title) < 5:
        # If title is too short, try the whole reference
        query_title = raw[:120]

    try:
        res = cr.works(query=query_title, limit=3)
    except Exception as e:
        result["notes"].append(f"Crossref query error: {e}")
        return result

    items = res.get("message", {}).get("items", [])
    if not items:
        result["notes"].append("No results from Crossref")
        return result

    best_score = 0.0
    best_item = None

    for item in items:
        cr_title = ""
        if item.get("title"):
            cr_title = item["title"][0] if isinstance(item["title"], list) else item["title"]

        tsim = similarity(parsed["title"], cr_title) if parsed["title"] else 0.0

        # Also check raw title similarity
        tsim2 = similarity(raw[:len(cr_title) + 30], cr_title) if cr_title else 0.0
        tsim = max(tsim, tsim2)

        score = tsim
        if score > best_score:
            best_score = score
            best_item = item

    if not best_item:
        return result

    # Extract Crossref metadata
    cr_title = ""
    if best_item.get("title"):
        cr_title = best_item["title"][0] if isinstance(best_item["title"], list) else best_item["title"]
    result["crossref_title"] = cr_title

    cr_authors_list = []
    for auth in best_item.get("author", []):
        name_parts = []
        if auth.get("given"):
            name_parts.append(auth["given"])
        if auth.get("family"):
            name_parts.append(auth["family"])
        if name_parts:
            cr_authors_list.append(" ".join(name_parts))
    result["crossref_authors"] = cr_authors_list

    cr_year = ""
    for date_field in ["published-print", "published-online", "created"]:
        if best_item.get(date_field, {}).get("date-parts"):
            parts = best_item[date_field]["date-parts"][0]
            if parts and parts[0]:
                cr_year = str(parts[0])
                break
    result["crossref_year"] = cr_year

    cr_venue = ""
    for venue_field in ["container-title", "short-container-title"]:
        if best_item.get(venue_field):
            v = best_item[venue_field]
            cr_venue = v[0] if isinstance(v, list) else v
            break
    result["crossref_venue"] = cr_venue
    result["crossref_doi"] = best_item.get("DOI", "")

    # Compute similarities
    title_sim = similarity(parsed["title"], cr_title) if parsed["title"] and cr_title else 0.0
    result["title_similarity"] = round(title_sim, 3)

    year_match = (parsed["year"] == cr_year) if parsed["year"] and cr_year else False
    result["year_match"] = year_match

    # Author check: see if any author last name from the article appears in Crossref authors
    article_authors_lower = normalize(parsed["authors"])
    author_match = False
    if cr_authors_list and article_authors_lower:
        for cr_auth in cr_authors_list:
            parts = cr_auth.split()
            if parts:
                last = normalize(parts[-1])
                if last and len(last) > 2 and last in article_authors_lower:
                    author_match = True
                    break
    result["author_match"] = author_match

    venue_sim = similarity(parsed["venue"], cr_venue) if parsed["venue"] and cr_venue else 0.0
    result["venue_similarity"] = round(venue_sim, 3)

    # Determine status
    if title_sim >= 0.75 and year_match and author_match:
        result["status"] = "VERIFIED"
    elif title_sim >= 0.55 and (year_match or author_match):
        result["status"] = "PARTIAL"
    elif title_sim >= 0.75:
        result["status"] = "PARTIAL"
    elif title_sim >= 0.45 and year_match and author_match:
        result["status"] = "PARTIAL"
    else:
        result["status"] = "NOT_FOUND"

    return result


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print(f"Reading references from {ARTICLE_PATH}")
    refs = extract_references(ARTICLE_PATH)
    print(f"Found {len(refs)} references\n")

    cr = Crossref(mailto=CONTACT_EMAIL)

    results = []
    for ref in refs:
        num = ref["num"]
        raw = ref["raw"]
        print(f"[{num:2d}/75] Checking: {raw[:80]}...")

        parsed = parse_reference(raw)

        entry = {
            "ref_num": num,
            "raw_text": raw,
            "parsed_authors": parsed["authors"],
            "parsed_year": parsed["year"],
            "parsed_title": parsed["title"],
            "parsed_venue": parsed["venue"],
        }

        # Check for placeholder
        if check_placeholder(raw):
            entry["status"] = "PLACEHOLDER"
            entry["notes"] = ["Reference appears to be a placeholder or incomplete"]
            results.append(entry)
            print(f"       -> PLACEHOLDER")
            time.sleep(0.2)
            continue

        # Check for non-Crossref types
        non_cr = is_non_crossref_type(raw)

        # Query Crossref
        cr_result = verify_with_crossref(cr, parsed, raw)
        entry.update(cr_result)

        # If not found but it's a thesis/arxiv/PMC, note that
        if entry["status"] == "NOT_FOUND" and non_cr:
            entry["notes"].append(f"This appears to be a {non_cr} reference, which may not be indexed in Crossref")
            entry["status"] = "NOT_FOUND"

        results.append(entry)
        print(f"       -> {entry['status']} (title_sim={entry.get('title_similarity', 0):.2f}, year={entry.get('year_match', False)}, author={entry.get('author_match', False)})")

        time.sleep(DELAY_BETWEEN_REQUESTS)

    # Save JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nJSON saved to {OUTPUT_JSON}")

    # Generate summary markdown
    verified = [r for r in results if r["status"] == "VERIFIED"]
    partial = [r for r in results if r["status"] == "PARTIAL"]
    not_found = [r for r in results if r["status"] == "NOT_FOUND"]
    placeholder = [r for r in results if r["status"] == "PLACEHOLDER"]

    # Check for duplicates
    titles_seen = {}
    duplicates = []
    for r in results:
        t = normalize(r.get("parsed_title", ""))
        if t and len(t) > 20:
            if t in titles_seen:
                duplicates.append((titles_seen[t], r["ref_num"], r.get("parsed_title", "")))
            else:
                titles_seen[t] = r["ref_num"]

    md_lines = [
        "# Bibliography Verification Report",
        "",
        f"**Total references:** {len(results)}",
        f"**VERIFIED:** {len(verified)}",
        f"**PARTIAL:** {len(partial)}",
        f"**NOT_FOUND:** {len(not_found)}",
        f"**PLACEHOLDER:** {len(placeholder)}",
        "",
    ]

    if duplicates:
        md_lines.append("## Potential Duplicates")
        md_lines.append("")
        for first, second, title in duplicates:
            md_lines.append(f"- [{first}] and [{second}]: {title}")
        md_lines.append("")

    md_lines.append("## Verified References")
    md_lines.append("")
    for r in verified:
        doi = r.get("crossref_doi", "")
        doi_link = f" [DOI](https://doi.org/{doi})" if doi else ""
        md_lines.append(f"- **[{r['ref_num']}]** {r['raw_text'][:100]}...{doi_link}")
    md_lines.append("")

    md_lines.append("## Partial Matches")
    md_lines.append("")
    for r in partial:
        notes = "; ".join(r.get("notes", []))
        title_sim = r.get("title_similarity", 0)
        yr = r.get("year_match", False)
        au = r.get("author_match", False)
        doi = r.get("crossref_doi", "")
        cr_title = r.get("crossref_title", "")
        md_lines.append(f"- **[{r['ref_num']}]** title_sim={title_sim:.2f}, year_match={yr}, author_match={au}")
        md_lines.append(f"  - Article: {r['raw_text'][:100]}")
        if cr_title:
            md_lines.append(f"  - Crossref: {cr_title[:100]}")
        if doi:
            md_lines.append(f"  - DOI: https://doi.org/{doi}")
        if notes:
            md_lines.append(f"  - Notes: {notes}")
    md_lines.append("")

    md_lines.append("## Not Found")
    md_lines.append("")
    for r in not_found:
        notes = "; ".join(r.get("notes", []))
        title_sim = r.get("title_similarity", 0)
        cr_title = r.get("crossref_title", "")
        md_lines.append(f"- **[{r['ref_num']}]** {r['raw_text'][:120]}")
        if cr_title:
            md_lines.append(f"  - Best Crossref match (sim={title_sim:.2f}): {cr_title[:100]}")
        if notes:
            md_lines.append(f"  - Notes: {notes}")
    md_lines.append("")

    if placeholder:
        md_lines.append("## Placeholders")
        md_lines.append("")
        for r in placeholder:
            md_lines.append(f"- **[{r['ref_num']}]** {r['raw_text'][:120]}")
        md_lines.append("")

    # Future year check
    future_refs = [r for r in results if r.get("parsed_year", "") and int(r.get("parsed_year", "0")) > 2026]
    if future_refs:
        md_lines.append("## Future-Dated References (possibly suspect)")
        md_lines.append("")
        for r in future_refs:
            md_lines.append(f"- **[{r['ref_num']}]** Year: {r['parsed_year']} - {r['raw_text'][:100]}")
        md_lines.append("")

    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"Markdown saved to {OUTPUT_MD}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  VERIFIED:    {len(verified):3d}")
    print(f"  PARTIAL:     {len(partial):3d}")
    print(f"  NOT_FOUND:   {len(not_found):3d}")
    print(f"  PLACEHOLDER: {len(placeholder):3d}")
    print(f"  TOTAL:       {len(results):3d}")
    if duplicates:
        print(f"\n  DUPLICATES DETECTED: {len(duplicates)}")
        for first, second, title in duplicates:
            print(f"    [{first}] == [{second}]: {title[:60]}")


if __name__ == "__main__":
    main()
