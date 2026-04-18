"""Download ALL Niles Library VHP transcripts (70 veterans)."""
import json, os, re, time, requests
from pathlib import Path
from PyPDF2 import PdfReader

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "source_data" / "new_transcripts"
PDF_DIR = OUTPUT_DIR / "_pdfs"

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# All 70 names from the Niles VHP page
NAMES = [
    "Albert Aronson", "Albert Dominick", "Allan Schaefer", "Anthony Dina",
    "Arthur Shapiro", "August Habighurst", "Bernard Warchol", "Bette Horstman",
    "Chuck Jacobs", "Charles Borowsky", "Charles Matz", "Chester Rogala",
    "David Besser", "Dennis Nilsson", "Don Lewan", "Donald Spitzer",
    "Edward Hawker", "Edward Murnane", "Fred Ziegler", "Gary K Warner",
    "Harold Horstman", "Ira Graham", "Irbe Hanson", "Irvin Blaszynski",
    "Irving Abramson", "Irwin Williger", "Jack Weinberg", "Jerry Levin",
    "John B Andres Jr", "John Bugajsky", "John DeCecco", "John McCann",
    "Kenneth Lee", "Kenneth Radnitzer", "Martha Barsky", "Martha Shipp",
    "Martin OGrady", "Martin Passarella", "Matthew Potoczek", "Matthew Wojtaszek",
    "Max Kolpas", "Michael A Tuscano", "Mike Kozyra", "Niels Larsen",
    "Norman Berkman", "Norman Karel", "Orville Skibbe", "Paul Schneller",
    "Peter J Smith", "Ralph Friedman", "Ray Hymen", "Ray Marchetta",
    "Richard Rogala", "Richard Vana", "Robert Barsky", "Robert Crandall",
    "Robert Morris", "Roberts Goldberg", "Roger Salamon", "Rolf Hellman",
    "Russell Zapel", "Sam Schechter", "Seymour Wachtenheim", "Sol J Schatz",
    "Thomas Hill", "Tom Davidson", "Walter Beusse", "Walter Tymczuk",
    "William Carr", "William Shipp",
]

# URL patterns to try for each name
def get_urls(name):
    encoded = name.replace(" ", "%20")
    return [
        f"https://www.nileslibrary.org/vhp/Searchable/{encoded}%20searchable.pdf",
        f"https://www.nileslibrary.org/vhp/{encoded}.pdf",
        f"https://www.nileslibrary.org/vhp/Searchable/{encoded}.pdf",
    ]

def extract_pdf_text(pdf_path):
    try:
        reader = PdfReader(str(pdf_path))
        text = ""
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"
        return text.strip()
    except Exception as e:
        # Try pdfminer as fallback
        try:
            from pdfminer.high_level import extract_text
            return extract_text(str(pdf_path))
        except:
            return ""

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    downloaded = 0
    skipped = 0
    failed = 0

    for name in NAMES:
        safe = re.sub(r"[^\w]", "_", name)
        txt_path = OUTPUT_DIR / f"niles_{safe}.txt"

        if txt_path.exists() and txt_path.stat().st_size > 500:
            print(f"  [SKIP] {name}")
            skipped += 1
            continue

        success = False
        for url in get_urls(name):
            try:
                resp = requests.get(url, headers=HEADERS, timeout=30)
                if resp.status_code == 200 and len(resp.content) > 1000:
                    pdf_path = PDF_DIR / f"niles_{safe}.pdf"
                    pdf_path.write_bytes(resp.content)

                    text = extract_pdf_text(pdf_path)
                    if text and len(text) > 200:
                        txt_path.write_text(text, encoding="utf-8")
                        print(f"  [OK]   {name:30s} ({len(text):,} chars) from {url.split('/')[-1]}")
                        downloaded += 1
                        results.append({"name": name, "chars": len(text), "url": url, "status": "ok"})
                        success = True
                        break
            except Exception as e:
                pass
            time.sleep(1)

        if not success:
            print(f"  [FAIL] {name}")
            failed += 1
            results.append({"name": name, "status": "not_found"})

        time.sleep(1.5)

    print(f"\nSummary: {downloaded} new, {skipped} skipped, {failed} failed")

    meta_path = OUTPUT_DIR / "niles_all_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
