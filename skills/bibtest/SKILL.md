---
name: bibtest
description: Fast bibliography checking, normalization, and hallucinated paper/placeholder detection using Crossref, OpenAlex, and BibTeX parsing.
---

# BibTest - Bibliography Testing Skill

This skill validates, normalizes, and detects hallucinated or placeholder references in bibliographies using three complementary tools:

- **habanero**: Python client for Crossref. Good for bulk bibliographic matching and DOI lookup.
- **pyalex**: Python client for OpenAlex. Good for metadata enrichment and fallback matching.
- **bibtexparser**: Good for reading/writing .bib files once you already have structured metadata.

## Core Capabilities

1. **DOI Validation**: Check if DOIs are valid and resolvable via Crossref
2. **Metadata Enrichment**: Fetch complete metadata (authors, title, journal, year, volume, pages)
3. **Normalization**: Standardize bibliographic entries to consistent format
4. **Hallucination Detection**: Identify fake/placeholder papers that don't exist
5. **Placeholder Detection**: Detect common placeholder patterns like "Author, Year", "[Citation needed]", etc.
6. **BibTeX Processing**: Parse and validate .bib files

## Usage

### Command Line Interface

```bash
# Check a single DOI
python -m bibtest check-doi 10.1038/nature12373

# Validate a BibTeX file
python -m bibtest check-bibtex references.bib

# Check multiple entries from a file
python -m bibtest check-file paper.tex

# Batch check DOIs from a list
python -m bibtest batch-dois dois.txt
```

### Python API

```python
from bibtest import BibliographyChecker

# Initialize checker
checker = BibliographyChecker()

# Check a single DOI
result = checker.check_doi("10.1038/nature12373")
print(result)

# Check a BibTeX entry
result = checker.check_bibtex_entry("@article{key, author=...}")

# Validate entire .bib file
results = checker.check_bibtex_file("references.bib")

# Detect placeholders
placeholders = checker.detect_placeholders([
    "Author, A. (2024). Title. Journal.",
    "[Citation needed]",
    "To be added"
])
```

## Entry Points

- `check-doi`: Validate a single DOI against Crossref
- `check-bibtex`: Parse and validate a .bib file
- `check-file`: Extract and check citations from a document file
- `batch-dois`: Check multiple DOIs from a text file
- `detect-hallucinations`: Identify likely hallucinated references

## Configuration

Optional configuration via environment variables:

- `CROSSREF_USERNAME`: Crossref API username (for higher rate limits)
- `CROSSREF_PASSWORD`: Crossref API password
- `OPENALEX_TOKEN`: OpenAlex API token (optional, for higher rate limits)
- `EMAIL`: Contact email for API requests (polite pool)

## Testing

Run tests with:

```bash
pytest BibTest/
```
