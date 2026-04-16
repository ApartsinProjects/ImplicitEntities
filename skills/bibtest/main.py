"""
BibTest - Bibliography Testing Skill

Fast bibliography checking, normalization, and hallucinated paper/placeholder detection
using Crossref, OpenAlex, and BibTeX parsing.
"""

import re
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
from dataclasses import dataclass, field, asdict
from enum import Enum

# Try importing optional dependencies
try:
    import habanero
    from habanero import Crossref
    HabaneroError = Exception
except ImportError:
    habanero = None
    Crossref = None
    HabaneroError = Exception

try:
    import pyalex
    from pyalex import Works
    PyAlexError = Exception
except ImportError:
    pyalex = None
    Works = None
    PyAlexError = Exception

try:
    import bibtexparser
    from bibtexparser.bparser import BibTexParser
    from bibtexparser.customization import author as bib_author
    BibtexparserError = Exception
except ImportError:
    bibtexparser = None
    BibTexParser = None
    bib_author = None
    BibtexparserError = Exception


class ReferenceStatus(Enum):
    VALID = "valid"
    INVALID = "invalid"
    PLACEHOLDER = "placeholder"
    HALLUCINATED = "hallucinated"
    NOT_FOUND = "not_found"
    ERROR = "error"


@dataclass
class ReferenceResult:
    """Result of checking a single reference."""
    key: str
    status: ReferenceStatus
    doi: Optional[str] = None
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    journal: Optional[str] = None
    year: Optional[int] = None
    volume: Optional[str] = None
    pages: Optional[str] = None
    publisher: Optional[str] = None
    message: Optional[str] = None
    source: str = "unknown"  # crossref, openalex, bibtex
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['status'] = self.status.value
        return result


@dataclass
class PlaceholderPattern:
    """A placeholder pattern to detect."""
    pattern: str
    description: str
    regex: re.Pattern = field(init=False)
    
    def __post_init__(self):
        self.regex = re.compile(self.pattern, re.IGNORECASE)


# Common placeholder patterns
DEFAULT_PLACEHOLDERS = [
    PlaceholderPattern(
        r'^\[?\s*citation\s+needed\s*\]?$',
        "Missing citation marker"
    ),
    PlaceholderPattern(
        r'^author[s]?,?\s*[a-z]+\.?[\s,]+.*\d{4}',
        "Generic author-year placeholder"
    ),
    PlaceholderPattern(
        r'^to\s+be\s+added$',
        "To be added placeholder"
    ),
    PlaceholderPattern(
        r'^tbd$',
        "TBD placeholder"
    ),
    PlaceholderPattern(
        r'^xxx+',
        "XXX placeholder"
    ),
    PlaceholderPattern(
        r'^\[\s*\d+\s*\]$',
        "Number-only citation bracket"
    ),
    PlaceholderPattern(
        r'^reference\s+\d+$',
        "Number-only reference"
    ),
    PlaceholderPattern(
        r'^in\s+press$',
        "In press placeholder"
    ),
    PlaceholderPattern(
        r'^submitted$',
        "Submitted placeholder"
    ),
    PlaceholderPattern(
        r'^manuscript\s+in\s+preparation$',
        "Manuscript in preparation"
    ),
]


class BibliographyChecker:
    """Main class for checking bibliography entries."""
    
    def __init__(self, email: Optional[str] = None):
        """
        Initialize the bibliography checker.
        
        Args:
            email: Contact email for API requests (polite pool)
        """
        self.email = email or "bibtest@example.com"
        self._crossref = None
        self._openalex = None
        
    @property
    def crossref(self):
        """Lazy initialization of Crossref client."""
        if self._crossref is None:
            if Crossref is None:
                raise ImportError("habanero not installed. Run: pip install habanero")
            self._crossref = Crossref(mail_to=self.email)
        return self._crossref
    
    @property
    def openalex(self):
        """Lazy initialization of OpenAlex client."""
        if self._openalex is None:
            if Works is None:
                raise ImportError("pyalex not installed. Run: pip install pyalex")
            pyalex.config.email = self.email
            self._openalex = Works()
        return self._openalex
    
    def check_doi(self, doi: str) -> ReferenceResult:
        """
        Check a DOI against Crossref.
        
        Args:
            doi: The DOI to check
            
        Returns:
            ReferenceResult with status and metadata
        """
        # Clean DOI
        doi = self._clean_doi(doi)
        
        try:
            # Try Crossref first
            work = self.crossref.works(ids=doi)
            if work.get('message'):
                return self._parse_crossref_work(doi, work['message'])
            
        except HabaneroError as e:
            pass
        
        try:
            # Fallback to OpenAlex
            work = self.openalex.get(doi=doi)
            if work:
                return self._parse_openalex_work(doi, work)
                
        except PyAlexError:
            pass
        
        # Not found in either
        return ReferenceResult(
            key=doi,
            status=ReferenceStatus.NOT_FOUND,
            doi=doi,
            message="DOI not found in Crossref or OpenAlex"
        )
    
    def check_bibtex_entry(self, entry: str) -> ReferenceResult:
        """
        Check a single BibTeX entry.
        
        Args:
            entry: BibTeX entry string
            
        Returns:
            ReferenceResult
        """
        if bibtexparser is None:
            raise ImportError("bibtexparser not installed. Run: pip install bibtexparser")
        
        parser = BibTexParser(common_strings=True)
        entries = parser.parse(entry)
        
        if not entries:
            return ReferenceResult(
                key="unknown",
                status=ReferenceStatus.INVALID,
                message="Could not parse BibTeX entry"
            )
        
        bib = entries[0]
        key = bib.get('ID', 'unknown')
        
        # Check for DOI in entry
        doi = bib.get('doi') or bib.get('DOI')
        if doi:
            return self.check_doi(doi)
        
        # Try to construct search from metadata
        title = bib.get('title', '')
        authors = bib.get('author', '')
        year = bib.get('year', '')
        
        if not title and not authors:
            return ReferenceResult(
                key=key,
                status=ReferenceStatus.INVALID,
                message="No DOI or sufficient metadata to check"
            )
        
        # Try to find by title
        return self._search_by_metadata(title, authors, year, key)
    
    def check_bibtex_file(self, filepath: Union[str, Path]) -> List[ReferenceResult]:
        """
        Check all entries in a BibTeX file.
        
        Args:
            filepath: Path to .bib file
            
        Returns:
            List of ReferenceResult for each entry
        """
        if bibtexparser is None:
            raise ImportError("bibtexparser not installed. Run: pip install bibtexparser")
        
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"BibTeX file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        parser = BibTexParser(common_strings=True)
        entries = parser.parse(content)
        
        results = []
        for entry in entries:
            key = entry.get('ID', 'unknown')
            
            # Check for DOI
            doi = entry.get('doi') or entry.get('DOI')
            if doi:
                result = self.check_doi(doi)
                result.key = key
                results.append(result)
                continue
            
            # Try metadata search
            title = entry.get('title', '')
            authors = entry.get('author', '')
            year = entry.get('year', '')
            
            if title or authors:
                result = self._search_by_metadata(title, authors, year, key)
                results.append(result)
            else:
                results.append(ReferenceResult(
                    key=key,
                    status=ReferenceStatus.INVALID,
                    message="No DOI or metadata to check"
                ))
        
        return results
    
    def check_file(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract and check citations from a document file.
        
        Args:
            filepath: Path to document (tex, md, etc.)
            
        Returns:
            Dictionary with extracted citations and results
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract DOIs
        dois = self._extract_dois(content)
        
        # Extract BibTeX entries
        bibtex_entries = self._extract_bibtex_entries(content)
        
        results = {
            'dois': [],
            'bibtex': [],
            'placeholders': [],
            'hallucinations': []
        }
        
        # Check DOIs
        for doi in dois:
            result = self.check_doi(doi)
            results['dois'].append(result.to_dict())
        
        # Check BibTeX entries
        for entry in bibtex_entries:
            result = self.check_bibtex_entry(entry)
            results['bibtex'].append(result.to_dict())
        
        # Detect placeholders
        placeholders = self.detect_placeholders([content])
        results['placeholders'] = placeholders
        
        # Detect likely hallucinations (entries with no DOI that couldn't be found)
        for r in results['dois'] + results['bibtex']:
            if r['status'] == 'not_found':
                results['hallucinations'].append(r)
        
        return results
    
    def batch_check_dois(self, dois: List[str]) -> List[ReferenceResult]:
        """
        Check multiple DOIs.
        
        Args:
            dois: List of DOIs to check
            
        Returns:
            List of ReferenceResults
        """
        results = []
        for doi in dois:
            doi = doi.strip()
            if doi:
                result = self.check_doi(doi)
                results.append(result)
        return results
    
    def detect_placeholders(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Detect placeholder citations in text.
        
        Args:
            texts: List of text strings to check
            
        Returns:
            List of found placeholders with their positions
        """
        found = []
        
        for text in texts:
            for placeholder in DEFAULT_PLACEHOLDERS:
                for match in placeholder.regex.finditer(text):
                    found.append({
                        'pattern': placeholder.pattern,
                        'description': placeholder.description,
                        'match': match.group(),
                        'position': match.start()
                    })
        
        return found
    
    def detect_hallucinations(self, results: List[ReferenceResult]) -> List[ReferenceResult]:
        """
        Identify likely hallucinated references from check results.
        
        Args:
            results: List of ReferenceResult to analyze
            
        Returns:
            List of likely hallucinated references
        """
        hallucinations = []
        
        for result in results:
            if result.status == ReferenceStatus.NOT_FOUND:
                # Check if it looks like a fake reference
                if result.title and self._is_likely_fake(result.title):
                    hallucinations.append(result)
                elif not result.doi and not result.title:
                    # No DOI and no title - likely placeholder
                    hallucinations.append(result)
        
        return hallucinations
    
    def _clean_doi(self, doi: str) -> str:
        """Clean DOI string."""
        doi = doi.strip()
        # Remove common prefixes
        for prefix in ['doi:', 'DOI:', 'https://doi.org/', 'http://doi.org/', 'doi.org/']:
            if doi.lower().startswith(prefix):
                doi = doi[len(prefix):]
                break
        return doi.strip()
    
    def _parse_crossref_work(self, doi: str, work: Dict) -> ReferenceResult:
        """Parse Crossref work response."""
        msg = work
        
        authors = []
        if 'author' in msg:
            for a in msg['author']:
                name = a.get('given', '') + ' ' + a.get('family', '')
                authors.append(name.strip())
        
        journal = None
        if 'container-title' in msg:
            journal = msg['container-title'][0]
        elif 'publisher' in msg:
            journal = msg['publisher']
        
        year = None
        if 'published-print' in msg:
            year = msg['published-print'].get('date-parts', [[None]])[0][0]
        elif 'created' in msg:
            year = msg['created'].get('date-parts', [[None]])[0][0]
        
        return ReferenceResult(
            key=doi,
            status=ReferenceStatus.VALID,
            doi=doi,
            title=msg.get('title', [None])[0] if msg.get('title') else None,
            authors=authors,
            journal=journal,
            year=year,
            volume=msg.get('volume'),
            pages=msg.get('page'),
            publisher=msg.get('publisher'),
            source='crossref'
        )
    
    def _parse_openalex_work(self, doi: str, work: Dict) -> ReferenceResult:
        """Parse OpenAlex work response."""
        authors = []
        if 'authorships' in work:
            for a in work['authorships']:
                if 'author' in a:
                    name = a['author'].get('display_name', '')
                    authors.append(name)
        
        journal = None
        if 'host_venue' in work:
            journal = work['host_venue'].get('display_name')
        
        year = work.get('publication_year')
        
        return ReferenceResult(
            key=doi,
            status=ReferenceStatus.VALID,
            doi=doi,
            title=work.get('title'),
            authors=authors,
            journal=journal,
            year=year,
            volume=work.get('biblio', {}).get('volume'),
            pages=work.get('biblio', {}).get('first_page'),
            publisher=work.get('primary_location', {}).get('source', {}).get('publisher'),
            source='openalex'
        )
    
    def _search_by_metadata(self, title: str, authors: str, year: str, key: str) -> ReferenceResult:
        """Search for a work by metadata when DOI is not available."""
        try:
            # Try OpenAlex search
            query = title if title else authors
            if query:
                # OpenAlex search API
                import pyalex
                results = pyalex.search_works(
                    filter={"publication_year": year} if year else {},
                    search=query
                )
                
                if results and results.get('results'):
                    work = results['results'][0]
                    doi = work.get('doi')
                    if doi:
                        return self._parse_openalex_work(doi, work)
                    return self._parse_openalex_work(key, work)
                    
        except PyAlexError:
            pass
        
        return ReferenceResult(
            key=key,
            status=ReferenceStatus.NOT_FOUND,
            title=title or None,
            message="Could not find via metadata search"
        )
    
    def _extract_dois(self, text: str) -> List[str]:
        """Extract DOIs from text."""
        # DOI pattern
        doi_pattern = r'10\.\d{4,}/[^\s]+'
        return re.findall(doi_pattern, text)
    
    def _extract_bibtex_entries(self, text: str) -> List[str]:
        """Extract BibTeX entries from text."""
        if bibtexparser is None:
            return []
        
        entries = []
        # Match @type{key, ...}
        pattern = r'@\w+\{[^@]+'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for match in matches:
            # Close the entry
            if match.count('{') > match.count('}'):
                match += '}'
            entries.append(match)
        
        return entries
    
    def _is_likely_fake(self, title: str) -> bool:
        """Check if a title looks like a hallucination."""
        fake_patterns = [
            r'^paper\s+\d+',
            r'^article\s+\d+',
            r'^study\s+\d+',
            r'^research\s+\d+',
            r'^unknown',
            r'^na$',
            r'^n/?a$',
            r'^none',
            r'^\d+$',
        ]
        
        title_lower = title.lower().strip()
        for pattern in fake_patterns:
            if re.match(pattern, title_lower):
                return True
        
        return False


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='BibTest - Bibliography Testing')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # check-doi
    doi_parser = subparsers.add_parser('check-doi', help='Check a DOI')
    doi_parser.add_argument('doi', help='DOI to check')
    doi_parser.add_argument('--email', help='Contact email', default='bibtest@example.com')
    
    # check-bibtex
    bib_parser = subparsers.add_parser('check-bibtex', help='Check a BibTeX file')
    bib_parser.add_argument('file', help='BibTeX file to check')
    bib_parser.add_argument('--email', help='Contact email', default='bibtest@example.com')
    bib_parser.add_argument('--output', help='Output JSON file')
    
    # check-file
    file_parser = subparsers.add_parser('check-file', help='Check citations in a document')
    file_parser.add_argument('file', help='Document file to check')
    file_parser.add_argument('--email', help='Contact email', default='bibtest@example.com')
    
    # batch-dois
    batch_parser = subparsers.add_parser('batch-dois', help='Check multiple DOIs from file')
    batch_parser.add_argument('file', help='File with DOIs (one per line)')
    batch_parser.add_argument('--email', help='Contact email', default='bibtest@example.com')
    
    # detect-hallucinations
    hall_parser = subparsers.add_parser('detect-hallucinations', help='Detect hallucinated references')
    hall_parser.add_argument('file', help='File with reference results (JSON)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    email = getattr(args, 'email', 'bibtest@example.com')
    checker = BibliographyChecker(email=email)
    
    if args.command == 'check-doi':
        result = checker.check_doi(args.doi)
        print(json.dumps(result.to_dict(), indent=2))
    
    elif args.command == 'check-bibtex':
        results = checker.check_bibtex_file(args.file)
        output = [r.to_dict() for r in results]
        print(json.dumps(output, indent=2))
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(output, f, indent=2)
    
    elif args.command == 'check-file':
        results = checker.check_file(args.file)
        print(json.dumps(results, indent=2))
    
    elif args.command == 'batch-dois':
        with open(args.file, 'r') as f:
            dois = [line.strip() for line in f if line.strip()]
        
        results = checker.batch_check_dois(dois)
        output = [r.to_dict() for r in results]
        print(json.dumps(output, indent=2))
    
    elif args.command == 'detect-hallucinations':
        with open(args.file, 'r') as f:
            data = json.load(f)
        
        # Convert dicts back to ReferenceResult
        results = []
        for item in data:
            status = ReferenceStatus(item['status'])
            result = ReferenceResult(
                key=item['key'],
                status=status,
                doi=item.get('doi'),
                title=item.get('title'),
                message=item.get('message')
            )
            results.append(result)
        
        hallucinations = checker.detect_hallucinations(results)
        output = [h.to_dict() for h in hallucinations]
        print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
