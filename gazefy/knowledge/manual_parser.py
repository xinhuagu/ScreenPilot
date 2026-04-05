"""ManualParser: HTML/PDF manual -> indexed text chunks.

Parses software manuals into searchable chunks for OntologyGenerator
and LLM context injection.

Usage:
    parser = ManualParser()
    parser.load_html("packs/vlc/knowledge/html/")
    parser.load_pdf("docs/manual.pdf")
    results = parser.search("play button")
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ManualChunk:
    """A searchable section of a manual."""

    title: str
    body: str
    source_page: str  # file path or "page N"

    @property
    def text(self) -> str:
        """Full text for search."""
        return f"{self.title}\n{self.body}"

    def __repr__(self) -> str:
        return f"ManualChunk(title={self.title!r}, len={len(self.body)}, src={self.source_page!r})"


@dataclass
class ManualParser:
    """Parse HTML/PDF manuals into searchable text chunks."""

    chunks: list[ManualChunk] = field(default_factory=list)
    _idf: dict[str, float] = field(default_factory=dict, repr=False)

    def load_html_dir(self, html_dir: Path | str) -> int:
        """Load all HTML files from a directory tree. Returns count of chunks added."""
        from bs4 import BeautifulSoup

        html_dir = Path(html_dir)
        if not html_dir.exists():
            logger.warning("HTML directory not found: %s", html_dir)
            return 0

        added = 0
        for html_file in sorted(html_dir.rglob("*.html")):
            try:
                raw = html_file.read_text(encoding="utf-8", errors="ignore")
                soup = BeautifulSoup(raw, "lxml")

                # Remove nav, script, style, footer, header
                for tag in soup.find_all(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()

                # Split by heading tags
                sections = self._split_by_headings(soup)
                rel_path = str(html_file.relative_to(html_dir))

                for title, body in sections:
                    body = body.strip()
                    if len(body) < 20:
                        continue
                    self.chunks.append(
                        ManualChunk(
                            title=title,
                            body=body,
                            source_page=rel_path,
                        )
                    )
                    added += 1
            except Exception as e:
                logger.warning("Failed to parse %s: %s", html_file, e)

        self._rebuild_idf()
        logger.info("Loaded %d chunks from %s", added, html_dir)
        return added

    def load_html_file(self, html_path: Path | str) -> int:
        """Load a single HTML file. Returns count of chunks added."""
        from bs4 import BeautifulSoup

        html_path = Path(html_path)
        if not html_path.exists():
            return 0

        raw = html_path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(raw, "lxml")
        for tag in soup.find_all(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        sections = self._split_by_headings(soup)
        added = 0
        for title, body in sections:
            body = body.strip()
            if len(body) < 20:
                continue
            self.chunks.append(
                ManualChunk(
                    title=title,
                    body=body,
                    source_page=str(html_path),
                )
            )
            added += 1

        self._rebuild_idf()
        return added

    def load_pdf(self, pdf_path: Path | str) -> int:
        """Load a PDF file, grouping text by pages. Returns count of chunks added."""
        import pdfplumber

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            logger.warning("PDF not found: %s", pdf_path)
            return 0

        added = 0
        with pdfplumber.open(pdf_path) as pdf:
            current_title = ""
            current_body = ""
            current_page = 1

            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if not text.strip():
                    continue

                lines = text.strip().split("\n")
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    # Heuristic: short ALL CAPS or bold-looking lines are headings
                    if self._looks_like_heading(line):
                        # Save previous section
                        if current_body.strip():
                            self.chunks.append(
                                ManualChunk(
                                    title=current_title,
                                    body=current_body.strip(),
                                    source_page=f"{pdf_path.name} page {current_page}",
                                )
                            )
                            added += 1
                        current_title = line
                        current_body = ""
                        current_page = i + 1
                    else:
                        current_body += line + " "

            # Save last section
            if current_body.strip():
                self.chunks.append(
                    ManualChunk(
                        title=current_title,
                        body=current_body.strip(),
                        source_page=f"{pdf_path.name} page {current_page}",
                    )
                )
                added += 1

        self._rebuild_idf()
        logger.info("Loaded %d chunks from %s", added, pdf_path)
        return added

    def load_knowledge_dir(self, knowledge_dir: Path | str) -> int:
        """Load all HTML and PDF files from a pack's knowledge directory."""
        knowledge_dir = Path(knowledge_dir)
        total = 0
        # Load all HTML subdirectories
        for sub in sorted(knowledge_dir.iterdir()):
            if sub.is_dir():
                total += self.load_html_dir(sub)
        # Load any PDFs directly in knowledge dir
        for pdf_file in sorted(knowledge_dir.glob("*.pdf")):
            total += self.load_pdf(pdf_file)
        return total

    def search(self, query: str, top_k: int = 5) -> list[ManualChunk]:
        """Search chunks by TF-IDF relevance. Returns top-k matches."""
        if not self.chunks:
            return []

        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        scores: list[tuple[float, int]] = []
        for i, chunk in enumerate(self.chunks):
            score = self._tfidf_score(query_terms, chunk.text)
            if score > 0:
                scores.append((score, i))

        scores.sort(reverse=True)
        return [self.chunks[i] for _, i in scores[:top_k]]

    def search_text(self, query: str, top_k: int = 5) -> str:
        """Search and return concatenated text of top matches."""
        results = self.search(query, top_k)
        parts = []
        for chunk in results:
            parts.append(f"### {chunk.title}\n{chunk.body}\n")
        return "\n".join(parts)

    # --- Internal ---

    def _split_by_headings(self, soup) -> list[tuple[str, str]]:
        """Split BeautifulSoup document by h1/h2/h3 headings."""
        heading_tags = {"h1", "h2", "h3"}
        sections: list[tuple[str, str]] = []
        current_title = ""
        current_body = ""

        # Get the main content area
        main = soup.find("main") or soup.find("article") or soup.find("body") or soup
        if main is None:
            return []

        for element in main.children:
            if not hasattr(element, "name"):
                text = str(element).strip()
                if text:
                    current_body += text + " "
                continue

            if element.name in heading_tags:
                # Save previous section
                if current_body.strip():
                    sections.append((current_title, current_body.strip()))
                current_title = element.get_text(strip=True)
                current_body = ""
            else:
                text = element.get_text(separator=" ", strip=True)
                if text:
                    current_body += text + " "

        # Save last section
        if current_body.strip():
            sections.append((current_title, current_body.strip()))

        # If no headings found, treat whole page as one chunk
        if not sections and main:
            text = main.get_text(separator=" ", strip=True)
            if text:
                title_tag = soup.find("title")
                title = title_tag.get_text(strip=True) if title_tag else "Untitled"
                sections.append((title, text))

        return sections

    @staticmethod
    def _looks_like_heading(line: str) -> bool:
        """Heuristic: is this PDF line a section heading?"""
        if len(line) > 80 or len(line) < 3:
            return False
        # ALL CAPS
        if line.isupper() and len(line.split()) <= 8:
            return True
        # Numbered heading: "1.2 Foo Bar"
        if re.match(r"^\d+(\.\d+)*\s+[A-Z]", line):
            return True
        return False

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace + lowercase tokenization."""
        return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1]

    def _rebuild_idf(self) -> None:
        """Rebuild IDF scores from all chunks."""
        n = len(self.chunks)
        if n == 0:
            self._idf = {}
            return

        doc_freq: Counter[str] = Counter()
        for chunk in self.chunks:
            terms = set(self._tokenize(chunk.text))
            for term in terms:
                doc_freq[term] += 1

        self._idf = {term: math.log(n / df) for term, df in doc_freq.items()}

    def _tfidf_score(self, query_terms: list[str], doc_text: str) -> float:
        """Compute TF-IDF score for query against a document."""
        doc_tokens = self._tokenize(doc_text)
        if not doc_tokens:
            return 0.0

        tf: Counter[str] = Counter(doc_tokens)
        doc_len = len(doc_tokens)

        score = 0.0
        for term in query_terms:
            if term in tf:
                term_tf = tf[term] / doc_len
                term_idf = self._idf.get(term, 0.0)
                score += term_tf * term_idf

        return score

    def __len__(self) -> int:
        return len(self.chunks)
