"""Document ingestion and preprocessing.

Turns a corpus in whatever shape it exists on disk into a list of document
records for the rest of the pipeline. See docs/ingestion/ for the reasoning
behind each function.
"""

import os
import unicodedata
from html.parser import HTMLParser

__all__ = [
    "load_text_file",
    "load_text_directory",
    "extract_text_from_html",
    "normalize_text",
    "make_document",
]

HIDDEN_ELEMENTS = {"script", "style"}


def load_text_file(path: str) -> str:
    """Read a UTF-8 text file, preserving its contents exactly."""
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def load_text_directory(directory_path: str) -> list[str]:
    """Load every ``.txt`` file in a directory, ordered by filename.

    Sorted before reading so the order is stable across machines, which
    downstream chunk ids and caches depend on. Not recursive.
    """
    texts = []

    for filename in sorted(os.listdir(directory_path)):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            texts.append(load_text_file(file_path))

    return texts


class VisibleTextParser(HTMLParser):
    """Collects text nodes, skipping the contents of HIDDEN_ELEMENTS."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.text_parts = []
        self.hidden_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag.lower() in HIDDEN_ELEMENTS:
            self.hidden_depth += 1

    def handle_endtag(self, tag):
        # Floored at zero so a stray closing tag cannot unbalance the count.
        if tag.lower() in HIDDEN_ELEMENTS and self.hidden_depth > 0:
            self.hidden_depth -= 1

    def handle_data(self, data):
        if self.hidden_depth == 0:
            self.text_parts.append(data)


def extract_text_from_html(html: str) -> str:
    """Extract visible text from HTML, decoding entities.

    Nothing is inserted between text nodes, so adjacent block elements run
    together: ``"<p>one</p><p>two</p>"`` yields ``"onetwo"``.
    """
    parser = VisibleTextParser()
    parser.feed(html)
    parser.close()

    return "".join(parser.text_parts).strip()


def normalize_text(text: str) -> str:
    """NFKC-fold unicode variants, collapse whitespace runs, strip the ends.

    NFKC first, because it can itself produce spaces that then need
    collapsing. Newlines are collapsed too, so paragraph structure is lost.
    """
    text = unicodedata.normalize("NFKC", text)
    return " ".join(text.split())


def make_document(text: str, source: str, title: str) -> dict:
    """Wrap text with its provenance into the pipeline's document record.

    The keys ``text``, ``source``, and ``title`` are a contract every later
    stage depends on. Values are stored as-is; nothing is validated here.
    """
    document = {
        "text": text,
        "source": source,
        "title": title,
    }

    return document
