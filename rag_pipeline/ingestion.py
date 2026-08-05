"""Document ingestion and preprocessing.

Turns a corpus in whatever shape it exists on disk into a predictable list of
document records for the rest of the pipeline:

    load_text_file          bytes on disk    -> str
    load_text_directory     a folder         -> list[str], filename order
    extract_text_from_html  markup           -> visible text
    normalize_text          messy str        -> tidy single line
    make_document           str + provenance -> document dict

Two rules hold across all of them. **Read faithfully, normalise explicitly**:
the loaders preserve exactly what was on disk, and every transformation
afterwards is a named function a caller can see in a stack trace and choose to
skip. **Fail loudly**: nothing here swallows an exception to keep a batch
running.
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

#: Elements whose text content is code, not prose, and must not reach the
#: extracted text.
HIDDEN_ELEMENTS = {"script", "style"}


def load_text_file(path: str) -> str:
    """Read a UTF-8 text file and return its full contents as one string.

    Newlines, leading/trailing whitespace, and unicode characters are
    preserved exactly as they appear on disk. Normalisation is
    :func:`normalize_text`'s job; this function's only contract is a faithful
    read.

    Args:
        path: Path to the file to read.

    Returns:
        The entire file contents, newlines and all.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        UnicodeDecodeError: If the file is not valid UTF-8.
    """
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def load_text_directory(directory_path: str) -> list[str]:
    """Load every ``.txt`` file in a directory, ordered by filename.

    Files are sorted lexicographically by filename *before* being read, so the
    returned order is identical on every machine and filesystem. Downstream
    stages derive chunk ids and cache keys from this order, so the stability
    matters more than the particular ordering rule.

    Non-``.txt`` entries are skipped. The scan is not recursive.

    Args:
        directory_path: Path to the directory to scan.

    Returns:
        Contents of each ``.txt`` file, in ascending filename order.

    Raises:
        FileNotFoundError: If ``directory_path`` does not exist.
    """
    texts = []

    for filename in sorted(os.listdir(directory_path)):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            texts.append(load_text_file(file_path))

    return texts


class VisibleTextParser(HTMLParser):
    """Collects the text nodes of an HTML document, skipping hidden elements.

    A depth counter rather than a boolean flag, so that malformed markup
    cannot leave the parser stuck. The counter never goes below zero, so a
    stray ``</script>`` with no opening tag is ignored instead of unbalancing
    everything that follows.

    ``convert_charrefs=True`` makes :meth:`handle_data` receive text with
    character references already decoded, so ``&amp;`` arrives as ``&``.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.text_parts = []
        self.hidden_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag.lower() in HIDDEN_ELEMENTS:
            self.hidden_depth += 1

    def handle_endtag(self, tag):
        if tag.lower() in HIDDEN_ELEMENTS and self.hidden_depth > 0:
            self.hidden_depth -= 1

    def handle_data(self, data):
        if self.hidden_depth == 0:
            self.text_parts.append(data)


def extract_text_from_html(html: str) -> str:
    """Extract the visible text from an HTML string.

    Tags are removed, character references such as ``&amp;`` are decoded, and
    the contents of ``<script>`` and ``<style>`` are skipped so that code does
    not leak into the prose. Comments and doctypes are dropped.

    Text nodes are concatenated with nothing between them, so whatever
    whitespace the markup contained is what separates the words. Adjacent
    block elements therefore run together -- ``"<p>one</p><p>two</p>"`` yields
    ``"onetwo"``. Only leading and trailing whitespace is stripped; interior
    whitespace is left for :func:`normalize_text`.

    Args:
        html: An HTML document or fragment. Malformed markup is tolerated;
            the parser does its best rather than raising.

    Returns:
        The visible text, stripped at both ends.
    """
    parser = VisibleTextParser()
    parser.feed(html)
    parser.close()

    return "".join(parser.text_parts).strip()


def normalize_text(text: str) -> str:
    """Fold unicode variants and collapse whitespace into a single tidy line.

    Applies three transformations, in this order:

    1. **Unicode NFKC normalisation**, mapping compatibility variants onto
       canonical forms -- fullwidth digits to ASCII, the ``fi`` ligature to two
       letters, exotic spaces (NBSP, em space, ideographic space) to ordinary
       ``U+0020``.
    2. **Whitespace collapsing**, replacing any run of whitespace with a single
       space. This includes tabs and newlines, so paragraph structure is
       discarded.
    3. **Stripping**, removing leading and trailing whitespace.

    Steps 2 and 3 both fall out of ``" ".join(text.split())``: ``str.split()``
    with no argument splits on runs of whitespace and drops empty leading and
    trailing fields.

    The NFKC pass must come first. It can itself produce ordinary spaces from
    exotic ones, and those need to be collapsed along with everything else.

    Args:
        text: Raw text, typically straight from a loader.

    Returns:
        A single line with words separated by single spaces, no leading or
        trailing whitespace. Empty or whitespace-only input returns ``""``.
    """
    text = unicodedata.normalize("NFKC", text)
    return " ".join(text.split())


def make_document(text: str, source: str, title: str) -> dict:
    """Wrap raw text together with its provenance into a document dict.

    The returned dict has exactly three top-level keys -- ``text``, ``source``,
    and ``title`` -- inserted in that order. This is the pipeline's document
    contract; every later stage depends on these exact key names at the top
    level, so neither renaming them nor nesting the metadata under a ``meta``
    sub-dict is compatible.

    A fresh dict is built on every call, so callers can mutate the result
    without affecting anything else.

    No validation or defaulting happens here. Whatever is passed in is stored
    as-is, including ``None`` -- the caller owns the question of what counts as
    a usable source or title.

    Args:
        text: The document's payload, typically the output of a loader and
            :func:`normalize_text`.
        source: Where the text came from -- a filename, path, or URL. This is
            what citation logic surfaces to the user.
        title: A human-readable name for the document.

    Returns:
        ``{"text": text, "source": source, "title": title}``.
    """
    document = {
        "text": text,
        "source": source,
        "title": title,
    }

    return document
