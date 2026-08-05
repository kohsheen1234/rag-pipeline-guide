"""Splitting documents into retrievable passages.

Every chunker takes text and returns ``list[str]`` in document order, so the
strategy is swappable without anything downstream changing. See docs/chunking/
for the trade-offs between them.
"""

import re

__all__ = [
    "chunk_fixed_size",
    "chunk_by_tokens",
    "chunk_by_sentences",
    "chunk_with_overlap",
    "attach_chunk_metadata",
]

# A run of non-terminator characters, plus any terminators that follow it, so
# the trailing .!? stays attached to its sentence.
SENTENCE_PATTERN = re.compile(r"[^.!?]+[.!?]*")


def chunk_fixed_size(text: str, chunk_size: int) -> list[str]:
    """Split text into consecutive non-overlapping character windows.

    The last chunk holds the remainder, so the chunks rejoin to the original
    exactly. ``chunk_size`` must be positive: zero raises, and a negative value
    returns no chunks at all.
    """
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def chunk_by_tokens(text: str, tokenizer, max_tokens: int) -> list[str]:
    """Split text into windows of at most ``max_tokens`` tokenizer ids.

    Bounds the unit a model's context window is actually measured in. Unlike
    :func:`chunk_fixed_size` the chunks do not rejoin exactly -- decoding a
    slice of ids is approximate at the seams.

    ``tokenizer`` needs Hugging Face's ``encode``/``decode`` surface. Note
    ``encode``: calling the tokenizer returns a dict, not a list of ids.
    """
    token_ids = tokenizer.encode(text)

    return [
        tokenizer.decode(token_ids[i : i + max_tokens])
        for i in range(0, len(token_ids), max_tokens)
    ]


def chunk_by_sentences(text: str, max_chars: int) -> list[str]:
    """Pack whole sentences into chunks of at most ``max_chars`` characters.

    Splits on ``.!?``, keeping the terminator attached, then greedily fills
    each chunk with as many consecutive sentences as fit, joined by a single
    space. A sentence longer than ``max_chars`` is emitted alone rather than
    split, so a chunk can exceed the limit.

    The split is naive: ``"3.5"`` and ``"Dr. Smith"`` are treated as sentence
    boundaries.
    """
    sentences = [match.strip() for match in SENTENCE_PATTERN.findall(text)]
    sentences = [sentence for sentence in sentences if sentence]

    chunks = []
    current = ""

    for sentence in sentences:
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= max_chars:
            current += " " + sentence
        else:
            chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    return chunks


def chunk_with_overlap(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Slide a ``chunk_size`` window over text, advancing by ``chunk_size - overlap``.

    Successive chunks share ``overlap`` characters, so a passage straddling a
    boundary survives intact in at least one of them.

    ``overlap`` must be less than ``chunk_size``: equal raises, greater returns
    no chunks. Near the end the window runs out of text, so the last chunks are
    short and may repeat content already covered.
    """
    step = chunk_size - overlap

    return [text[i : i + chunk_size] for i in range(0, len(text), step)]


def attach_chunk_metadata(chunks: list[str], source: str) -> list[dict]:
    """Wrap each chunk with its source, 0-indexed position, and chunk id.

    The id is ``f"{source}::{position}"`` -- stable, readable, and unaffected
    by two chunks sharing the same text.
    """
    return [
        {
            "text": chunk,
            "source": source,
            "position": position,
            "chunk_id": f"{source}::{position}",
        }
        for position, chunk in enumerate(chunks)
    ]
