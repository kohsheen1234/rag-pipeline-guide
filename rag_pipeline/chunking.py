"""Splitting documents into retrievable units.

Ingestion produces whole documents; embedding models take a bounded amount of
text and retrieval wants passages small enough to be specific. Chunking is the
step in between, and the strategy chosen here decides what retrieval is even
capable of returning.

    chunk_fixed_size  windows of a fixed number of characters
    chunk_by_tokens   windows of a fixed number of tokenizer ids

Sentence and overlapping variants follow.
"""

__all__ = ["chunk_fixed_size", "chunk_by_tokens"]


def chunk_fixed_size(text: str, chunk_size: int) -> list[str]:
    """Split text into consecutive non-overlapping chunks of ``chunk_size``.

    Chunk *i* spans ``text[i * chunk_size : (i + 1) * chunk_size]``, giving
    ``ceil(len(text) / chunk_size)`` chunks. Every chunk has length exactly
    ``chunk_size`` except possibly the last, which holds the remainder.

    Two properties hold for any positive ``chunk_size``, and the tests pin
    both: ``"".join(chunks) == text`` (nothing is lost or duplicated) and every
    chunk satisfies ``1 <= len(chunk) <= chunk_size``.

    Windows do not overlap. Overlap is a separate strategy, not a variation on
    this one.

    Args:
        text: The text to split. Empty text yields an empty list.
        chunk_size: Characters per chunk. Must be positive -- ``0`` raises
            ``ValueError`` from ``range``, and a negative value returns ``[]``,
            silently discarding the text.

    Returns:
        The chunks, in document order.
    """
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def chunk_by_tokens(text: str, tokenizer, max_tokens: int) -> list[str]:
    """Split text into chunks of at most ``max_tokens`` tokenizer ids.

    The text is encoded once, the id sequence is sliced into consecutive
    windows, and each window is decoded back to a string -- so the bound is on
    tokens, the unit a model's context window is actually measured in, rather
    than on characters.

    Unlike :func:`chunk_fixed_size`, the chunks are **not** guaranteed to
    rejoin into the original text. Decoding a slice of ids and re-encoding it
    can produce different ids, because BPE merges that spanned the cut are no
    longer available, and decoding usually normalises whitespace at the seams.
    Reconstruction is approximate.

    Args:
        text: The text to split.
        tokenizer: Anything with Hugging Face's ``encode``/``decode`` surface.
            Note ``encode`` -- calling the tokenizer itself returns a dict of
            tensors, not a flat list of ids.
        max_tokens: Maximum token ids per chunk.

    Returns:
        The decoded chunks, in document order. Empty text yields ``[]``,
        provided the tokenizer does not add special tokens -- see the step doc.
    """
    token_ids = tokenizer.encode(text)

    return [
        tokenizer.decode(token_ids[i : i + max_tokens])
        for i in range(0, len(token_ids), max_tokens)
    ]
