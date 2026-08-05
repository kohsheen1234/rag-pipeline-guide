"""Splitting documents into retrievable passages.

Every chunker takes text and returns ``list[str]`` in document order, so the
strategy is swappable without anything downstream changing. See docs/chunking/
for the trade-offs between them.
"""

__all__ = ["chunk_fixed_size", "chunk_by_tokens"]


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
