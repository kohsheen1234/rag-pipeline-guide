"""Embeddings and corpus storage.

Turns chunks into a dense matrix the retriever can multiply against, and
persists it so a serving process does not re-encode the corpus on every start.
See docs/embeddings/ for the reasoning behind each function.
"""

import json
import os

import numpy as np

__all__ = [
    "load_embedding_model",
    "embed_text",
    "embed_chunks",
    "l2_normalize",
    "save_corpus",
]

EMBEDDINGS_FILE = "embeddings.npy"
CHUNKS_FILE = "chunks.json"


def load_embedding_model(model_name: str):
    """Load a sentence-transformers model by name.

    Loading is expensive -- weights and tokenizer are re-initialised each time
    -- so hold onto the returned instance rather than calling this per query.
    """
    # Imported here so the rest of this module works without torch installed.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_text(model, text: str) -> np.ndarray:
    """Embed one string as a 1D float32 vector of shape ``(d,)``.

    The reshape drops the leading singleton axis that batch-style encoders
    return for a single input.
    """
    vector = model.encode(text)

    return np.asarray(vector, dtype=np.float32).reshape(-1)


def embed_chunks(model, chunks, batch_size: int = 32) -> np.ndarray:
    """Embed chunks into a ``(n_chunks, d)`` float32 matrix, in input order.

    Accepts raw strings or chunk dicts with a ``text`` field, so it works
    before and after :func:`~rag_pipeline.chunking.attach_chunk_metadata`.
    """
    texts = [chunk["text"] if isinstance(chunk, dict) else chunk for chunk in chunks]
    embeddings = model.encode(texts, batch_size=batch_size)

    return np.asarray(embeddings, dtype=np.float32)


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Rescale each row to unit length, so dot products become cosines.

    All-zero rows are left as they are rather than producing NaNs.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)

    return matrix / np.where(norms == 0, 1, norms)


def save_corpus(embeddings: np.ndarray, chunks: list, directory: str) -> dict:
    """Write the matrix and its chunk metadata to ``directory``, then reload both.

    Reloading in the same call is what proves the corpus survives a fresh
    process: row *i* of ``embeddings`` still lines up with ``chunks[i]``.
    """
    os.makedirs(directory, exist_ok=True)
    embeddings_path = os.path.join(directory, EMBEDDINGS_FILE)
    chunks_path = os.path.join(directory, CHUNKS_FILE)

    np.save(embeddings_path, embeddings)
    with open(chunks_path, "w", encoding="utf-8") as file:
        json.dump(chunks, file)

    with open(chunks_path, "r", encoding="utf-8") as file:
        return {"embeddings": np.load(embeddings_path), "chunks": json.load(file)}
