import json
import os

import numpy as np
import pytest

from rag_pipeline.embeddings import (
    CHUNKS_FILE,
    EMBEDDINGS_FILE,
    embed_chunks,
    embed_text,
    l2_normalize,
    load_embedding_model,
    save_corpus,
)


class FakeModel:
    """A stand-in with SentenceTransformer's encode surface.

    Deliberately returns float64 and a leading singleton axis for the
    single-string case, which is the shape real batch-style encoders produce
    and the one embed_text has to flatten.
    """

    def __init__(self, dimension=4):
        self.dimension = dimension
        self.batch_sizes = []

    def encode(self, text, batch_size=32):
        self.batch_sizes.append(batch_size)
        if isinstance(text, str):
            return np.full((1, self.dimension), len(text), dtype=np.float64)
        return np.array(
            [[len(t)] * self.dimension for t in text], dtype=np.float64
        ).reshape(len(text), self.dimension)

    def get_sentence_embedding_dimension(self):
        return self.dimension


@pytest.fixture
def model():
    return FakeModel()


# --- load_embedding_model ---


def test_importing_the_module_does_not_require_sentence_transformers():
    """The import is lazy, so l2_normalize and save_corpus work without torch."""
    import rag_pipeline.embeddings as module

    assert "sentence_transformers" not in dir(module)


def test_loads_a_real_model():
    """Skipped unless sentence-transformers is installed; downloads weights."""
    pytest.importorskip("sentence_transformers")

    model = load_embedding_model("sentence-transformers/all-MiniLM-L6-v2")

    assert type(model).__name__ == "SentenceTransformer"
    assert model.get_sentence_embedding_dimension() == 384


# --- embed_text ---


def test_returns_a_one_dimensional_vector(model):
    assert embed_text(model, "hello world").shape == (4,)


def test_flattens_a_leading_singleton_axis(model):
    """The fake returns (1, d); the contract is (d,)."""
    assert model.encode("hello").shape == (1, 4)
    assert embed_text(model, "hello").ndim == 1


def test_returns_float32(model):
    assert embed_text(model, "hello").dtype == np.float32


def test_casts_from_float64(model):
    assert model.encode("hello").dtype == np.float64
    assert embed_text(model, "hello").dtype == np.float32


def test_preserves_the_values(model):
    assert embed_text(model, "abc").tolist() == [3.0, 3.0, 3.0, 3.0]


def test_embeds_the_empty_string(model):
    assert embed_text(model, "").shape == (4,)


# --- embed_chunks ---


def test_returns_a_two_dimensional_matrix(model):
    assert embed_chunks(model, ["hello world", "goodbye world"]).shape == (2, 4)


def test_matrix_is_float32(model):
    assert embed_chunks(model, ["a", "b"]).dtype == np.float32


def test_accepts_raw_strings(model):
    assert embed_chunks(model, ["a", "bb"]).tolist() == [
        [1.0, 1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0, 2.0],
    ]


def test_accepts_chunk_dicts(model):
    assert embed_chunks(model, [{"text": "a"}, {"text": "bb"}]).tolist() == [
        [1.0, 1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0, 2.0],
    ]


def test_strings_and_dicts_give_the_same_matrix(model):
    from_strings = embed_chunks(model, ["a", "bb"])
    from_dicts = embed_chunks(model, [{"text": "a"}, {"text": "bb"}])

    assert np.array_equal(from_strings, from_dicts)


def test_ignores_other_metadata_fields(model):
    chunk = {"text": "a", "source": "d", "position": 0, "chunk_id": "d::0"}

    assert embed_chunks(model, [chunk]).tolist() == embed_chunks(model, ["a"]).tolist()


def test_row_order_matches_input_order(model):
    matrix = embed_chunks(model, ["a", "bbb", "cc"])

    assert [row[0] for row in matrix.tolist()] == [1.0, 3.0, 2.0]


def test_batch_size_is_passed_through(model):
    embed_chunks(model, ["a"], batch_size=8)

    assert model.batch_sizes == [8]


def test_batch_size_defaults_to_32(model):
    embed_chunks(model, ["a"])

    assert model.batch_sizes == [32]


def test_a_single_chunk_still_gives_a_two_dimensional_matrix(model):
    assert embed_chunks(model, ["only"]).shape == (1, 4)


def test_embeds_a_whole_pipeline_output(model):
    from rag_pipeline.chunking import attach_chunk_metadata, chunk_fixed_size

    records = attach_chunk_metadata(chunk_fixed_size("abcdefgh", 3), "doc1")

    assert embed_chunks(model, records).shape == (3, 4)


# --- l2_normalize ---


def test_documented_example():
    matrix = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 0.0]])

    assert l2_normalize(matrix).tolist() == [[0.6, 0.8], [1.0, 0.0], [0.0, 0.0]]


def test_rows_become_unit_length():
    matrix = np.array([[3.0, 4.0], [1.0, 0.0]])

    norms = np.linalg.norm(l2_normalize(matrix), axis=1)

    assert np.allclose(norms, 1.0)


def test_zero_rows_are_left_alone():
    matrix = np.array([[0.0, 0.0], [3.0, 4.0]])

    result = l2_normalize(matrix)

    assert result[0].tolist() == [0.0, 0.0]
    assert not np.isnan(result).any()


def test_an_all_zero_matrix_produces_no_nans():
    result = l2_normalize(np.zeros((3, 5)))

    assert not np.isnan(result).any()
    assert result.tolist() == np.zeros((3, 5)).tolist()


def test_shape_is_preserved():
    assert l2_normalize(np.ones((7, 13))).shape == (7, 13)


def test_the_input_is_not_modified():
    matrix = np.array([[3.0, 4.0]])
    before = matrix.copy()

    l2_normalize(matrix)

    assert matrix.tolist() == before.tolist()


def test_float32_stays_float32():
    """A dtype change would break a FAISS index built on the original."""
    assert l2_normalize(np.array([[3.0, 4.0]], dtype=np.float32)).dtype == np.float32


def test_float64_stays_float64():
    assert l2_normalize(np.array([[3.0, 4.0]], dtype=np.float64)).dtype == np.float64


def test_normalizing_is_idempotent():
    matrix = np.array([[3.0, 4.0], [1.0, 2.0]])

    once = l2_normalize(matrix)

    assert np.allclose(l2_normalize(once), once)


def test_dot_product_of_normalized_rows_is_the_cosine():
    matrix = np.array([[1.0, 0.0], [1.0, 1.0]])

    normalized = l2_normalize(matrix)
    cosine = normalized[0] @ normalized[1]

    assert np.isclose(cosine, 1 / np.sqrt(2))


def test_direction_is_preserved():
    matrix = np.array([[3.0, 4.0]])

    scaled = l2_normalize(matrix)[0] * 5.0

    assert np.allclose(scaled, [3.0, 4.0])


def test_negative_values_are_handled():
    result = l2_normalize(np.array([[-3.0, -4.0]]))

    assert np.allclose(result, [[-0.6, -0.8]])


def test_a_single_row_matrix():
    assert np.allclose(l2_normalize(np.array([[3.0, 4.0]])), [[0.6, 0.8]])


# --- save_corpus ---


def test_round_trips_the_documented_example(tmp_path):
    embeddings = np.array([[1.0, 2.0]], dtype=np.float32)
    chunks = [{"text": "hi", "id": 0}]

    result = save_corpus(embeddings, chunks, str(tmp_path))

    assert result["embeddings"].tolist() == [[1.0, 2.0]]
    assert result["chunks"] == [{"text": "hi", "id": 0}]


def test_creates_the_directory_if_missing(tmp_path):
    target = os.path.join(str(tmp_path), "nested", "corpus")

    save_corpus(np.zeros((1, 2), dtype=np.float32), [{"text": "a"}], target)

    assert os.path.isdir(target)


def test_writes_both_files_under_the_expected_names(tmp_path):
    save_corpus(np.zeros((1, 2), dtype=np.float32), [{"text": "a"}], str(tmp_path))

    assert sorted(os.listdir(str(tmp_path))) == sorted([EMBEDDINGS_FILE, CHUNKS_FILE])


def test_dtype_survives_the_round_trip(tmp_path):
    """float32 -> float64 would break a FAISS index built on the original."""
    embeddings = np.zeros((2, 3), dtype=np.float32)

    result = save_corpus(embeddings, [{"t": 1}, {"t": 2}], str(tmp_path))

    assert result["embeddings"].dtype == np.float32


def test_shape_survives_the_round_trip(tmp_path):
    result = save_corpus(np.zeros((5, 7), dtype=np.float32), [], str(tmp_path))

    assert result["embeddings"].shape == (5, 7)


def test_values_survive_the_round_trip(tmp_path):
    embeddings = np.random.RandomState(0).rand(4, 6).astype(np.float32)

    result = save_corpus(embeddings, [], str(tmp_path))

    assert np.array_equal(result["embeddings"], embeddings)


def test_row_order_still_matches_the_chunks(tmp_path):
    embeddings = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    chunks = [{"position": 0}, {"position": 1}, {"position": 2}]

    result = save_corpus(embeddings, chunks, str(tmp_path))

    for row, chunk in zip(result["embeddings"], result["chunks"]):
        assert row[0] == chunk["position"]


def test_the_files_are_readable_by_a_fresh_process(tmp_path):
    """What the round trip is actually asserting: no in-memory state involved."""
    embeddings = np.array([[1.5, 2.5]], dtype=np.float32)
    save_corpus(embeddings, [{"text": "hi"}], str(tmp_path))

    reloaded = np.load(os.path.join(str(tmp_path), EMBEDDINGS_FILE))
    with open(os.path.join(str(tmp_path), CHUNKS_FILE), encoding="utf-8") as file:
        chunks = json.load(file)

    assert reloaded.tolist() == [[1.5, 2.5]]
    assert chunks == [{"text": "hi"}]


def test_overwrites_an_existing_corpus(tmp_path):
    save_corpus(np.zeros((1, 2), dtype=np.float32), [{"v": 1}], str(tmp_path))

    result = save_corpus(np.ones((1, 2), dtype=np.float32), [{"v": 2}], str(tmp_path))

    assert result["embeddings"].tolist() == [[1.0, 1.0]]
    assert result["chunks"] == [{"v": 2}]


def test_an_empty_corpus_round_trips(tmp_path):
    result = save_corpus(np.zeros((0, 4), dtype=np.float32), [], str(tmp_path))

    assert result["embeddings"].shape == (0, 4)
    assert result["chunks"] == []


def test_unicode_in_chunks_survives(tmp_path):
    chunks = [{"text": "café — 東京"}]

    result = save_corpus(np.zeros((1, 2), dtype=np.float32), chunks, str(tmp_path))

    assert result["chunks"] == chunks


def test_numpy_scalars_in_chunks_are_not_serialisable(tmp_path):
    """Documented sharp edge: JSON takes Python types, not numpy ones."""
    chunks = [{"position": np.int64(0)}]

    with pytest.raises(TypeError):
        save_corpus(np.zeros((1, 2), dtype=np.float32), chunks, str(tmp_path))


def test_composes_with_the_chunking_stage(tmp_path, model):
    from rag_pipeline.chunking import attach_chunk_metadata, chunk_fixed_size

    records = attach_chunk_metadata(chunk_fixed_size("abcdefgh", 3), "doc1")
    embeddings = l2_normalize(embed_chunks(model, records))

    result = save_corpus(embeddings, records, str(tmp_path))

    assert result["embeddings"].shape == (3, 4)
    assert [c["chunk_id"] for c in result["chunks"]] == [
        "doc1::0",
        "doc1::1",
        "doc1::2",
    ]
