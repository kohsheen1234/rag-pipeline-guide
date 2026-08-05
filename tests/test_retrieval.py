import os

import numpy as np
import pytest

from rag_pipeline.embeddings import l2_normalize
from rag_pipeline.retrieval import (
    build_faiss_index,
    compare_faiss_to_numpy,
    cosine_similarity_search,
    faiss_search,
    retrieve,
    save_faiss_index,
    top_k_chunks,
    top_k_indices,
)


class FakeIndex:
    """IndexFlatIP semantics: exact inner-product search over stored rows.

    Real faiss is a large optional dependency, so the duck-typed functions are
    tested against this. The ones that construct or serialise a real index are
    skipped unless faiss is installed.
    """

    def __init__(self, matrix):
        self.matrix = np.asarray(matrix, dtype=np.float32)
        self.ntotal, self.d = self.matrix.shape

    def search(self, queries, k):
        similarities = np.asarray(queries, dtype=np.float32) @ self.matrix.T
        indices = np.argsort(-similarities, axis=1, kind="stable")[:, :k]
        scores = np.take_along_axis(similarities, indices, axis=1)
        return scores.astype(np.float32), indices.astype(np.int64)


class FakeModel:
    """Maps a handful of known queries to fixed unit vectors."""

    VECTORS = {"q1": [1.0, 0.0], "q2": [0.0, 1.0]}

    def encode(self, text, batch_size=32):
        return np.array([self.VECTORS[text]], dtype=np.float32)


# --- cosine_similarity_search ---


def test_documented_example():
    query = np.array([1.0, 0.0])
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    scores = cosine_similarity_search(query, matrix)

    assert np.round(scores, 4).tolist() == [1.0, 0.0, 0.7071]


def test_returns_one_score_per_row():
    assert cosine_similarity_search(np.ones(3), np.ones((7, 3))).shape == (7,)


def test_identical_direction_scores_one():
    score = cosine_similarity_search(np.array([2.0, 0.0]), np.array([[5.0, 0.0]]))

    assert np.isclose(score[0], 1.0)


def test_opposite_direction_scores_minus_one():
    score = cosine_similarity_search(np.array([1.0, 0.0]), np.array([[-1.0, 0.0]]))

    assert np.isclose(score[0], -1.0)


def test_magnitude_is_ignored():
    """Cosine measures direction only -- a 1000x longer row scores the same."""
    query = np.array([1.0, 1.0])
    matrix = np.array([[1.0, 0.0], [1000.0, 0.0]])

    scores = cosine_similarity_search(query, matrix)

    assert np.isclose(scores[0], scores[1])


def test_normalizes_along_the_row_axis():
    """A non-square matrix would break if norms were taken along axis=0."""
    query = np.array([1.0, 0.0, 0.0])
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])

    assert np.allclose(cosine_similarity_search(query, matrix), [1.0, 0.0])


def test_a_square_matrix_would_hide_an_axis_bug():
    """Same check where n == d, which is where the wrong axis stays silent."""
    query = np.array([1.0, 0.0])
    matrix = np.array([[3.0, 0.0], [0.0, 5.0]])

    assert np.allclose(cosine_similarity_search(query, matrix), [1.0, 0.0])


def test_zero_rows_score_zero_without_nans():
    scores = cosine_similarity_search(np.array([1.0, 0.0]), np.zeros((2, 2)))

    assert not np.isnan(scores).any()
    assert scores.tolist() == [0.0, 0.0]


def test_a_zero_query_gives_no_nans():
    scores = cosine_similarity_search(np.zeros(2), np.array([[1.0, 0.0]]))

    assert not np.isnan(scores).any()


def test_scores_stay_within_the_cosine_range():
    rng = np.random.RandomState(0)
    scores = cosine_similarity_search(rng.rand(8) - 0.5, rng.rand(20, 8) - 0.5)

    assert scores.min() >= -1.0000001 and scores.max() <= 1.0000001


def test_matches_a_dot_product_on_normalized_input():
    rng = np.random.RandomState(1)
    query = rng.rand(5)
    matrix = rng.rand(6, 5)

    expected = l2_normalize(matrix) @ (query / np.linalg.norm(query))

    assert np.allclose(cosine_similarity_search(query, matrix), expected)


# --- top_k_indices ---


def test_returns_the_largest_scores_descending():
    scores = np.array([0.1, 0.9, 0.4, 0.7, 0.2])

    assert top_k_indices(scores, 3).tolist() == [1, 3, 2]


def test_k_larger_than_n_returns_everything():
    assert top_k_indices(np.array([0.5, 0.5]), 5).tolist() == [0, 1]


def test_descending_not_ascending():
    """The classic bug: the obvious sort is ascending."""
    scores = np.array([0.1, 0.9])

    assert top_k_indices(scores, 1).tolist() == [1]


def test_scores_are_in_descending_order():
    rng = np.random.RandomState(2)
    scores = rng.rand(50)

    picked = scores[top_k_indices(scores, 10)]

    assert (np.diff(picked) <= 0).all()


def test_k_of_one_returns_the_argmax():
    scores = np.array([0.3, 0.1, 0.8, 0.2])

    assert top_k_indices(scores, 1).tolist() == [int(np.argmax(scores))]


def test_ties_are_broken_by_index():
    """Reproducible across runs, which matters when comparing backends."""
    assert top_k_indices(np.array([0.5, 0.5, 0.5]), 2).tolist() == [0, 1]


def test_handles_negative_scores():
    scores = np.array([-0.9, -0.1, -0.5])

    assert top_k_indices(scores, 2).tolist() == [1, 2]


def test_k_of_zero_returns_nothing():
    assert top_k_indices(np.array([0.1, 0.9]), 0).tolist() == []


def test_a_single_score():
    assert top_k_indices(np.array([0.4]), 3).tolist() == [0]


# --- top_k_chunks ---


def test_pairs_chunks_with_their_scores():
    scores = np.array([0.1, 0.9, 0.5, 0.7])
    chunks = [{"id": 0}, {"id": 1}, {"id": 2}, {"id": 3}]

    assert top_k_chunks(scores, chunks, 2) == [({"id": 1}, 0.9), ({"id": 3}, 0.7)]


def test_scores_are_plain_python_floats():
    """Not numpy scalars -- json.dump rejects those."""
    _, score = top_k_chunks(np.array([0.5]), [{"id": 0}], 1)[0]

    assert type(score) is float


def test_each_chunk_keeps_its_own_score():
    scores = np.array([0.2, 0.8, 0.5])
    chunks = ["a", "b", "c"]

    assert top_k_chunks(scores, chunks, 3) == [("b", 0.8), ("c", 0.5), ("a", 0.2)]


def test_truncates_to_k():
    assert len(top_k_chunks(np.arange(10.0), list(range(10)), 4)) == 4


def test_truncates_to_the_chunk_count():
    assert len(top_k_chunks(np.arange(3.0), [0, 1, 2], 99)) == 3


def test_results_are_in_descending_score_order():
    scores = np.array([0.1, 0.9, 0.5])

    returned = [score for _, score in top_k_chunks(scores, ["a", "b", "c"], 3)]

    assert returned == sorted(returned, reverse=True)


def test_works_with_real_chunk_records():
    from rag_pipeline.chunking import attach_chunk_metadata

    chunks = attach_chunk_metadata(["alpha", "beta"], "doc1")

    top = top_k_chunks(np.array([0.2, 0.9]), chunks, 1)

    assert top[0][0]["chunk_id"] == "doc1::1"


def test_k_of_zero_returns_nothing():
    assert top_k_chunks(np.array([0.5]), [{"id": 0}], 0) == []


# --- retrieve ---


@pytest.fixture
def corpus():
    matrix = np.array([[0.6, 0.8], [0.8, 0.6]], dtype=np.float32)
    chunks = [{"chunk_id": "c0"}, {"chunk_id": "c1"}]
    return matrix, chunks


def test_end_to_end_retrieval(corpus):
    matrix, chunks = corpus

    hits = retrieve("q1", FakeModel(), matrix, chunks, 2)

    assert [(c["chunk_id"], round(s, 4)) for c, s in hits] == [
        ("c1", 0.8),
        ("c0", 0.6),
    ]


def test_returns_chunk_dicts_not_indices(corpus):
    """Downstream code reads chunk['text']; indices would break it."""
    matrix, chunks = corpus

    chunk, _ = retrieve("q1", FakeModel(), matrix, chunks, 1)[0]

    assert isinstance(chunk, dict)


def test_respects_k(corpus):
    matrix, chunks = corpus

    assert len(retrieve("q1", FakeModel(), matrix, chunks, 1)) == 1


def test_a_different_query_reorders_the_results(corpus):
    matrix, chunks = corpus

    first = retrieve("q1", FakeModel(), matrix, chunks, 2)[0][0]["chunk_id"]
    second = retrieve("q2", FakeModel(), matrix, chunks, 2)[0][0]["chunk_id"]

    assert first != second


def test_scores_are_json_serialisable(corpus):
    import json

    matrix, chunks = corpus
    hits = retrieve("q1", FakeModel(), matrix, chunks, 2)

    assert json.dumps([{"chunk": c, "score": s} for c, s in hits])


# --- faiss_search ---


def test_returns_flat_one_dimensional_arrays():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]], dtype=np.float32)

    scores, indices = faiss_search(FakeIndex(matrix), np.array([1.0, 0.0], np.float32), 2)

    assert scores.ndim == 1 and indices.ndim == 1


def test_documented_search_example():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]], dtype=np.float32)

    scores, indices = faiss_search(FakeIndex(matrix), np.array([1.0, 0.0], np.float32), 2)

    assert indices.tolist() == [0, 2]
    assert [round(float(s), 4) for s in scores] == [1.0, 0.7071]


def test_dtypes_are_float32_and_int64():
    matrix = np.eye(3, dtype=np.float32)

    scores, indices = faiss_search(FakeIndex(matrix), np.eye(3, dtype=np.float32)[0], 2)

    assert scores.dtype == np.float32
    assert indices.dtype == np.int64


def test_adds_the_batch_axis_for_the_caller():
    """A (d,) query would be rejected by index.search without the reshape."""
    matrix = np.eye(2, dtype=np.float32)
    query = np.array([1.0, 0.0], dtype=np.float32)

    assert query.ndim == 1
    assert faiss_search(FakeIndex(matrix), query, 1)[1].tolist() == [0]


def test_returns_k_results():
    matrix = np.eye(5, dtype=np.float32)

    scores, indices = faiss_search(FakeIndex(matrix), matrix[0], 3)

    assert len(scores) == 3 and len(indices) == 3


def test_scores_are_descending():
    matrix = l2_normalize(np.random.RandomState(3).rand(10, 4)).astype(np.float32)

    scores, _ = faiss_search(FakeIndex(matrix), matrix[0], 5)

    assert (np.diff(scores) <= 1e-6).all()


# --- compare_faiss_to_numpy ---


def test_backends_agree_on_an_identity_corpus():
    matrix = np.eye(4, dtype=np.float32)
    query = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    assert compare_faiss_to_numpy(query, matrix, FakeIndex(matrix), k=2)


def test_backends_agree_on_a_random_normalized_corpus():
    matrix = l2_normalize(np.random.RandomState(4).rand(30, 8)).astype(np.float32)
    query = matrix[7]

    assert compare_faiss_to_numpy(query, matrix, FakeIndex(matrix), k=5)


@pytest.mark.parametrize("k", [1, 3, 10, 30])
def test_backends_agree_across_k(k):
    matrix = l2_normalize(np.random.RandomState(5).rand(30, 6)).astype(np.float32)

    assert compare_faiss_to_numpy(matrix[0], matrix, FakeIndex(matrix), k=k)


def test_detects_a_corrupt_index():
    """The check is only useful if it can fail -- index over different vectors."""
    matrix = np.eye(4, dtype=np.float32)
    wrong = FakeIndex(np.eye(4, dtype=np.float32)[::-1])
    query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    assert not compare_faiss_to_numpy(query, matrix, wrong, k=1)


def test_ties_do_not_count_as_disagreement():
    """Equal scores may order differently; set comparison ignores that."""
    matrix = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    query = np.array([1.0, 0.0], dtype=np.float32)

    assert compare_faiss_to_numpy(query, matrix, FakeIndex(matrix), k=2)


# --- real faiss ---


@pytest.fixture
def faiss_module():
    return pytest.importorskip("faiss")


def test_builds_an_index(faiss_module):
    matrix = np.eye(3, dtype=np.float32)

    index = build_faiss_index(matrix)

    assert index.ntotal == 3
    assert index.d == 3


def test_real_faiss_agrees_with_numpy(faiss_module):
    matrix = l2_normalize(np.random.RandomState(6).rand(40, 16)).astype(np.float32)
    index = build_faiss_index(matrix)

    assert compare_faiss_to_numpy(matrix[3], matrix, index, k=5)


def test_index_round_trips_through_disk(faiss_module, tmp_path):
    matrix = l2_normalize(np.random.RandomState(7).rand(20, 8)).astype(np.float32)
    index = build_faiss_index(matrix)
    path = os.path.join(str(tmp_path), "index.bin")

    reloaded = save_faiss_index(index, path)

    assert reloaded.ntotal == index.ntotal
    assert reloaded.d == index.d
    assert os.path.exists(path)


def test_reloaded_index_returns_the_same_neighbours(faiss_module, tmp_path):
    matrix = l2_normalize(np.random.RandomState(8).rand(20, 8)).astype(np.float32)
    index = build_faiss_index(matrix)

    reloaded = save_faiss_index(index, os.path.join(str(tmp_path), "index.bin"))

    _, before = faiss_search(index, matrix[2], 5)
    _, after = faiss_search(reloaded, matrix[2], 5)
    assert before.tolist() == after.tolist()


def test_reloaded_index_is_a_different_object(faiss_module, tmp_path):
    """Returning the original would pass every shape check and prove nothing."""
    index = build_faiss_index(np.eye(3, dtype=np.float32))

    reloaded = save_faiss_index(index, os.path.join(str(tmp_path), "index.bin"))

    assert reloaded is not index
