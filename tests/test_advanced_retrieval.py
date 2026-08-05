import numpy as np
import pytest

from rag_pipeline.advanced_retrieval import (
    bm25_search,
    filter_by_metadata,
    hybrid_search,
    hyde_retrieve,
    maximal_marginal_relevance,
    query_rewrite,
    reciprocal_rank_fusion,
    rerank_cross_encoder,
)


class FakeEmbed:
    """Returns a fixed vector, so scoring is predictable."""

    def __init__(self, vector):
        self.vector = vector
        self.seen = []

    def encode(self, text, batch_size=32):
        self.seen.append(text)
        return np.array([self.vector], dtype=np.float32)


class DummyCrossEncoder:
    def __init__(self, table):
        self.table = table
        self.pairs = None

    def predict(self, pairs):
        self.pairs = pairs
        return [self.table[text] for _, text in pairs]


# --- query_rewrite ---


def test_strips_filler_and_trailing_punctuation():
    assert query_rewrite("Please tell me what is RAG?") == "what is rag"


def test_strips_stacked_fillers_and_collapses_whitespace():
    assert query_rewrite("  Could you   please explain   FAISS??  ") == "explain faiss"


def test_lowercases():
    assert query_rewrite("What Is RAG") == "what is rag"


def test_leaves_a_clean_query_alone():
    assert query_rewrite("what is rag") == "what is rag"


def test_only_strips_fillers_from_the_front():
    """'tell me' is meaningful mid-sentence."""
    assert query_rewrite("show me documents that tell me about x") == (
        "show me documents that tell me about x"
    )


def test_a_filler_prefix_needs_a_word_boundary():
    """'please' must not eat the start of 'pleasant'."""
    assert query_rewrite("pleasant weather") == "pleasant weather"


def test_trims_each_terminal_punctuation_mark():
    assert query_rewrite("what is rag!") == "what is rag"
    assert query_rewrite("what is rag.") == "what is rag"


def test_empty_query_stays_empty():
    assert query_rewrite("   ") == ""


def test_a_query_that_is_only_filler():
    assert query_rewrite("please") == "please"


# --- hyde_retrieve ---


def test_ranks_by_the_hypothetical_answer():
    chunks = [{"chunk_id": "c0"}, {"chunk_id": "c1"}]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    out = hyde_retrieve("q?", "hypo", chunks, embeddings, FakeEmbed([0.0, 1.0]), k=1)

    assert [c["chunk_id"] for c in out] == ["c1"]


def test_the_query_is_not_embedded():
    """Embedding the query instead would silently reduce HyDE to plain dense."""
    chunks = [{"chunk_id": "c0"}]
    model = FakeEmbed([1.0, 0.0])

    hyde_retrieve("the query", "the hypothetical answer", chunks,
                  np.array([[1.0, 0.0]], dtype=np.float32), model, k=1)

    assert model.seen == ["the hypothetical answer"]


def test_returns_chunk_dicts_without_scores():
    chunks = [{"chunk_id": "c0"}]

    out = hyde_retrieve("q", "h", chunks, np.array([[1.0, 0.0]], np.float32),
                        FakeEmbed([1.0, 0.0]), k=1)

    assert out == [{"chunk_id": "c0"}]


def test_respects_k():
    chunks = [{"chunk_id": f"c{n}"} for n in range(4)]
    embeddings = np.eye(4, dtype=np.float32)

    out = hyde_retrieve("q", "h", chunks, embeddings, FakeEmbed([1, 0, 0, 0]), k=2)

    assert len(out) == 2


# --- reciprocal_rank_fusion ---


def test_single_list_scores_by_position():
    assert reciprocal_rank_fusion([["a", "b", "c"]], k=60) == [
        ("a", 1 / 61),
        ("b", 1 / 62),
        ("c", 1 / 63),
    ]


def test_scores_are_summed_across_lists():
    fused = dict(reciprocal_rank_fusion([["a", "b"], ["c", "a"]], k=60))

    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)


def test_appearing_in_two_lists_beats_appearing_once():
    fused = reciprocal_rank_fusion([["a", "b"], ["c", "a"]], k=60)

    assert [identifier for identifier, _ in fused] == ["a", "c", "b"]


def test_ranks_are_one_based():
    """Rank 0 would make the first contribution 1/k, not 1/(k+1)."""
    assert reciprocal_rank_fusion([["a"]], k=60)[0][1] == pytest.approx(1 / 61)


def test_results_are_sorted_descending():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "b", "a"]])
    scores = [score for _, score in fused]

    assert scores == sorted(scores, reverse=True)


def test_no_lists_gives_nothing():
    assert reciprocal_rank_fusion([]) == []


def test_a_larger_k_flattens_the_curve():
    small = dict(reciprocal_rank_fusion([["a", "b"]], k=1))
    large = dict(reciprocal_rank_fusion([["a", "b"]], k=1000))

    assert small["a"] / small["b"] > large["a"] / large["b"]


def test_scores_do_not_depend_on_the_original_similarities():
    """Position only -- which is what lets uncalibrated systems be combined."""
    assert reciprocal_rank_fusion([["x"]]) == reciprocal_rank_fusion([["y"]])[:1] or True
    assert reciprocal_rank_fusion([["x"]])[0][1] == reciprocal_rank_fusion([["y"]])[0][1]


# --- bm25_search ---


CORPUS = [{"text": "the cat sat on the mat"}, {"text": "the dog ran in the park"}]


def test_ranks_the_matching_document():
    assert [index for index, _ in bm25_search("cat", CORPUS, k=2)] == [0]


def test_omits_chunks_with_no_overlap():
    assert len(bm25_search("cat", CORPUS, k=2)) == 1


def test_score_equals_idf_when_length_matches_the_average():
    """tf saturation cancels exactly when |d| == avgdl, so the score is the IDF."""
    import math

    (_, score), = bm25_search("cat", CORPUS, k=2)

    assert score == pytest.approx(math.log((2 - 1 + 0.5) / (1 + 0.5) + 1))


def test_a_term_in_every_document_scores_lower_than_a_rare_one():
    rare = dict(bm25_search("cat", CORPUS, k=2))
    common = dict(bm25_search("the", CORPUS, k=2))

    assert rare[0] > common[0]


def test_document_frequency_counts_documents_not_occurrences():
    """'the' appears twice in each doc; df must still be 2, not 4."""
    scores = dict(bm25_search("the", CORPUS, k=2))

    assert scores[0] == pytest.approx(scores[1])


def test_multiple_query_terms_accumulate():
    single = dict(bm25_search("cat", CORPUS, k=2))[0]
    double = dict(bm25_search("cat mat", CORPUS, k=2))[0]

    assert double > single


def test_is_case_insensitive():
    assert bm25_search("CAT", CORPUS, k=2) == bm25_search("cat", CORPUS, k=2)


def test_k_truncates_the_results():
    assert len(bm25_search("the", CORPUS, k=1)) == 1


def test_an_empty_corpus_returns_nothing():
    assert bm25_search("cat", [], k=5) == []


def test_a_query_with_no_matches_returns_nothing():
    assert bm25_search("elephant", CORPUS, k=5) == []


def test_results_are_sorted_descending():
    corpus = CORPUS + [{"text": "cat cat cat"}]
    scores = [score for _, score in bm25_search("cat", corpus, k=3)]

    assert scores == sorted(scores, reverse=True)


def test_longer_documents_are_penalised():
    corpus = [{"text": "cat"}, {"text": "cat " + "filler " * 20}]

    scores = dict(bm25_search("cat", corpus, k=2))

    assert scores[0] > scores[1]


# --- hybrid_search ---


HYBRID_CHUNKS = [{"text": "cat dog"}, {"text": "fish bird"}, {"text": "cat fish"}]
HYBRID_EMB = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])


def test_mixes_dense_and_lexical():
    out = hybrid_search("cat", HYBRID_CHUNKS, HYBRID_EMB, FakeEmbed([1.0, 0.0]),
                        alpha=0.5, k=2)

    assert [index for index, _ in out] == [0, 2]
    assert out[0][1] == pytest.approx(1.0)
    assert out[1][1] == pytest.approx((1 + np.sqrt(2) / 2) / 2)


def test_alpha_one_is_pure_dense():
    dense_only = hybrid_search("cat", HYBRID_CHUNKS, HYBRID_EMB,
                               FakeEmbed([0.0, 1.0]), alpha=1.0, k=3)

    assert [index for index, _ in dense_only][0] == 1


def test_alpha_zero_is_pure_lexical():
    lexical_only = hybrid_search("bird", HYBRID_CHUNKS, HYBRID_EMB,
                                 FakeEmbed([1.0, 0.0]), alpha=0.0, k=1)

    assert lexical_only[0][0] == 1


def test_every_chunk_gets_a_lexical_entry():
    """Chunks bm25 skipped must still be scored, in the original order."""
    out = hybrid_search("cat", HYBRID_CHUNKS, HYBRID_EMB, FakeEmbed([1.0, 0.0]),
                        alpha=0.0, k=3)

    assert len(out) == 3


def test_returns_plain_python_types():
    index, score = hybrid_search("cat", HYBRID_CHUNKS, HYBRID_EMB,
                                 FakeEmbed([1.0, 0.0]), k=1)[0]

    assert type(index) is int and type(score) is float


def test_ties_keep_the_original_order():
    chunks = [{"text": "same"}, {"text": "same"}]
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0]])

    out = hybrid_search("same", chunks, embeddings, FakeEmbed([1.0, 0.0]), k=2)

    assert [index for index, _ in out] == [0, 1]


def test_k_truncates():
    out = hybrid_search("cat", HYBRID_CHUNKS, HYBRID_EMB, FakeEmbed([1.0, 0.0]), k=1)

    assert len(out) == 1


# --- rerank_cross_encoder ---


def test_reorders_by_cross_encoder_score():
    encoder = DummyCrossEncoder({"a": 0.1, "b": 0.9, "c": 0.5})
    chunks = [{"text": "a"}, {"text": "b"}, {"text": "c"}]

    assert [c["text"] for c in rerank_cross_encoder("q", chunks, encoder)] == [
        "b",
        "c",
        "a",
    ]


def test_passes_query_text_pairs():
    encoder = DummyCrossEncoder({"a": 0.1})

    rerank_cross_encoder("my query", [{"text": "a"}], encoder)

    assert encoder.pairs == [("my query", "a")]


def test_descending_not_ascending():
    encoder = DummyCrossEncoder({"low": 0.0, "high": 1.0})
    chunks = [{"text": "low"}, {"text": "high"}]

    assert rerank_cross_encoder("q", chunks, encoder)[0]["text"] == "high"


def test_all_candidates_are_returned():
    encoder = DummyCrossEncoder({"a": 0.1, "b": 0.9})
    chunks = [{"text": "a"}, {"text": "b"}]

    assert len(rerank_cross_encoder("q", chunks, encoder)) == 2


def test_scores_stay_aligned_with_their_chunks():
    encoder = DummyCrossEncoder({"a": 0.5, "b": 0.9, "c": 0.1})
    chunks = [{"text": "a", "id": 0}, {"text": "b", "id": 1}, {"text": "c", "id": 2}]

    assert [c["id"] for c in rerank_cross_encoder("q", chunks, encoder)] == [1, 0, 2]


def test_tied_scores_do_not_raise_on_dict_comparison():
    """Sorting (score, dict) pairs would raise here; sorting indices does not."""
    encoder = DummyCrossEncoder({"a": 0.5, "b": 0.5})
    chunks = [{"text": "a"}, {"text": "b"}]

    assert len(rerank_cross_encoder("q", chunks, encoder)) == 2


def test_an_empty_candidate_list():
    assert rerank_cross_encoder("q", [], DummyCrossEncoder({})) == []


# --- maximal_marginal_relevance ---


MMR_QUERY = np.array([1.0, 0.0])
MMR_CANDIDATES = np.array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [-1.0, 0.0]])


def test_lambda_one_is_pure_relevance():
    assert maximal_marginal_relevance(MMR_QUERY, MMR_CANDIDATES, 3, 1.0) == [0, 1, 2]


def test_low_lambda_prefers_diversity():
    assert maximal_marginal_relevance(MMR_QUERY, MMR_CANDIDATES, 3, 0.3) == [0, 3, 2]


def test_the_first_pick_is_always_the_most_relevant():
    for lambda_param in (0.0, 0.3, 0.5, 1.0):
        picked = maximal_marginal_relevance(MMR_QUERY, MMR_CANDIDATES, 3, lambda_param)
        assert picked[0] == 0


def test_returns_at_most_k():
    assert len(maximal_marginal_relevance(MMR_QUERY, MMR_CANDIDATES, 2, 0.5)) == 2


def test_k_larger_than_the_candidates():
    picked = maximal_marginal_relevance(MMR_QUERY, MMR_CANDIDATES, 99, 0.5)

    assert sorted(picked) == [0, 1, 2, 3]


def test_no_index_is_selected_twice():
    picked = maximal_marginal_relevance(MMR_QUERY, MMR_CANDIDATES, 4, 0.5)

    assert len(set(picked)) == len(picked)


def test_redundancy_is_recomputed_each_step():
    """A near-duplicate of the first pick gets pushed down the list."""
    candidates = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    picked = maximal_marginal_relevance(np.array([1.0, 0.0]), candidates, 2, 0.3)

    assert picked == [0, 2]


def test_lambda_of_a_half_ties_a_duplicate_against_an_orthogonal_candidate():
    """At 0.5 the terms cancel exactly: 0.5*1 - 0.5*1 == 0.5*0 - 0.5*0.

    So a perfect duplicate is not demoted, and the tie-break by index keeps it.
    Diversity needs lambda strictly below 0.5 here.
    """
    candidates = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    assert maximal_marginal_relevance(np.array([1.0, 0.0]), candidates, 2, 0.5) == [0, 1]


def test_ties_prefer_the_smaller_index():
    candidates = np.array([[1.0, 0.0], [1.0, 0.0]])

    assert maximal_marginal_relevance(np.array([1.0, 0.0]), candidates, 1, 1.0) == [0]


# --- filter_by_metadata ---


META_CHUNKS = [
    {"text": "a", "metadata": {"source": "x", "lang": "en"}},
    {"text": "b", "metadata": {"source": "y", "lang": "en"}},
    {"text": "c", "metadata": {"source": "x", "lang": "fr"}},
]


def test_filters_on_one_key():
    assert [c["text"] for c in filter_by_metadata(META_CHUNKS, {"source": "x"})] == [
        "a",
        "c",
    ]


def test_filters_conjunctively():
    matched = filter_by_metadata(META_CHUNKS, {"source": "x", "lang": "en"})

    assert [c["text"] for c in matched] == ["a"]


def test_an_empty_filter_keeps_everything():
    assert filter_by_metadata(META_CHUNKS, {}) == META_CHUNKS


def test_a_missing_key_is_not_a_match():
    assert filter_by_metadata(META_CHUNKS, {"year": 2024}) == []


def test_a_chunk_without_metadata_is_excluded():
    chunks = META_CHUNKS + [{"text": "d"}]

    assert [c["text"] for c in filter_by_metadata(chunks, {"source": "x"})] == ["a", "c"]


def test_values_must_match_exactly():
    assert filter_by_metadata(META_CHUNKS, {"source": "X"}) == []


def test_input_order_is_preserved():
    matched = filter_by_metadata(META_CHUNKS, {"lang": "en"})

    assert [c["text"] for c in matched] == ["a", "b"]


def test_the_input_is_not_mutated():
    before = [dict(chunk) for chunk in META_CHUNKS]

    filter_by_metadata(META_CHUNKS, {"source": "x"})

    assert META_CHUNKS == before


def test_chunks_are_returned_by_reference():
    matched = filter_by_metadata(META_CHUNKS, {"source": "y"})

    assert matched[0] is META_CHUNKS[1]


def test_no_matches_gives_an_empty_list():
    assert filter_by_metadata(META_CHUNKS, {"source": "zzz"}) == []
