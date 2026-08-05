import numpy as np

from rag_pipeline.generation import REFUSAL
from rag_pipeline.robustness import (
    cache_query_embedding,
    deduplicate_chunks,
    handle_no_context,
    rewrite_followup,
    update_chat_memory,
)


class CountingModel:
    def __init__(self, vector=(1.0, 2.0)):
        self.vector = list(vector)
        self.calls = 0

    def encode(self, text, batch_size=32):
        self.calls += 1
        return np.array([self.vector], dtype=np.float32)


# --- handle_no_context ---


def test_abstains_below_the_threshold():
    assert handle_no_context([("a", 0.1), ("b", 0.15)], threshold=0.2) == {
        "abstain": True,
        "message": REFUSAL,
    }


def test_answers_above_the_threshold():
    assert handle_no_context([("a", 0.5)], threshold=0.2) == {
        "abstain": False,
        "message": "",
    }


def test_the_comparison_is_strict():
    """A score exactly at the threshold does not clear the bar."""
    assert handle_no_context([("a", 0.2)], threshold=0.2)["abstain"] is True


def test_an_empty_retrieval_abstains():
    """max([]) would raise; the empty case has to be handled."""
    assert handle_no_context([], threshold=0.2)["abstain"] is True


def test_reads_scores_from_dicts():
    assert handle_no_context([{"score": 0.5}], threshold=0.2)["abstain"] is False


def test_reads_scores_from_tuples_and_dicts_the_same_way():
    assert (
        handle_no_context([("a", 0.5)], 0.2)
        == handle_no_context([{"score": 0.5}], 0.2)
    )


def test_only_the_best_score_matters():
    assert handle_no_context([("a", 0.0), ("b", 0.9)], threshold=0.5)["abstain"] is False


def test_the_refusal_matches_the_canonical_phrase():
    """Downstream abstention checks match on this exact string."""
    assert handle_no_context([], 0.2)["message"] == "I do not know"


def test_a_negative_threshold_lets_weak_matches_through():
    assert handle_no_context([("a", 0.0)], threshold=-0.1)["abstain"] is False


# --- deduplicate_chunks ---


def test_drops_a_near_duplicate():
    chunks = [{"chunk_id": 0}, {"chunk_id": 1}, {"chunk_id": 2}]
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    kept, kept_embeddings = deduplicate_chunks(chunks, embeddings, 0.95)

    assert [c["chunk_id"] for c in kept] == [0, 2]
    assert kept_embeddings.tolist() == [[1.0, 0.0], [0.0, 1.0]]


def test_keeps_the_first_occurrence():
    chunks = [{"chunk_id": "first"}, {"chunk_id": "second"}]
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0]])

    kept, _ = deduplicate_chunks(chunks, embeddings, 0.95)

    assert [c["chunk_id"] for c in kept] == ["first"]


def test_the_comparison_is_strict():
    """At exactly the threshold the chunk is kept, not dropped."""
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0]])

    kept, _ = deduplicate_chunks([{"i": 0}, {"i": 1}], embeddings, 1.0)

    assert len(kept) == 2


def test_a_chunk_is_never_a_duplicate_of_itself():
    kept, _ = deduplicate_chunks([{"i": 0}], np.array([[1.0, 0.0]]), 0.5)

    assert len(kept) == 1


def test_nothing_is_dropped_below_the_threshold():
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])

    kept, _ = deduplicate_chunks([{"i": 0}, {"i": 1}], embeddings, 0.5)

    assert len(kept) == 2


def test_chunks_and_rows_stay_aligned():
    chunks = [{"i": 0}, {"i": 1}, {"i": 2}, {"i": 3}]
    embeddings = np.array([[1.0, 0.0], [0.99, 0.14], [0.0, 1.0], [1.0, 0.0]])

    kept, kept_embeddings = deduplicate_chunks(chunks, embeddings, 0.95)

    assert len(kept) == len(kept_embeddings)


def test_order_is_preserved():
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]])

    kept, _ = deduplicate_chunks([{"i": 0}, {"i": 1}, {"i": 2}], embeddings, 0.9)

    assert [c["i"] for c in kept] == [0, 1, 2]


def test_an_empty_corpus():
    kept, kept_embeddings = deduplicate_chunks([], np.zeros((0, 2)), 0.9)

    assert kept == [] and kept_embeddings.shape == (0, 2)


def test_all_identical_collapses_to_one():
    embeddings = np.ones((5, 2)) / np.sqrt(2)

    kept, _ = deduplicate_chunks([{"i": n} for n in range(5)], embeddings, 0.95)

    assert len(kept) == 1


# --- cache_query_embedding ---


def test_computes_and_stores_on_a_miss():
    cache = {}
    model = CountingModel()

    vector = cache_query_embedding("hi", model, cache)

    assert vector.shape == (2,)
    assert "hi" in cache
    assert model.calls == 1


def test_does_not_recompute_on_a_hit():
    cache = {}
    model = CountingModel()

    cache_query_embedding("hi", model, cache)
    cache_query_embedding("hi", model, cache)

    assert model.calls == 1


def test_a_hit_returns_the_stored_vector():
    cache = {}
    model = CountingModel()

    first = cache_query_embedding("hi", model, cache)
    second = cache_query_embedding("hi", model, cache)

    assert first is second


def test_the_cache_is_mutated_in_place():
    """Not rebound -- the caller's dict must see the new entry."""
    cache = {}

    cache_query_embedding("hi", CountingModel(), cache)

    assert len(cache) == 1


def test_different_queries_get_separate_entries():
    cache = {}
    model = CountingModel()

    cache_query_embedding("a", model, cache)
    cache_query_embedding("b", model, cache)

    assert set(cache) == {"a", "b"} and model.calls == 2


def test_keys_are_used_as_given():
    """No normalisation: 'Hi' and 'hi' are different entries."""
    cache = {}
    model = CountingModel()

    cache_query_embedding("Hi", model, cache)
    cache_query_embedding("hi", model, cache)

    assert model.calls == 2


def test_a_prepopulated_cache_is_honoured():
    stored = np.array([9.0, 9.0], dtype=np.float32)
    model = CountingModel()

    result = cache_query_embedding("hi", model, {"hi": stored})

    assert result is stored and model.calls == 0


# --- update_chat_memory ---


def test_appends_both_turns():
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]

    updated = update_chat_memory(history, "what is RAG?", "retrieval augmented generation")

    assert updated == history + [
        {"role": "user", "content": "what is RAG?"},
        {"role": "assistant", "content": "retrieval augmented generation"},
    ]


def test_the_user_turn_comes_first():
    updated = update_chat_memory([], "q", "a")

    assert [turn["role"] for turn in updated] == ["user", "assistant"]


def test_the_original_history_is_not_mutated():
    history = [{"role": "user", "content": "hi"}]

    update_chat_memory(history, "q", "a")

    assert len(history) == 1


def test_returns_a_new_list():
    history = []

    assert update_chat_memory(history, "q", "a") is not history


def test_starting_from_an_empty_history():
    assert len(update_chat_memory([], "q", "a")) == 2


def test_turns_accumulate_across_calls():
    history = update_chat_memory([], "q1", "a1")
    history = update_chat_memory(history, "q2", "a2")

    assert len(history) == 4
    assert history[2]["content"] == "q2"


def test_each_turn_has_role_and_content():
    for turn in update_chat_memory([], "q", "a"):
        assert set(turn) == {"role", "content"}


# --- rewrite_followup ---


HISTORY = [
    {"role": "user", "content": "Tell me about Mars."},
    {"role": "assistant", "content": "Mars is red."},
]


def test_no_history_returns_the_followup():
    assert rewrite_followup("How big is it?", []) == "How big is it?"


def test_prepends_the_previous_user_turn():
    assert rewrite_followup("How big is it?", HISTORY) == (
        "Tell me about Mars. How big is it?"
    )


def test_the_assistant_turn_is_not_used():
    """Grabbing the last entry regardless of role would pull in 'Mars is red.'"""
    assert "Mars is red" not in rewrite_followup("How big is it?", HISTORY)


def test_uses_the_last_user_turn_not_the_first():
    history = [
        {"role": "user", "content": "First question."},
        {"role": "assistant", "content": "First answer."},
        {"role": "user", "content": "Second question."},
        {"role": "assistant", "content": "Second answer."},
    ]

    assert rewrite_followup("And?", history).startswith("Second question.")


def test_whitespace_is_normalised():
    assert rewrite_followup("  How   big  ", HISTORY) == (
        "Tell me about Mars. How big"
    )


def test_a_history_with_only_assistant_turns():
    history = [{"role": "assistant", "content": "hello"}]

    assert rewrite_followup("How big is it?", history) == "How big is it?"


def test_case_is_preserved():
    """normalize_text collapses whitespace; it does not lowercase."""
    assert rewrite_followup("How BIG?", []) == "How BIG?"


def test_turns_without_a_role_are_ignored():
    assert rewrite_followup("q", [{"content": "no role"}]) == "q"
