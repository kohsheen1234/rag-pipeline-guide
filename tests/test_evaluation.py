import pytest

from rag_pipeline.evaluation import (
    build_eval_set,
    faithfulness_score,
    hit_rate_at_k,
    mean_reciprocal_rank,
    recall_at_k,
    relevance_score,
)


# --- build_eval_set ---


def test_has_at_least_three_entries():
    assert len(build_eval_set()) >= 3


def test_entries_have_the_contract_keys_in_order():
    for entry in build_eval_set():
        assert list(entry) == ["question", "answer", "relevant_ids"]


def test_first_entry_matches_the_documented_example():
    first = build_eval_set()[0]

    assert first["question"] == "What is RAG?"
    assert first["relevant_ids"] == ["c1", "c2"]


def test_no_entry_has_empty_relevant_ids():
    """Empty gold ids silently zero out recall and MRR."""
    assert all(entry["relevant_ids"] for entry in build_eval_set())


def test_relevant_ids_are_lists():
    assert all(isinstance(e["relevant_ids"], list) for e in build_eval_set())


def test_questions_are_distinct():
    questions = [entry["question"] for entry in build_eval_set()]

    assert len(set(questions)) == len(questions)


def test_returns_a_fresh_list_each_call():
    build_eval_set()[0]["question"] = "mutated"

    assert build_eval_set()[0]["question"] == "What is RAG?"


# --- hit_rate_at_k ---


def test_hit_rate_documented_example():
    assert hit_rate_at_k([[1, 2, 3], [4, 5, 6]], [[3], [9]], k=3) == 0.5


def test_hit_rate_is_binary_per_query():
    """Two hits in one query still count once."""
    assert hit_rate_at_k([["a", "b"]], [["a", "b"]], k=2) == 1.0


def test_hit_rate_respects_k():
    assert hit_rate_at_k([["x", "y", "a"]], [["a"]], k=2) == 0.0
    assert hit_rate_at_k([["x", "y", "a"]], [["a"]], k=3) == 1.0


def test_hit_rate_of_zero():
    assert hit_rate_at_k([["a"], ["b"]], [["z"], ["z"]], k=1) == 0.0


def test_hit_rate_of_one():
    assert hit_rate_at_k([["a"], ["b"]], [["a"], ["b"]], k=1) == 1.0


def test_hit_rate_empty_input():
    assert hit_rate_at_k([], [], k=3) == 0.0


def test_hit_rate_is_a_float_in_range():
    score = hit_rate_at_k([["a"], ["z"], ["a"]], [["a"], ["a"], ["a"]], k=1)

    assert isinstance(score, float) and 0.0 <= score <= 1.0


# --- recall_at_k ---


def test_recall_documented_example():
    retrieved = [["a", "b", "c"], ["x", "y", "z"]]
    relevant = [["a", "d"], ["y"]]

    assert recall_at_k(retrieved, relevant, k=2) == 0.75


def test_recall_divides_by_the_gold_count_not_k():
    """Dividing by k would give precision: 1/3, not 1/1."""
    assert recall_at_k([["a", "x", "y"]], [["a"]], k=3) == 1.0


def test_recall_is_capped_by_k():
    assert recall_at_k([["a", "b", "c"]], [["a", "b", "c"]], k=1) == pytest.approx(1 / 3)


def test_recall_with_no_gold_ids_scores_zero():
    """Scored as 0 rather than skipped, so the denominator stays the query count."""
    assert recall_at_k([["a"], ["b"]], [["a"], []], k=1) == 0.5


def test_recall_of_zero():
    assert recall_at_k([["a"]], [["z"]], k=1) == 0.0


def test_recall_empty_input():
    assert recall_at_k([], [], k=3) == 0.0


def test_recall_ignores_duplicates_in_the_retrieved_list():
    assert recall_at_k([["a", "a"]], [["a"]], k=2) == 1.0


# --- mean_reciprocal_rank ---


def test_mrr_documented_example():
    retrieved = [["a", "b", "c"], ["x", "y", "z"]]
    relevant = [["b"], ["z"]]

    assert round(mean_reciprocal_rank(retrieved, relevant), 4) == 0.4167


def test_mrr_ranks_are_one_based():
    """A 0-based rank would divide by zero on a first-position hit."""
    assert mean_reciprocal_rank([["a"]], [["a"]]) == 1.0


def test_mrr_only_the_first_hit_counts():
    with_one = mean_reciprocal_rank([["a", "b"]], [["a"]])
    with_two = mean_reciprocal_rank([["a", "b"]], [["a", "b"]])

    assert with_one == with_two == 1.0


def test_mrr_rewards_a_higher_position():
    high = mean_reciprocal_rank([["a", "x", "y"]], [["a"]])
    low = mean_reciprocal_rank([["x", "y", "a"]], [["a"]])

    assert high > low


def test_mrr_no_hit_contributes_zero():
    assert mean_reciprocal_rank([["x"], ["a"]], [["a"], ["a"]]) == 0.5


def test_mrr_empty_input():
    assert mean_reciprocal_rank([], []) == 0.0


def test_mrr_stays_in_range():
    score = mean_reciprocal_rank([["a", "b"], ["b", "a"]], [["b"], ["b"]])

    assert 0.0 <= score <= 1.0


# --- faithfulness_score ---


CONTEXT = [{"text": "the cat sat on the mat"}]


def test_fully_supported_answer_scores_one():
    assert faithfulness_score("the cat sat", CONTEXT) == 1.0


def test_partially_supported_answer():
    assert round(faithfulness_score("the dog sat", CONTEXT), 4) == 0.6667


def test_unsupported_answer_scores_zero():
    assert faithfulness_score("elephants fly", CONTEXT) == 0.0


def test_empty_answer_scores_zero():
    assert faithfulness_score("", CONTEXT) == 0.0


def test_is_case_insensitive():
    assert faithfulness_score("The CAT", CONTEXT) == 1.0


def test_repetition_counts_on_the_answer_side():
    """A set on the answer side would forgive repeated unsupported tokens."""
    assert faithfulness_score("cat zzz zzz zzz", CONTEXT) == 0.25


def test_context_is_concatenated_across_chunks():
    context = [{"text": "the cat"}, {"text": "sat down"}]

    assert faithfulness_score("cat sat", context) == 1.0


def test_no_context_makes_everything_unsupported():
    assert faithfulness_score("anything", []) == 0.0


def test_stays_in_range():
    assert 0.0 <= faithfulness_score("the dog sat on a hill", CONTEXT) <= 1.0


# --- relevance_score ---


def test_relevance_documented_example():
    score = relevance_score(
        "Paris is the capital of France", "What is the capital of France?"
    )

    assert score == 0.5


def test_identical_strings_score_one():
    assert relevance_score("the capital", "the capital") == 1.0


def test_disjoint_strings_score_zero():
    assert relevance_score("apples", "zebras") == 0.0


def test_both_empty_scores_zero():
    assert relevance_score("", "") == 0.0


def test_one_empty_scores_zero():
    assert relevance_score("", "anything") == 0.0


def test_is_symmetric():
    a, q = "one two three", "two three four"

    assert relevance_score(a, q) == relevance_score(q, a)


def test_repetition_does_not_change_the_score():
    """Set-based, so 'cat cat cat' and 'cat' are the same."""
    assert relevance_score("cat cat cat", "cat dog") == relevance_score("cat", "cat dog")


def test_attached_punctuation_does_not_match():
    """normalize_text collapses whitespace; it does not strip punctuation."""
    assert relevance_score("france", "france?") == 0.0


def test_stays_in_range():
    assert 0.0 <= relevance_score("a b c", "b c d") <= 1.0
