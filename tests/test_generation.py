import numpy as np
import pytest
import torch

from rag_pipeline.generation import (
    REFUSAL,
    add_system_instruction,
    append_source_references,
    build_prompt_template,
    format_context,
    generate_answer,
    load_generator,
    rag_answer,
    track_source_chunk_ids,
    truncate_context,
)


class FakeTokenizer:
    """Enough of the tokenizer surface for generate_answer: call and decode."""

    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = 0

    def __call__(self, prompt, return_tensors=None):
        ids = [(ord(character) % 50) + 1 for character in prompt][:24]
        return {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
        }

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(65 + int(i) % 26) for i in ids)


class FakeGenerator:
    """Appends a deterministic run of new token ids after the prompt."""

    def __init__(self):
        self.last_call = None

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        self.last_call = kwargs
        new = torch.arange(kwargs["max_new_tokens"]).unsqueeze(0)
        return torch.cat([input_ids, new], dim=1)


class FakeEmbedModel:
    def __init__(self, vector):
        self.vector = vector

    def encode(self, text, batch_size=32):
        return np.array([self.vector], dtype=np.float32)


# --- build_prompt_template ---


def test_has_both_placeholders():
    template = build_prompt_template()

    assert "{context}" in template and "{question}" in template


def test_each_placeholder_appears_exactly_once():
    template = build_prompt_template()

    assert template.count("{context}") == 1
    assert template.count("{question}") == 1


def test_formats_without_error():
    filled = build_prompt_template().format(context="ctx", question="q")

    assert "ctx" in filled and "q" in filled


def test_formatted_prompt_starts_with_context():
    filled = build_prompt_template().format(
        context="Paris is the capital of France.",
        question="What is the capital of France?",
    )

    assert filled[:7] == "Context"


def test_no_stray_braces_that_would_break_format():
    """Any unescaped literal brace would raise here."""
    build_prompt_template().format(context="{}", question="{}")


def test_template_is_not_prematurely_interpolated():
    """An f-string at definition time would leave no placeholders behind."""
    assert "{" in build_prompt_template()


def test_template_names_the_refusal_phrase():
    assert REFUSAL in build_prompt_template()


# --- format_context ---


def test_renders_numbered_lines():
    retrieved = [
        ({"text": "Cats purr.", "source": "a.txt"}, 0.9),
        ({"text": "Dogs bark.", "source": "b.txt"}, 0.7),
    ]

    assert format_context(retrieved) == (
        "[1] Cats purr. (source=a.txt)\n[2] Dogs bark. (source=b.txt)"
    )


def test_indices_are_one_based():
    retrieved = [({"text": "t", "source": "s"}, 0.5)]

    assert format_context(retrieved).startswith("[1]")


def test_scores_are_not_rendered():
    """Useful for filtering, noise in the prompt."""
    retrieved = [({"text": "t", "source": "s"}, 0.98765)]

    assert "0.98765" not in format_context(retrieved)


def test_empty_retrieval_gives_an_empty_string():
    assert format_context([]) == ""


def test_one_line_per_chunk():
    retrieved = [({"text": f"t{n}", "source": "s"}, 0.1) for n in range(4)]

    assert len(format_context(retrieved).split("\n")) == 4


def test_extra_metadata_is_ignored():
    retrieved = [({"text": "t", "source": "s", "position": 3}, 0.5)]

    assert format_context(retrieved) == "[1] t (source=s)"


def test_composes_with_retrieval_output():
    from rag_pipeline.retrieval import top_k_chunks

    chunks = [{"text": "a", "source": "x"}, {"text": "b", "source": "y"}]
    retrieved = top_k_chunks(np.array([0.2, 0.9]), chunks, 2)

    assert format_context(retrieved) == "[1] b (source=y)\n[2] a (source=x)"


# --- truncate_context ---


def test_short_context_is_returned_unchanged():
    assert truncate_context("hello world", 50) == "hello world"


def test_cuts_at_the_last_whitespace_before_the_limit():
    assert truncate_context("the quick brown fox jumps", 15) == "the quick brown"


def test_never_exceeds_the_limit():
    text = "the quick brown fox jumps over the lazy dog"

    assert all(len(truncate_context(text, m)) <= m for m in range(1, len(text) + 5))


def test_falls_back_to_a_hard_cut_without_whitespace():
    assert truncate_context("abcdefghij", 5) == "abcde"


def test_cuts_at_the_last_not_the_first_whitespace():
    """A first-whitespace cut would return 'a'; the last gives the full budget."""
    assert truncate_context("a b c d e f", 7) == "a b c d"


def test_exact_length_is_unchanged():
    assert truncate_context("abcde", 5) == "abcde"


def test_cuts_at_newlines_too():
    """format_context produces newline-separated lines, not spaces."""
    context = "[1] aaa\n[2] bbb\n[3] ccc"

    assert truncate_context(context, 16) == "[1] aaa\n[2] bbb"


def test_empty_context_is_unchanged():
    assert truncate_context("", 10) == ""


def test_no_trailing_whitespace_is_left_behind():
    assert truncate_context("hello world again", 12) == "hello world"


# --- add_system_instruction ---


def test_matches_the_documented_output():
    prompt = (
        "Context: Paris is the capital of France.\n"
        "Question: What is the capital of France?"
    )

    assert add_system_instruction(prompt) == (
        "You are a helpful assistant. Answer the question using ONLY the "
        "provided context. If the answer is not in the context, say "
        "'I do not know'.\n\n" + prompt
    )


def test_separated_by_a_blank_line():
    assert "\n\n" in add_system_instruction("prompt")


def test_the_prompt_is_unchanged():
    prompt = "Context: x\nQuestion: y"

    assert add_system_instruction(prompt).endswith(prompt)


def test_names_the_canonical_refusal():
    """One spelling everywhere, so abstention checks can match on it."""
    assert REFUSAL in add_system_instruction("p")


def test_instruction_is_identical_across_calls():
    assert add_system_instruction("a")[: -len("a")] == (
        add_system_instruction("b")[: -len("b")]
    )


# --- generate_answer ---


@pytest.fixture
def generator():
    return FakeGenerator(), FakeTokenizer()


def test_returns_a_string(generator):
    model, tokenizer = generator

    assert isinstance(generate_answer(model, tokenizer, "Hello", 4), str)


def test_is_deterministic(generator):
    model, tokenizer = generator

    first = generate_answer(model, tokenizer, "Hello", 4)
    second = generate_answer(model, tokenizer, "Hello", 4)

    assert first == second


def test_only_the_continuation_is_returned(generator):
    """The prompt is sliced off in token space, not string-replaced."""
    model, tokenizer = generator

    answer = generate_answer(model, tokenizer, "Hello there", 4)

    assert len(answer) == 4


def test_respects_the_token_budget(generator):
    model, tokenizer = generator

    assert len(generate_answer(model, tokenizer, "Hello", 9)) == 9


def test_uses_greedy_decoding(generator):
    model, tokenizer = generator

    generate_answer(model, tokenizer, "Hello", 4)

    assert model.last_call["do_sample"] is False


def test_passes_the_pad_token_id(generator):
    model, tokenizer = generator

    generate_answer(model, tokenizer, "Hello", 4)

    assert model.last_call["pad_token_id"] == tokenizer.pad_token_id


def test_skips_special_tokens_when_decoding(generator):
    model, tokenizer = generator
    seen = {}

    def decode(ids, skip_special_tokens=False):
        seen["skip"] = skip_special_tokens
        return "x"

    tokenizer.decode = decode
    generate_answer(model, tokenizer, "Hello", 2)

    assert seen["skip"] is True


# --- rag_answer ---


@pytest.fixture
def corpus():
    chunks = [
        {"id": "c0", "text": "apple", "source": "s1"},
        {"id": "c1", "text": "banana", "source": "s2"},
    ]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    return chunks, embeddings


def test_returns_answer_sources_and_query(corpus):
    chunks, embeddings = corpus

    out = rag_answer(
        "q", chunks, embeddings, FakeEmbedModel([1.0, 0.0]),
        FakeGenerator(), FakeTokenizer(), k=1,
    )

    assert set(out) == {"answer", "sources", "query"}


def test_documented_example(corpus):
    chunks, embeddings = corpus

    out = rag_answer(
        "q", chunks, embeddings, FakeEmbedModel([1.0, 0.0]),
        FakeGenerator(), FakeTokenizer(), k=1,
    )

    assert (out["query"], out["sources"][0]["id"]) == ("q", "c0")


def test_the_original_query_is_returned_verbatim(corpus):
    chunks, embeddings = corpus

    out = rag_answer(
        "What fruit?", chunks, embeddings, FakeEmbedModel([1.0, 0.0]),
        FakeGenerator(), FakeTokenizer(), k=1,
    )

    assert out["query"] == "What fruit?"


def test_sources_carry_no_scores(corpus):
    chunks, embeddings = corpus

    out = rag_answer(
        "q", chunks, embeddings, FakeEmbedModel([1.0, 0.0]),
        FakeGenerator(), FakeTokenizer(), k=2,
    )

    assert all(isinstance(source, dict) for source in out["sources"])


def test_sources_are_in_ranked_order(corpus):
    chunks, embeddings = corpus

    out = rag_answer(
        "q", chunks, embeddings, FakeEmbedModel([0.0, 1.0]),
        FakeGenerator(), FakeTokenizer(), k=2,
    )

    assert [s["id"] for s in out["sources"]] == ["c1", "c0"]


def test_k_limits_the_sources(corpus):
    chunks, embeddings = corpus

    out = rag_answer(
        "q", chunks, embeddings, FakeEmbedModel([1.0, 0.0]),
        FakeGenerator(), FakeTokenizer(), k=1,
    )

    assert len(out["sources"]) == 1


def test_the_original_query_reaches_the_prompt(corpus):
    """Not the embedded or rewritten form -- the raw string the user asked."""
    chunks, embeddings = corpus
    seen = {}

    class RecordingTokenizer(FakeTokenizer):
        def __call__(self, prompt, return_tensors=None):
            seen["prompt"] = prompt
            return super().__call__(prompt, return_tensors)

    rag_answer(
        "What fruit?", chunks, embeddings, FakeEmbedModel([1.0, 0.0]),
        FakeGenerator(), RecordingTokenizer(), k=1,
    )

    assert "What fruit?" in seen["prompt"]
    assert "apple" in seen["prompt"]


# --- track_source_chunk_ids ---


def test_collects_ids_in_order():
    chunks = [{"id": "doc1::0", "text": "a"}, {"id": "doc1::1", "text": "b"}]

    assert track_source_chunk_ids(chunks) == ["doc1::0", "doc1::1"]


def test_empty_input_gives_an_empty_list():
    assert track_source_chunk_ids([]) == []


def test_chunks_without_an_id_are_skipped():
    chunks = [{"id": "c0"}, {"text": "no id here"}, {"id": "c1"}]

    assert track_source_chunk_ids(chunks) == ["c0", "c1"]


def test_duplicates_are_kept():
    """A set would drop these, and two windows from one chunk is meaningful."""
    assert track_source_chunk_ids([{"id": "c0"}, {"id": "c0"}]) == ["c0", "c0"]


def test_order_is_retrieval_order_not_sorted():
    chunks = [{"id": "z"}, {"id": "a"}, {"id": "m"}]

    assert track_source_chunk_ids(chunks) == ["z", "a", "m"]


def test_reads_id_not_chunk_id():
    """The guide's chunk records use 'chunk_id'; this step reads 'id'."""
    assert track_source_chunk_ids([{"chunk_id": "doc1::0"}]) == []


# --- append_source_references ---


def test_appends_a_sources_line():
    chunks = [{"id": "c0", "text": "a"}, {"id": "c1", "text": "b"}]

    assert append_source_references("The answer is 42.", chunks) == (
        "The answer is 42.\nSources: [c0, c1]"
    )


def test_ids_are_bare_not_quoted():
    result = append_source_references("a", [{"id": "c0"}])

    assert "'" not in result and '"' not in result


def test_integer_ids_render_the_same_way():
    assert append_source_references("a", [{"id": 0}, {"id": 1}]) == (
        "a\nSources: [0, 1]"
    )


def test_no_sources_gives_an_empty_bracket():
    assert append_source_references("a", []) == "a\nSources: []"


def test_the_answer_is_unchanged():
    answer = "Multi\nline\nanswer."

    assert append_source_references(answer, [{"id": "c0"}]).startswith(answer)


def test_order_mirrors_the_input():
    chunks = [{"id": "z"}, {"id": "a"}]

    assert append_source_references("x", chunks) == "x\nSources: [z, a]"


# --- load_generator ---


def test_loads_a_real_generator():
    """Skipped unless transformers is installed; downloads a tiny model."""
    pytest.importorskip("transformers")

    model, tokenizer = load_generator("sshleifer/tiny-gpt2")

    assert tokenizer.pad_token == tokenizer.eos_token
    assert type(model).__name__ == "GPT2LMHeadModel"


def test_real_generation_is_deterministic():
    pytest.importorskip("transformers")

    model, tokenizer = load_generator("sshleifer/tiny-gpt2")

    first = generate_answer(model, tokenizer, "Hello", max_new_tokens=4)
    second = generate_answer(model, tokenizer, "Hello", max_new_tokens=4)

    assert isinstance(first, str) and first == second
