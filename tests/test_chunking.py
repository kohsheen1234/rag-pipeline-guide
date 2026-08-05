import pytest

from rag_pipeline.chunking import (
    attach_chunk_metadata,
    chunk_by_sentences,
    chunk_by_tokens,
    chunk_fixed_size,
    chunk_with_overlap,
)


class WordTokenizer:
    """A stand-in with the encode/decode surface chunk_by_tokens relies on.

    One token per whitespace-separated word, so token counts are obvious in
    the assertions below. Real tokenizers are heavier and need a download; the
    behaviour under test is the slicing, not the vocabulary.
    """

    def __init__(self):
        self.words = []
        self.ids = {}

    def encode(self, text):
        token_ids = []
        for word in text.split():
            if word not in self.ids:
                self.ids[word] = len(self.words)
                self.words.append(word)
            token_ids.append(self.ids[word])
        return token_ids

    def decode(self, token_ids):
        return " ".join(self.words[i] for i in token_ids)


class SpecialTokenTokenizer(WordTokenizer):
    """BERT-style: wraps every encoding in [CLS] ... [SEP]."""

    CLS, SEP = -1, -2

    def encode(self, text):
        return [self.CLS] + super().encode(text) + [self.SEP]

    def decode(self, token_ids):
        return " ".join(
            "[CLS]" if i == self.CLS else "[SEP]" if i == self.SEP else self.words[i]
            for i in token_ids
        )


@pytest.fixture
def tokenizer():
    return WordTokenizer()


# --- chunk_fixed_size ---

# the documented examples


def test_splits_with_a_short_final_chunk():
    assert chunk_fixed_size("abcdefgh", 3) == ["abc", "def", "gh"]


def test_text_shorter_than_or_equal_to_chunk_size_is_one_chunk():
    assert chunk_fixed_size("hello", 5) == ["hello"]


# sizes and boundaries


def test_exact_multiple_has_no_short_chunk():
    assert chunk_fixed_size("abcdef", 3) == ["abc", "def"]


def test_text_shorter_than_chunk_size():
    assert chunk_fixed_size("ab", 5) == ["ab"]


def test_chunk_size_of_one_splits_every_character():
    assert chunk_fixed_size("abc", 1) == ["a", "b", "c"]


def test_empty_text_yields_no_chunks():
    assert chunk_fixed_size("", 3) == []


def test_one_character_over_a_multiple_leaves_a_single_character_chunk():
    assert chunk_fixed_size("abcdefg", 6) == ["abcdef", "g"]


# the two invariants, over a spread of inputs


@pytest.mark.parametrize("length", [0, 1, 2, 5, 10, 33, 100])
@pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, 50])
def test_chunks_rejoin_to_the_original(length, chunk_size):
    text = "x" * length

    assert "".join(chunk_fixed_size(text, chunk_size)) == text


@pytest.mark.parametrize("length", [0, 1, 2, 5, 10, 33, 100])
@pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, 50])
def test_every_chunk_is_within_size_bounds(length, chunk_size):
    chunks = chunk_fixed_size("x" * length, chunk_size)

    assert all(1 <= len(chunk) <= chunk_size for chunk in chunks)


@pytest.mark.parametrize("length", [0, 1, 2, 5, 10, 33, 100])
@pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, 50])
def test_chunk_count_is_the_ceiling_of_the_division(length, chunk_size):
    expected = -(-length // chunk_size)  # ceil, without importing math

    assert len(chunk_fixed_size("x" * length, chunk_size)) == expected


def test_all_chunks_but_the_last_are_exactly_chunk_size():
    chunks = chunk_fixed_size("a" * 20, 6)

    assert [len(c) for c in chunks] == [6, 6, 6, 2]


def test_windows_do_not_overlap():
    """Rejoining is only equal to the original if nothing was duplicated."""
    text = "abcdefghij"
    chunks = chunk_fixed_size(text, 4)

    assert chunks == ["abcd", "efgh", "ij"]
    assert "".join(chunks) == text


# content is not inspected


def test_whitespace_and_newlines_are_split_like_any_character():
    assert chunk_fixed_size("a b\nc", 2) == ["a ", "b\n", "c"]


def test_splits_mid_word():
    """No word-boundary awareness -- that is a later chunker's job."""
    assert chunk_fixed_size("hello world", 4) == ["hell", "o wo", "rld"]


def test_counts_characters_not_bytes():
    text = "東京タワー"

    assert chunk_fixed_size(text, 2) == ["東京", "タワ", "ー"]


# documented behaviour for invalid sizes


def test_chunk_size_of_zero_raises():
    with pytest.raises(ValueError):
        chunk_fixed_size("abc", 0)


@pytest.mark.parametrize("chunk_size", [-1, -3])
def test_negative_chunk_size_silently_discards_the_text(chunk_size):
    """Documented, not endorsed -- see the step doc."""
    assert chunk_fixed_size("abcdef", chunk_size) == []


# --- chunk_by_tokens ---


def test_splits_on_token_count(tokenizer):
    text = "hello world this is a small example"  # 7 tokens

    assert chunk_by_tokens(text, tokenizer, 3) == [
        "hello world this",
        "is a small",
        "example",
    ]


def test_chunk_count_is_ceiling_of_token_count(tokenizer):
    text = "hello world this is a small example"
    total = len(tokenizer.encode(text))

    assert len(chunk_by_tokens(text, tokenizer, 3)) == -(-total // 3)


def test_empty_text_yields_no_chunks(tokenizer):
    assert chunk_by_tokens("", tokenizer, 3) == []


def test_text_shorter_than_max_tokens_is_one_chunk(tokenizer):
    assert chunk_by_tokens("hello world", tokenizer, 10) == ["hello world"]


def test_exact_multiple_has_no_short_chunk(tokenizer):
    assert chunk_by_tokens("a b c d", tokenizer, 2) == ["a b", "c d"]


def test_max_tokens_of_one_gives_one_token_per_chunk(tokenizer):
    assert chunk_by_tokens("a b c", tokenizer, 1) == ["a", "b", "c"]


def test_every_chunk_is_within_the_token_bound(tokenizer):
    text = " ".join(str(n) for n in range(50))

    chunks = chunk_by_tokens(text, tokenizer, 7)

    assert all(1 <= len(tokenizer.encode(c)) <= 7 for c in chunks)


def test_chunks_are_in_document_order(tokenizer):
    text = " ".join(str(n) for n in range(10))

    assert chunk_by_tokens(text, tokenizer, 4) == ["0 1 2 3", "4 5 6 7", "8 9"]


def test_uses_encode_not_call(tokenizer):
    """Calling the tokenizer would return a dict; encode returns a flat list."""
    text = "a b c"

    assert chunk_by_tokens(text, tokenizer, 2) == ["a b", "c"]
    assert isinstance(tokenizer.encode(text), list)


def test_special_tokens_break_the_empty_input_contract():
    """Documented fragility: the empty-list result is the tokenizer's doing.

    A BERT-style tokenizer encodes "" to [CLS] [SEP], so the function returns a
    chunk of special tokens rather than nothing.
    """
    assert chunk_by_tokens("", SpecialTokenTokenizer(), 3) == ["[CLS] [SEP]"]


@pytest.mark.parametrize("max_tokens", [-1, -3])
def test_negative_max_tokens_silently_discards_the_text(tokenizer, max_tokens):
    assert chunk_by_tokens("a b c", tokenizer, max_tokens) == []


def test_max_tokens_of_zero_raises(tokenizer):
    with pytest.raises(ValueError):
        chunk_by_tokens("a b c", tokenizer, 0)


# --- chunk_by_sentences ---

# the documented examples

SENTENCES = "Hello world. How are you? I am fine."


def test_packs_sentences_up_to_the_limit():
    assert chunk_by_sentences(SENTENCES, 30) == [
        "Hello world. How are you?",
        "I am fine.",
    ]


def test_a_tighter_limit_gives_one_sentence_per_chunk():
    assert chunk_by_sentences(SENTENCES, 20) == [
        "Hello world.",
        "How are you?",
        "I am fine.",
    ]


# packing


def test_everything_fits_in_a_single_chunk():
    assert chunk_by_sentences(SENTENCES, 100) == [SENTENCES]


def test_packing_respects_the_joining_space():
    """'One. Two.' is 9 chars: 4 + 1 separator + 4."""
    assert chunk_by_sentences("One. Two.", 9) == ["One. Two."]
    assert chunk_by_sentences("One. Two.", 8) == ["One.", "Two."]


def test_chunks_stay_within_the_limit_when_sentences_allow_it():
    text = " ".join(f"Sentence number {n}." for n in range(30))

    assert all(len(c) <= 60 for c in chunk_by_sentences(text, 60))


def test_sentences_stay_in_document_order():
    text = "One. Two. Three. Four."

    assert chunk_by_sentences(text, 10) == ["One. Two.", "Three.", "Four."]


# terminators


def test_terminator_stays_attached_to_its_sentence():
    for chunk in chunk_by_sentences("Stop! Go? Wait.", 6):
        assert chunk[-1] in ".!?"


def test_handles_all_three_terminators():
    assert chunk_by_sentences("A. B! C?", 2) == ["A.", "B!", "C?"]


def test_repeated_terminators_stay_with_the_sentence():
    assert chunk_by_sentences("Wait!!! Really?", 50) == ["Wait!!! Really?"]


def test_text_with_no_terminator_is_one_sentence():
    assert chunk_by_sentences("no terminator here", 50) == ["no terminator here"]


def test_final_sentence_without_a_terminator_is_kept():
    text = "Ends with one. Like this"

    assert chunk_by_sentences(text, 50) == [text]


# empty and overlong


def test_empty_text_yields_no_chunks():
    assert chunk_by_sentences("", 10) == []


@pytest.mark.parametrize("text", ["   ", "\n\t ", "  \n  "])
def test_whitespace_only_yields_no_chunks(text):
    assert chunk_by_sentences(text, 10) == []


def test_no_empty_chunks_are_produced():
    assert all(chunk_by_sentences("A.  B.   C.", 4))


def test_an_overlong_sentence_is_emitted_whole():
    """Never split mid-sentence, even though it breaks the limit."""
    text = "A very long single sentence that exceeds the limit."

    assert chunk_by_sentences(text, 10) == [text]


def test_an_overlong_sentence_does_not_swallow_the_next_one():
    text = "A very long single sentence that exceeds it. Short."

    assert chunk_by_sentences(text, 10) == [
        "A very long single sentence that exceeds it.",
        "Short.",
    ]


def test_no_content_is_dropped_when_a_sentence_overflows():
    text = "Tiny. " + "x" * 100 + ". End."

    assert "".join(chunk_by_sentences(text, 20)).count("x") == 100


# documented limitations of the naive split


def test_decimals_are_split_and_a_space_is_inserted():
    """Corrupts the number -- see the step doc."""
    assert chunk_by_sentences("It costs 3.5 dollars.", 50) == ["It costs 3. 5 dollars."]


def test_abbreviations_split_once_packing_overflows():
    assert chunk_by_sentences("Dr. Smith went home.", 50) == ["Dr. Smith went home."]
    assert chunk_by_sentences("Dr. Smith went home.", 10) == [
        "Dr.",
        "Smith went home.",
    ]


def test_leading_terminators_are_dropped():
    assert chunk_by_sentences("...leading dots then text.", 50) == [
        "leading dots then text."
    ]


def test_whitespace_is_normalised_only_at_sentence_boundaries():
    assert chunk_by_sentences("Hi.  Lots   of   space.", 50) == [
        "Hi. Lots   of   space."
    ]


# --- chunk_with_overlap ---

# the documented examples


def test_windows_share_overlap_characters():
    assert chunk_with_overlap("abcdefghij", 4, 2) == [
        "abcd",
        "cdef",
        "efgh",
        "ghij",
        "ij",
    ]


def test_zero_overlap_matches_fixed_size_chunking():
    assert chunk_with_overlap("abcdef", 3, 0) == ["abc", "def"]


@pytest.mark.parametrize("length", [0, 1, 5, 20, 47])
@pytest.mark.parametrize("chunk_size", [1, 3, 8])
def test_zero_overlap_is_equivalent_to_chunk_fixed_size(length, chunk_size):
    text = "abcdefghij" * 5
    text = text[:length]

    assert chunk_with_overlap(text, chunk_size, 0) == chunk_fixed_size(text, chunk_size)


# the step size


def test_step_is_chunk_size_minus_overlap():
    chunks = chunk_with_overlap("abcdefghij", 5, 1)

    assert chunks == ["abcde", "efghi", "ij"]


def test_consecutive_chunks_share_exactly_overlap_characters():
    chunk_size, overlap = 5, 2
    chunks = chunk_with_overlap("abcdefghijklmno", chunk_size, overlap)

    for earlier, later in zip(chunks, chunks[1:]):
        if len(earlier) == chunk_size:
            assert earlier[-overlap:] == later[:overlap]


def test_heavier_overlap_produces_more_chunks():
    text = "abcdefghij"

    assert len(chunk_with_overlap(text, 4, 3)) > len(chunk_with_overlap(text, 4, 1))


# coverage


@pytest.mark.parametrize("overlap", [0, 1, 2, 3])
def test_every_character_is_covered(overlap):
    text = "abcdefghijklmno"

    joined = "".join(chunk_with_overlap(text, 4, overlap))

    assert all(character in joined for character in text)


def test_first_chunk_starts_at_the_beginning():
    assert chunk_with_overlap("abcdefghij", 4, 2)[0] == "abcd"


def test_last_chunk_reaches_the_end():
    assert chunk_with_overlap("abcdefghij", 4, 2)[-1].endswith("j")


# boundaries


def test_empty_text_yields_no_chunks():
    assert chunk_with_overlap("", 4, 2) == []


def test_text_shorter_than_chunk_size_is_one_chunk():
    assert chunk_with_overlap("abc", 10, 2) == ["abc"]


def test_no_chunk_exceeds_chunk_size():
    chunks = chunk_with_overlap("abcdefghijklmno", 4, 2)

    assert all(1 <= len(chunk) <= 4 for chunk in chunks)


def test_overlap_equal_to_chunk_size_raises():
    """Step would be zero -- the window would never advance."""
    with pytest.raises(ValueError):
        chunk_with_overlap("abcdefgh", 4, 4)


def test_overlap_greater_than_chunk_size_silently_discards_the_text():
    assert chunk_with_overlap("abcdefgh", 4, 5) == []


def test_trailing_chunks_can_repeat_earlier_content():
    """Documented redundancy: 'ij' is entirely inside 'ghij'."""
    chunks = chunk_with_overlap("abcdefghij", 4, 2)

    assert chunks[-1] in chunks[-2]


def test_heavy_overlap_produces_several_redundant_tail_chunks():
    chunks = chunk_with_overlap("abcdefghij", 4, 3)

    redundant = [c for i, c in enumerate(chunks) if i and c in chunks[i - 1]]
    assert redundant == ["hij", "ij", "j"]


# --- attach_chunk_metadata ---


def test_documented_example():
    assert attach_chunk_metadata(["hello", "world"], "doc1") == [
        {"text": "hello", "source": "doc1", "position": 0, "chunk_id": "doc1::0"},
        {"text": "world", "source": "doc1", "position": 1, "chunk_id": "doc1::1"},
    ]


def test_record_has_exactly_the_four_contract_keys():
    record = attach_chunk_metadata(["a"], "doc1")[0]

    assert set(record) == {"text", "source", "position", "chunk_id"}


def test_keys_are_in_contract_order():
    record = attach_chunk_metadata(["a"], "doc1")[0]

    assert list(record) == ["text", "source", "position", "chunk_id"]


def test_positions_are_zero_indexed_and_sequential():
    records = attach_chunk_metadata(list("abcde"), "doc1")

    assert [r["position"] for r in records] == [0, 1, 2, 3, 4]


def test_position_is_an_int_not_a_string():
    assert attach_chunk_metadata(["a"], "doc1")[0]["position"] == 0


def test_chunk_id_joins_source_and_position():
    records = attach_chunk_metadata(list("ab"), "notes.txt")

    assert [r["chunk_id"] for r in records] == ["notes.txt::0", "notes.txt::1"]


def test_ids_are_unique_even_when_chunks_are_identical():
    """Why position, not a content hash: duplicate text still gets distinct ids."""
    records = attach_chunk_metadata(["same", "same", "same"], "doc1")

    assert len({r["chunk_id"] for r in records}) == 3


def test_ids_differ_across_sources():
    a = attach_chunk_metadata(["x"], "a.txt")[0]["chunk_id"]
    b = attach_chunk_metadata(["x"], "b.txt")[0]["chunk_id"]

    assert a != b


def test_chunk_order_is_preserved():
    chunks = ["first", "second", "third"]

    assert [r["text"] for r in attach_chunk_metadata(chunks, "doc1")] == chunks


def test_text_is_stored_verbatim():
    assert attach_chunk_metadata(["  raw\ttext  "], "d")[0]["text"] == "  raw\ttext  "


def test_empty_chunk_list_yields_no_records():
    assert attach_chunk_metadata([], "doc1") == []


def test_each_record_is_a_separate_dict():
    records = attach_chunk_metadata(["a", "b"], "doc1")
    records[0]["text"] = "mutated"

    assert records[1]["text"] == "b"


def test_composes_with_a_chunker():
    records = attach_chunk_metadata(chunk_fixed_size("abcdefgh", 3), "doc1")

    assert [r["chunk_id"] for r in records] == ["doc1::0", "doc1::1", "doc1::2"]
    assert [r["text"] for r in records] == ["abc", "def", "gh"]


def test_against_a_real_tokenizer():
    """Skipped unless transformers is installed; downloads a tiny model."""
    transformers = pytest.importorskip("transformers")

    tok = transformers.AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    text = "hello world this is a small example"
    total = len(tok.encode(text))

    chunks = chunk_by_tokens(text, tok, 3)

    assert len(chunks) == -(-total // 3)
    assert all(1 <= len(tok.encode(c)) <= 3 for c in chunks)
    assert chunk_by_tokens("", tok, 3) == []
