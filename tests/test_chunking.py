import pytest

from rag_pipeline.chunking import chunk_by_tokens, chunk_fixed_size


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
