import os

import pytest

from rag_pipeline.ingestion import (
    load_text_directory,
    load_text_file,
    make_document,
    normalize_text,
)


# --- load_text_file ---


def test_load_text_file_returns_full_contents(tmp_path):
    path = tmp_path / "doc.txt"
    path.write_text("first line\nsecond line\n", encoding="utf-8")

    assert load_text_file(str(path)) == "first line\nsecond line\n"


def test_load_text_file_handles_empty_file(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")

    assert load_text_file(str(path)) == ""


def test_load_text_file_decodes_utf8(tmp_path):
    path = tmp_path / "unicode.txt"
    path.write_text("café — naïve — 東京", encoding="utf-8")

    assert load_text_file(str(path)) == "café — naïve — 東京"


def test_load_text_file_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_text_file(str(tmp_path / "nope.txt"))


# --- load_text_directory ---


def test_load_text_directory_returns_contents_in_filename_order(tmp_path):
    (tmp_path / "b.txt").write_text("two", encoding="utf-8")
    (tmp_path / "a.txt").write_text("one", encoding="utf-8")
    (tmp_path / "c.txt").write_text("three", encoding="utf-8")

    assert load_text_directory(str(tmp_path)) == ["one", "two", "three"]


def test_load_text_directory_order_is_lexicographic_not_numeric(tmp_path):
    """doc10 sorts before doc2 -- lexicographic, as the contract promises."""
    (tmp_path / "doc2.txt").write_text("two", encoding="utf-8")
    (tmp_path / "doc10.txt").write_text("ten", encoding="utf-8")

    assert load_text_directory(str(tmp_path)) == ["ten", "two"]


def test_load_text_directory_skips_non_txt_files(tmp_path):
    (tmp_path / "keep.txt").write_text("kept", encoding="utf-8")
    (tmp_path / "skip.md").write_text("skipped", encoding="utf-8")
    (tmp_path / "skip.csv").write_text("skipped", encoding="utf-8")
    (tmp_path / "no_extension").write_text("skipped", encoding="utf-8")

    assert load_text_directory(str(tmp_path)) == ["kept"]


def test_load_text_directory_returns_empty_list_for_empty_directory(tmp_path):
    assert load_text_directory(str(tmp_path)) == []


def test_load_text_directory_returns_empty_list_when_no_txt_files(tmp_path):
    (tmp_path / "notes.md").write_text("not a txt", encoding="utf-8")

    assert load_text_directory(str(tmp_path)) == []


def test_load_text_directory_does_not_recurse(tmp_path):
    (tmp_path / "top.txt").write_text("top", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "deep.txt").write_text("deep", encoding="utf-8")

    assert load_text_directory(str(tmp_path)) == ["top"]


def test_load_text_directory_ignores_directories_named_like_txt(tmp_path):
    (tmp_path / "real.txt").write_text("real", encoding="utf-8")
    (tmp_path / "trap.txt").mkdir()

    with pytest.raises(IsADirectoryError):
        load_text_directory(str(tmp_path))


def test_load_text_directory_raises_on_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_text_directory(os.path.join(str(tmp_path), "nope"))


# --- extract_text_from_html  -- not yet implemented ---


# --- normalize_text ---

# --- the documented examples ---


def test_normalize_collapses_runs_and_strips():
    assert normalize_text("  hello   world\n") == "hello world"


def test_normalize_collapses_tabs_and_newlines():
    assert normalize_text("foo\t\tbar\nbaz") == "foo bar baz"


# --- whitespace collapsing ---


def test_normalize_collapses_crlf():
    assert normalize_text("a\r\nb") == "a b"


def test_normalize_collapses_mixed_whitespace_run():
    assert normalize_text("a \t\n\r\x0b\x0c b") == "a b"


def test_normalize_empty_string_returns_empty():
    assert normalize_text("") == ""


def test_normalize_whitespace_only_returns_empty():
    assert normalize_text("  \n\t  ") == ""


def test_normalize_single_word_is_unchanged():
    assert normalize_text("word") == "word"


def test_normalize_interior_single_spaces_are_preserved():
    assert normalize_text("already tidy text") == "already tidy text"


def test_normalize_is_idempotent():
    once = normalize_text("  Ｈello \t wörld\n")
    assert normalize_text(once) == once


# --- NFKC folding ---


def test_normalize_folds_fullwidth_characters():
    assert normalize_text("Ａ１") == "A1"


def test_normalize_folds_ligatures():
    assert normalize_text("ﬁle") == "file"


def test_normalize_folds_superscripts():
    assert normalize_text("x²") == "x2"


def test_normalize_folds_micro_sign_to_greek_mu():
    assert normalize_text("µm") == "μm"


def test_normalize_folds_combining_sequence_to_precomposed():
    assert normalize_text("café") == "café"


def test_normalize_runs_nfkc_before_collapsing():
    """NBSP and em space become ordinary spaces, then collapse into one."""
    assert normalize_text("a  b") == "a b"


def test_normalize_folds_ideographic_space():
    assert normalize_text("　東　京　") == "東 京"


# --- documented non-behaviour ---


def test_normalize_leaves_zero_width_space():
    """ZWSP is not whitespace to str.split() and has no NFKC decomposition."""
    assert normalize_text("a​b") == "a​b"


def test_normalize_leaves_bom():
    assert normalize_text("﻿hello") == "﻿hello"


def test_normalize_leaves_soft_hyphen():
    assert normalize_text("co­operate") == "co­operate"


def test_normalize_preserves_case():
    assert normalize_text("Hello World") == "Hello World"


def test_normalize_preserves_punctuation():
    assert normalize_text("Hello, world!  (Really.)") == "Hello, world! (Really.)"


# --- make_document ---

# --- the documented example ---


def test_make_document_documented_example():
    assert make_document("Hello world.", "notes.txt", "Greeting") == {
        "text": "Hello world.",
        "source": "notes.txt",
        "title": "Greeting",
    }


# --- the schema contract ---


def test_make_document_has_exactly_the_three_contract_keys():
    doc = make_document("t", "s", "ti")

    assert set(doc) == {"text", "source", "title"}


def test_make_document_keys_are_in_contract_order():
    """`==` ignores key order, so the order has to be asserted explicitly."""
    doc = make_document("t", "s", "ti")

    assert list(doc) == ["text", "source", "title"]


def test_make_document_metadata_is_flat_not_nested():
    doc = make_document("t", "s", "ti")

    assert "meta" not in doc
    assert doc["source"] == "s"


def test_make_document_arguments_map_to_their_own_keys():
    doc = make_document("the payload", "the source", "the title")

    assert doc["text"] == "the payload"
    assert doc["source"] == "the source"
    assert doc["title"] == "the title"


def test_make_document_arguments_are_positional_in_declared_order():
    assert make_document("a", "b", "c") == make_document(
        text="a", source="b", title="c"
    )


# --- values are stored as-is ---


def test_make_document_stores_text_verbatim():
    """No normalising here -- that is normalize_text's job, called by the caller."""
    doc = make_document("  messy\ttext\n", "s", "t")

    assert doc["text"] == "  messy\ttext\n"


def test_make_document_preserves_empty_strings():
    doc = make_document("", "", "")

    assert doc == {"text": "", "source": "", "title": ""}


def test_make_document_does_not_reject_none():
    """No validation: the caller owns what counts as a usable source or title."""
    doc = make_document("t", None, None)

    assert doc["source"] is None
    assert doc["title"] is None


# --- independence between calls ---


def test_make_document_returns_a_fresh_dict_each_call():
    first = make_document("t", "s", "ti")
    second = make_document("t", "s", "ti")

    assert first == second
    assert first is not second


def test_make_document_mutation_does_not_affect_later_calls():
    first = make_document("t", "s", "ti")
    first["text"] = "mutated"

    assert make_document("t", "s", "ti")["text"] == "t"


# --- The steps composed end to end ---


def test_ingesting_a_directory_into_documents(tmp_path):
    """load -> normalize -> make_document, the way the README shows it."""
    (tmp_path / "a.txt").write_text("  Hello   world.\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Ｇoodbye\tﬁle", encoding="utf-8")
    (tmp_path / "skip.md").write_text("ignored", encoding="utf-8")

    corpus = str(tmp_path)
    filenames = sorted(f for f in os.listdir(corpus) if f.endswith(".txt"))
    texts = load_text_directory(corpus)

    documents = [
        make_document(normalize_text(text), filename, filename)
        for text, filename in zip(texts, filenames)
    ]

    assert documents == [
        {"text": "Hello world.", "source": "a.txt", "title": "a.txt"},
        {"text": "Goodbye file", "source": "b.txt", "title": "b.txt"},
    ]
