# Step 4 · `normalize_text`

> **Part 1 · Document Ingestion & Preprocessing** — step 4 of 51
> Code: [`rag_pipeline/ingestion.py`](../../rag_pipeline/ingestion.py) · Tests: [`tests/test_ingestion.py`](../../tests/test_ingestion.py)
> Previous: [Step 2 · `load_text_directory`](step-02-load-text-directory.md)

---

## The task

```python
def normalize_text(text: str) -> str: ...
```

Clean a raw text string before it gets chunked. Apply Unicode NFKC
normalisation to fold compatibility variants into canonical forms, collapse any
run of whitespace (spaces, tabs, newlines) into a single space, and strip
leading and trailing whitespace. The result is a single tidy line.

---

## Why this step exists

Raw text from files, HTML, and PDFs is full of inconsistent whitespace and
unicode look-alikes. Retrieval is a similarity comparison, and the embedding
model has no idea that `Ａ` (fullwidth) and `A` are the same letter to a human —
it sees different tokens and produces a different vector.

The concrete failure: the same sentence appearing in two documents, one pasted
from a PDF with non-breaking spaces and one typed plainly, produces two
different embeddings. A query that matches one will rank the other lower than it
should. Nothing errors; retrieval just quietly gets worse. Normalising at
ingestion means the model only ever sees one spelling of a given passage.

---

## What's happening

```python
text = unicodedata.normalize("NFKC", text)
return " ".join(text.split())
```

Two lines, three transformations.

### `unicodedata.normalize("NFKC", text)`

Unicode has four normalisation forms, along two axes. **Composition** (C vs D)
decides whether `e` + combining acute becomes the single character `é` or stays
as two. **Compatibility** (K or not) decides whether characters that merely
*look* like others get folded onto them.

NFKC is the aggressive corner of that square: compose, and fold compatibility
variants. That is what makes it right here:

| Input | NFKC output | What it was |
| --- | --- | --- |
| `Ａ１` | `A1` | Fullwidth forms, common in CJK text |
| `ﬁ` | `fi` | Typographic ligature, common from PDFs |
| `x²` | `x2` | Superscript |
| `µ` (U+00B5) | `μ` (U+03BC) | Micro sign vs Greek mu — visually identical |
| `Ⅸ` | `IX` | Roman numeral character |
| `e` + `◌́` | `é` | Combining sequence composed |
| NBSP, em space, ideographic space | ordinary space | Exotic spaces |

That last row is why the ordering matters — see below.

### `" ".join(text.split())`

`str.split()` **with no argument** is a different function from
`str.split(" ")`. With no argument it splits on *runs* of whitespace and discards
empty leading and trailing fields. So it collapses and strips in one move:

```python
"  a \t\n b  ".split()        → ['a', 'b']      # runs collapsed, ends dropped
"  a \t\n b  ".split(" ")     → ['', '', 'a', '\t\n', 'b', '', '']
```

Joining with `" "` then rebuilds the string with exactly one space between
words. The whitespace it recognises is Unicode-aware — anything for which
`str.isspace()` is true, including `\r`, `\x0b`, `\x0c`, and NBSP.

---

## Why NFKC must come first

The guide flags this and it is worth being precise about why. NFKC *produces*
whitespace. `U+00A0` (non-breaking space), `U+2003` (em space), and `U+3000`
(ideographic space) all fold to an ordinary `U+0020`.

Collapse first and you get:

```python
>>> " ".join("a  b".split())     # then NFKC
'a b'
```

which happens to look right, because Python's `split()` already treats NBSP and
em space as whitespace. But the guarantee is accidental — it depends on
`str.isspace()` agreeing with NFKC about every exotic space, for every character
NFKC might ever emit. Normalising first makes the invariant unconditional:
whatever spaces NFKC generates are collapsed by construction.

---

## Boundaries of the contract

**Newlines are destroyed.** This is the significant one. Every paragraph break
becomes a single space, so a chunker downstream cannot split on paragraph
boundaries — the information is gone. That is a real trade-off, not an
oversight: this step exists to produce "a single tidy line", and structure-aware
chunking needs a different preprocessing path that preserves `\n\n`. Worth
remembering when Part 2 starts.

**Zero-width characters survive.** These are neither whitespace to `split()` nor
decomposed by NFKC, so they pass straight through:

| Character | Why it survives |
| --- | --- |
| `U+200B` zero-width space | `isspace()` is `False`; no NFKC decomposition |
| `U+FEFF` BOM / zero-width no-break space | Same — a leading BOM stays in the string |
| `U+00AD` soft hyphen | Same — common in PDF text extraction |

Each is invisible on screen and splits a word in two as far as a tokeniser is
concerned. If they show up in your corpus, they need explicit removal; this
function will not do it for you.

**NFKC is lossy.** `x²` becomes `x2`, so `x²` and `x2` are no longer
distinguishable. For a maths or chemistry corpus that is a real information loss
and NFC (canonical only, no compatibility folding) would be the safer choice.

**Case is preserved.** Lowercasing is a separate decision — modern embedding
models are trained on mixed case and generally do better with it.

**Punctuation is preserved.** Nothing here strips or standardises punctuation.

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| `text.split(" ")` instead of `text.split()` | Only splits on literal spaces. Tabs and newlines survive untouched, and empty strings appear in the output for every doubled space. |
| `text.replace(" ", "")` | Glues words together. `"hello world"` → `"helloworld"`, destroying the word boundaries embeddings depend on. |
| Collapsing before normalising | Works by coincidence today. The invariant is only guaranteed with NFKC first. |
| Using NFC instead of NFKC | Composes accents but leaves fullwidth forms, ligatures, and superscripts alone — exactly the look-alikes you wanted folded. |
| `re.sub(r"\s+", " ", text).strip()` | Fine, and equivalent for most input, but `\s` in Python 3 `str` patterns is Unicode-aware in a subtly different set than `str.isspace()`. The `split()`/`join()` idiom is clearer and needs no import. |
| Assuming the output is ASCII | It is not. NFKC folds *compatibility* variants; `東京`, `é`, and emoji all pass through unchanged, as they should. |

---

## Example

```python
>>> normalize_text('  hello   world\n')
'hello world'
>>> normalize_text('foo\t\tbar\nbaz')
'foo bar baz'
>>> normalize_text('Ｈello  ﬁle')
'Hello file'
>>> normalize_text('   ')
''
```

`normalize_text` is **idempotent** — running it on its own output changes
nothing, which is what you want from a normaliser that may be applied at both
ingestion and query time.

---

## Where it fits

```
  list[str] ──▶ [ normalize_text ] ──▶ list[str] ──▶ Part 2 · Chunking
   (loaders)      per document          tidy lines
```

Apply it **at both ends**. Documents are normalised at ingestion; queries must
be normalised at search time with the same function, or a query containing a
fullwidth character will fail to match a corpus that had them folded away. Any
asymmetry between the two paths reintroduces exactly the mismatch this step
exists to remove.
