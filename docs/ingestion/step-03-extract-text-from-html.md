# Step 3 · `extract_text_from_html`

> **Part 1 · Document Ingestion & Preprocessing** — step 3 of 51
> Code: [`rag_pipeline/ingestion.py`](../../rag_pipeline/ingestion.py) · Tests: [`tests/test_ingestion.py`](../../tests/test_ingestion.py)
> Previous: [Step 2 · `load_text_directory`](step-02-load-text-directory.md) · Next: [Step 4 · `normalize_text`](step-04-normalize-text.md)

---

## The task

```python
def extract_text_from_html(html: str) -> str: ...
```

Take an HTML string and return only the visible text content, with all tags
removed. Skip the contents of non-visible elements like `<script>` and
`<style>` so their bodies do not leak into the extracted text. Decode HTML
entities such as `&amp;` to their character form.

---

## Why this step exists

Real corpora arrive as HTML — scraped pages, exported wikis, documentation
sites. Downstream chunking and embedding want human-readable prose, and
everything else in the file is noise that costs you twice: it wastes context
window, and it pollutes the embedding so retrieval matches on markup instead of
meaning.

`<script>` bodies are the sharpest case. A page with an inline analytics blob
can carry more JavaScript than prose. Embed that and the document's vector is
substantially a description of its tracking code.

---

## What's happening

The work is done by a subclass of the standard library's
[`html.parser.HTMLParser`](https://docs.python.org/3/library/html.parser.html),
an event-driven parser that calls a method on you for each thing it encounters.

```python
class VisibleTextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.text_parts = []
        self.hidden_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag.lower() in HIDDEN_ELEMENTS:
            self.hidden_depth += 1

    def handle_endtag(self, tag):
        if tag.lower() in HIDDEN_ELEMENTS and self.hidden_depth > 0:
            self.hidden_depth -= 1

    def handle_data(self, data):
        if self.hidden_depth == 0:
            self.text_parts.append(data)
```

### The three callbacks

`handle_data` fires for every text node — the content *between* tags. Collecting
those and ignoring everything else is the whole extraction strategy. Tags are
not removed so much as never collected in the first place, which is why
attributes, comments, and doctypes cannot leak: the parser routes them to
methods we did not override.

`handle_starttag` and `handle_endtag` exist only to track whether we are
currently inside something whose text should be dropped.

### `convert_charrefs=True`

This makes the parser decode character references before handing text to
`handle_data`, so `&amp;` arrives as `&` and `&#39;` as `'`. It covers both
named and numeric forms, including astral ones — `&#8212;` becomes `—`.

It is the default in modern Python, but naming it explicitly documents the
intent rather than relying on a default that once differed.

### Why a depth counter, not a boolean

`hidden_depth` is an integer rather than an `in_script` flag, and the guard
`and self.hidden_depth > 0` matters more than it looks.

Malformed HTML is the normal case, not the exception. A stray `</script>` with
no matching opening tag would drive a naive counter to `-1`, and then the next
real `<script>` would only bring it back to `0` — leaving the parser convinced
it is looking at visible text while reading JavaScript. Flooring at zero makes
the stray tag a no-op instead:

```python
>>> extract_text_from_html("</script><p>visible</p>")
'visible'
```

A boolean flag would behave acceptably here too, but the counter also handles
the nesting case correctly and costs nothing.

### `.strip()`

Only the very ends of the joined result are trimmed. Interior whitespace is left
exactly as the markup had it, because collapsing it is
[`normalize_text`](step-04-normalize-text.md)'s job — and doing it here as well
would mean two functions quietly competing over the same concern.

---

## The word-gluing problem

This is the one thing to know about this function, because it is a real defect
in the output and it is **not** recoverable downstream.

Text nodes are joined with `""`. Nothing is inserted at element boundaries. So
whatever whitespace the markup happened to contain is the only thing separating
words:

```python
>>> extract_text_from_html('<p>one</p><p>two</p>')
'onetwo'
>>> extract_text_from_html('<li>a</li><li>b</li>')
'ab'
>>> extract_text_from_html('<div>a<br>b</div>')
'ab'
```

Two paragraphs become one nonsense token. Minified HTML — which is most HTML
served in production — has no whitespace between block elements at all, so this
fires constantly on exactly the input you are most likely to feed it.

And `normalize_text` cannot save you. It collapses runs of whitespace; it cannot
invent a boundary that was never there:

```python
>>> normalize_text(extract_text_from_html('<p>one</p><p>two</p>'))
'onetwo'
```

The information is destroyed at this step. Anything the chunker or the embedding
model sees afterwards is already wrong.

**Why not just fix it?** Inserting a space at every tag boundary would break the
documented contract:

```python
'<p>Hello <b>World</b></p>'  ->  'Hello  World'   # two spaces, not one
```

The specified behaviour for the given example requires exactly `'Hello World'`,
and inline elements like `<b>` must *not* introduce a break — `<b>` splitting a
word is legitimate markup. A correct fix distinguishes block-level elements
(`p`, `div`, `li`, `br`, `h1`…) from inline ones and inserts a separator only
for the former, then relies on `normalize_text` to collapse the result. That is
a real design decision with a tag list attached, so it is left out of a step
whose stated scope is "remove tags and decode entities" — but it is the first
thing to revisit if HTML is a serious part of your corpus.

---

## Boundaries of the contract

**An unclosed `<script>` swallows the rest of the document.** With no closing
tag, `hidden_depth` never returns to zero:

```python
>>> extract_text_from_html('<p>unclosed <script>secret')
'unclosed'
```

Silent truncation. Arguably the safe failure — leaking a script body is worse
than losing prose — but it is worth knowing that a malformed page can come back
nearly empty.

**Only `script` and `style` are hidden.** Real boilerplate lives in `<nav>`,
`<header>`, `<footer>`, and cookie banners, and all of it is extracted as prose.
Stripping it is a genuinely harder problem (it needs heuristics, not a tag list)
and is what libraries like `trafilatura` and `readability` exist for.

**`<textarea>` and `<title>` are extracted.** Neither is body prose in the usual
sense; `<title>` is arguably useful, form defaults usually are not.

**Malformed markup never raises.** `HTMLParser` is lenient by design. You get
best-effort output, not an error — so a broken page fails silently rather than
loudly, which is the opposite of the rule the loaders follow.

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| `re.sub(r"<[^>]+>", "", html)` | Breaks on `<p title="a > b">` — the regex ends the match at the `>` inside the attribute and emits `b">` as text. Leaves `<script>` bodies entirely intact. Does not decode entities. Three separate failures in one line. |
| Forgetting `.lower()` on the tag | Actually harmless with `HTMLParser`, which lowercases tag names already — but the habit is right, and it is load-bearing with parsers that do not. |
| A boolean `in_script` flag | A stray `</script>` flips it to "visible" and the next real script body leaks through. |
| Omitting the `hidden_depth > 0` guard | The counter goes negative on unbalanced markup and the next hidden element is not hidden. |
| Collapsing whitespace here too | Duplicates `normalize_text`, and makes two functions responsible for one decision. |
| Assuming the output is clean prose | It is tags-removed text, not article text. Navigation and footers are still in there. |

---

## Example

```python
>>> extract_text_from_html('<p>Hello <b>World</b></p>')
'Hello World'
>>> extract_text_from_html('<p>a &amp; b</p>')
'a & b'
>>> extract_text_from_html('<script>var x = 1;</script>hi')
'hi'
>>> extract_text_from_html('<!-- hidden -->visible')
'visible'
```

---

## Where it fits

```
  .txt  ──▶ load_text_file ──────────┐
                                     ├──▶ [ normalize_text ] ──▶ [ make_document ]
  .html ──▶ extract_text_from_html ──┘
```

This is the second entry point into the pipeline, parallel to the file loaders
rather than downstream of them. Both paths converge on plain text, and from
`normalize_text` onwards nothing knows or cares whether a document started as
markup — which is the point of putting the extraction here rather than letting
HTML-awareness spread into the chunker.
