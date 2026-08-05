# Step 49 · `cache_query_embedding`

> **Part 8 · Robustness, Caching, and Chat Memory** — step 49 of 51
> Code: [`rag_pipeline/robustness.py`](../../rag_pipeline/robustness.py) · Tests: [`tests/test_robustness.py`](../../tests/test_robustness.py)
> Previous: [Step 48 · `deduplicate_chunks`](step-48-deduplicate-chunks.md) · Next: [Step 50 · `update_chat_memory`](step-50-update-chat-memory.md)

---

## The task

```python
def cache_query_embedding(query: str, model, cache: dict) -> np.ndarray: ...
```

Return the embedding of a query, memoising it in a shared dict. On a hit, return
the stored vector without invoking the model. On a miss, compute via
[`embed_text`](../embeddings/step-12-embed-text.md), store, and return. The
cache is updated in place.

---

## Why this step exists

Embedding a query is a transformer forward pass — milliseconds, and by far the
slowest thing in the retrieval path once the corpus is precomputed. Looking up a
dict is nanoseconds.

That trade only pays if queries repeat, and in a chat loop they do more than you
would expect: users rephrase and resend, clients retry, a UI re-runs the last
query on reconnect, and evaluation harnesses run the same fixed question set on
every iteration. That last one is the case where this matters most in this
repo — running Part 7's eval set fifty times while tuning a chunker embeds the
same three questions fifty times.

The reason it is *safe* is that a query embedding is a pure function of the
string and the model weights. The weights do not change within a process, so the
string alone is a sound key.

---

## What's happening

```python
if query in cache:
    return cache[query]

vector = embed_text(model, query)
cache[query] = vector

return vector
```

Textbook memoisation. Three details are load-bearing.

**Mutated in place, never rebound.** `cache[query] = vector` writes through to
the caller's dict. Writing `cache = {**cache, query: vector}` would build a new
dict, populate it, and discard it at the return — the caller's dict would stay
empty and every call would be a miss. Silent, and it would look like the cache
simply was not helping.

**The model is not touched on a hit.** The early return happens before
`embed_text` is reached. This is the entire point, and it is worth asserting
rather than assuming: the tests use a model that counts its calls and check the
count stays at 1 across two invocations. A version that computes first and then
checks the cache would pass every value-based test while providing no speedup at
all.

**The stored array is returned directly**, not copied. So callers share one
array per query, and a caller mutating it in place corrupts the cache for
everyone. In practice nothing here mutates embeddings, and the copy would cost
more than it saves — but it is a real sharp edge if you start normalising in
place.

---

## Keys are raw strings

No normalisation of the key. `"Hi"` and `"hi"` are separate entries, as are
`"what is rag"` and `"what is rag "`. A test pins this.

That is the specified behaviour and it is defensible: the function caches what
it is given, and the caller decides what counts as the same query. If you want
case-insensitive hits, run
[`query_rewrite`](../advanced-retrieval/step-33-query-rewrite.md) *before*
caching, which lowercases and collapses whitespace — and then the cache key is
the same string you are actually embedding, which is exactly right.

Normalising inside would be worse: the cache would return a vector for a
different string than the one requested, which is surprising in a function whose
contract is "the embedding of this query".

---

## Boundaries of the contract

**Unbounded.** Nothing evicts. A long-running service with varied queries grows
this dict until the process dies. An LRU with a size cap —
`functools.lru_cache`, or an explicit `OrderedDict` — is the fix, and this
signature cannot express it because the cache is a plain dict owned by the
caller.

**Not keyed on the model.** Two models sharing one cache silently return each
other's vectors. If you might swap models, key on `(model_name, query)` or use
one cache per model. This is the same class of failure as
[the one-model invariant](../embeddings/step-11-load-embedding-model.md#why-this-step-exists),
and here the cache makes it easier to hit.

**Not thread-safe** in a meaningful sense. Individual dict operations are atomic
under the GIL so nothing corrupts, but two threads can both miss and both
compute — wasteful, not wrong.

**Caller-owned lifetime.** Passing a fresh `{}` each call disables it entirely,
which is easy to do by accident.

**Query embeddings only.** Chunk embeddings are already precomputed by
[`embed_chunks`](../embeddings/step-13-embed-chunks.md).

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| Rebinding `cache` instead of mutating | The caller's dict never fills; every call misses, silently. |
| Computing before checking the cache | Passes value tests, delivers no speedup. |
| Sharing one cache across models | Returns the wrong model's vectors. No error. |
| Passing a new `{}` each call | The cache is disabled and looks like it is working. |
| Letting it grow forever | Unbounded memory in a long-running service. |
| Assuming the returned array is private | It is shared; mutating it corrupts the cache. |

---

## Example

```python
>>> cache = {}
>>> v = cache_query_embedding('hi', model, cache)
>>> v.shape
(2,)
>>> 'hi' in cache
True
>>> v2 = cache_query_embedding('hi', model, cache)   # cache hit, model untouched
>>> v2 is v
True
```

---

## Where it fits

```
  query ──▶ [ cache_query_embedding ] ──┬── hit ──▶ stored vector
                       │                └── miss ─▶ [ embed_text ] ──▶ store
                       ▼
              retrieval / scoring
```
