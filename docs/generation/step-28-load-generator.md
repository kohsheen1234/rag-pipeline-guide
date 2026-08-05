# Step 28 · `load_generator`

> **Part 5 · Prompting and Answer Generation** — step 28 of 51
> Code: [`rag_pipeline/generation.py`](../../rag_pipeline/generation.py) · Tests: [`tests/test_generation.py`](../../tests/test_generation.py)
> Previous: [Step 27 · `add_system_instruction`](step-27-add-system-instruction.md) · Next: [Step 29 · `generate_answer`](step-29-generate-answer.md)

---

## The task

```python
def load_generator(model_name: str = "sshleifer/tiny-gpt2"): ...
```

Load a small local causal language model and its tokenizer, returning a
`(model, tokenizer)` tuple. If `tokenizer.pad_token` is missing, set it to the
end-of-sequence token. Default to a tiny model so it runs quickly on CPU.

---

## Why this step exists

Same argument as
[`load_embedding_model`](../embeddings/step-11-load-embedding-model.md): loading
is expensive, the instance is reusable, and the model and tokenizer must be a
matched pair. A tokenizer from one model produces token ids that mean something
else to another — not an error, just wrong text.

Returning them together makes them hard to mismatch.

---

## What's happening

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

return model, tokenizer
```

**`AutoTokenizer` / `AutoModelForCausalLM`** read the model's config and
instantiate the right concrete classes, so `sshleifer/tiny-gpt2` comes back as a
`GPT2LMHeadModel` without naming it.

*Causal* means decoder-only: each position attends only to positions before it,
and the head predicts the next token. That is what makes generation a loop of
"predict, append, repeat".

### The pad token

The interesting line, and a genuine piece of Hugging Face lore.

GPT-2 was trained on a continuous stream of text with no padding, so its
tokenizer ships with no pad token at all. As long as you generate one sequence
at a time this never comes up. The moment you batch — different prompt lengths,
shorter ones padded to match — the tokenizer has nothing to pad *with*, and
`generate` either raises or emits a warning and behaves unpredictably.

Aliasing pad to EOS is the standard fix. It works because the attention mask
marks padded positions as ignored, so which id sits there does not matter; it
only needs to be a valid id. The two roles are distinguished by the mask, not by
the token.

Setting it here rather than at the call site means every consumer gets a
tokenizer that is already safe to batch with, and
[`generate_answer`](step-29-generate-answer.md) can pass `pad_token_id`
unconditionally.

**The lazy import** keeps `transformers` optional, as elsewhere in this repo.

---

## Boundaries of the contract

**The default model is a test fixture, not a useful one.**
`sshleifer/tiny-gpt2` has random-ish weights and a handful of layers. It runs
fast and produces gibberish. It is the right default for a test suite and the
wrong one for an answer.

**Downloads on first use.** Cached under `~/.cache/huggingface` afterwards.
Offline environments need the cache pre-warmed.

**CPU by default.** No `device_map`, no quantisation, no `torch_dtype`. Real
models want at least `torch_dtype=torch.float16` on a GPU.

**No caching between calls.** Two calls load two models.

**Not tied to the embedding model.** The generator and the retriever are
independent, which is correct — but it means nothing records which pair produced
a given answer.

**No `eval()` or `no_grad()`.** `from_pretrained` returns a model in eval mode
already, and `generate` runs under `no_grad` internally, so this is fine in
practice rather than by explicit statement.

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| Not setting a pad token | Batched `generate` raises or misbehaves. Single-sequence use hides it. |
| Mixing a tokenizer and model from different checkpoints | Ids decode to the wrong text. No error. |
| Loading inside a request handler | Seconds per call instead of milliseconds. |
| Shipping the tiny default to production | It generates noise. |
| Assuming a pad token means padding is free | You still need the attention mask, which `tokenizer(...)` provides. |

---

## Example

```python
>>> model, tokenizer = load_generator('sshleifer/tiny-gpt2')
>>> tokenizer.pad_token == tokenizer.eos_token
True
>>> type(model).__name__
'GPT2LMHeadModel'
```

> The test asserting this skips unless `transformers` is installed. Everything
> downstream is tested against a small stand-in with the same surface.

---

## Where it fits

```
  model_name ──▶ [ load_generator ] ──▶ (model, tokenizer) ──▶ [ generate_answer ]
                                                │
                                                └── pad_token aliased to eos_token
```
