# Step 29 · `generate_answer`

> **Part 5 · Prompting and Answer Generation** — step 29 of 51
> Code: [`rag_pipeline/generation.py`](../../rag_pipeline/generation.py) · Tests: [`tests/test_generation.py`](../../tests/test_generation.py)
> Previous: [Step 28 · `load_generator`](step-28-load-generator.md) · Next: [Step 30 · `rag_answer`](step-30-rag-answer.md)

---

## The task

```python
def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int = 64) -> str: ...
```

Return the model's freshly generated continuation as a decoded string with the
prompt stripped off. Use greedy decoding, seed torch for determinism, skip
special tokens when decoding, and pass the pad token id to `generate`.

---

## Why this step exists

This is the generation half of RAG. Everything upstream — ingestion, chunking,
embedding, retrieval, prompt assembly — exists to produce the string this
function is handed.

---

## What's happening

```python
import torch

torch.manual_seed(0)

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_new_tokens=max_new_tokens,
    do_sample=False,
    pad_token_id=tokenizer.pad_token_id,
)
generated = outputs[0][inputs["input_ids"].shape[1] :]

return tokenizer.decode(generated, skip_special_tokens=True)
```

### Slicing in token space

The most important line is `outputs[0][inputs["input_ids"].shape[1]:]`.

`generate` returns the prompt *and* the continuation: shape
`(1, T_in + T_new)`. To get only the answer you have to remove the prompt, and
there are two ways to try.

The tempting one is to decode everything and then strip the prompt text back
off — `full.replace(prompt, "")` or `full[len(prompt):]`. It does not reliably
work, because **tokenisation is not invertible character-for-character**. Encode
a string and decode it and you can get back something subtly different:
whitespace normalised at token boundaries, a leading space added or dropped. The
decoded prefix then does not match your original prompt string, `replace` finds
nothing, and you return the prompt concatenated with the answer.

Slicing in token space sidesteps it entirely. `inputs["input_ids"].shape[1]` is
exactly how many tokens went in, so everything after that index is new by
construction. No string comparison, nothing to fail to match.

### Determinism

`do_sample=False` selects greedy decoding: take the argmax at each step rather
than sampling from the distribution. Same prompt, same output, every time.

`torch.manual_seed(0)` is then belt-and-braces — greedy decoding does not
consume randomness, so the seed changes nothing for this configuration. It is
there so that turning on sampling later does not silently make the function
non-reproducible, and the tests can assert equality across calls regardless.

Reproducibility matters here more than output quality: an evaluation harness
that returns different answers per run cannot attribute a metric change to a
code change.

### The remaining two arguments

`pad_token_id=tokenizer.pad_token_id` silences the "Setting `pad_token_id` to
`eos_token_id`" warning that `generate` emits otherwise. It is only meaningful
because [step 28](step-28-load-generator.md) guaranteed the attribute exists.

`skip_special_tokens=True` drops `<|endoftext|>` and friends from the decoded
string, so the answer does not end in markup.

---

## Boundaries of the contract

**Greedy is not the best decoder.** It is repetitive and can loop. Beam search
or nucleus sampling generally read better; both cost determinism, and sampling
needs the seed to mean something.

**`max_new_tokens` is a hard cut, not a sentence boundary.** A long answer stops
mid-word. Nothing trims to the last complete sentence.

**No length check on the prompt.** A prompt longer than the model's window is
truncated by the tokenizer, usually from the left — quietly removing your system
instruction while keeping the least relevant retrieved chunk. See
[`truncate_context`](step-26-truncate-context.md).

**One prompt at a time**, despite the pad-token work making batching possible.

**No stop sequences.** The model may run past the answer into a hallucinated
next question.

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| Decoding the full output and string-stripping the prompt | Tokenisation is not character-invertible; the strip silently fails and you return prompt + answer. |
| `do_sample=True` in an eval harness | Different answers per run; metric deltas become unattributable. |
| Omitting `pad_token_id` | A warning per call, and undefined behaviour when batching. |
| `skip_special_tokens=False` | `<|endoftext|>` ends up in the answer. |
| Confusing `max_new_tokens` with `max_length` | `max_length` counts the prompt too, so a long prompt leaves no room to generate. |
| Assuming the answer is complete | It stops at the budget, mid-sentence if necessary. |

---

## Example

```python
>>> model, tokenizer = load_generator()
>>> ans = generate_answer(model, tokenizer, 'Hello', max_new_tokens=4)
>>> isinstance(ans, str)
True
>>> generate_answer(model, tokenizer, 'Hello', max_new_tokens=4) == ans
True
```

---

## Where it fits

```
  prompt ──▶ tokenizer ──▶ (1, T_in) ──▶ model.generate ──▶ (1, T_in + T_new)
                                                                    │
                                              slice off T_in  ──────┤
                                                                    ▼
                                                    decode ──▶ answer string
```
