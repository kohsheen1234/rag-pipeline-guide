"""Prompting and grounded answer generation.

Assembles retrieved chunks into a prompt, runs a local causal LM over it, and
returns the answer with the sources it was given. See docs/generation/ for the
reasoning behind each function.
"""

from rag_pipeline.embeddings import embed_text
from rag_pipeline.retrieval import cosine_similarity_search, top_k_chunks

__all__ = [
    "build_prompt_template",
    "format_context",
    "truncate_context",
    "add_system_instruction",
    "load_generator",
    "generate_answer",
    "rag_answer",
    "track_source_chunk_ids",
    "append_source_references",
]

SYSTEM_INSTRUCTION = (
    "You are a helpful assistant. Answer the question using ONLY the provided "
    "context. If the answer is not in the context, say 'I do not know'."
)

#: The one spelling of the refusal, so abstention checks downstream can match it.
REFUSAL = "I do not know"

DEFAULT_GENERATOR = "sshleifer/tiny-gpt2"


def build_prompt_template() -> str:
    """Return the RAG prompt template, with ``{context}`` and ``{question}`` slots.

    A plain string, not an f-string: the placeholders must survive until
    ``.format`` is called on them later.
    """
    return (
        "Context:\n"
        "{context}\n\n"
        "Question: {question}\n\n"
        "Answer the question using only the context above. "
        "If the context does not contain the answer, say '" + REFUSAL + "'.\n"
        "Answer:"
    )


def format_context(retrieved: list) -> str:
    """Render ``(chunk, score)`` tuples as one numbered line per chunk.

    Lines read ``[i] {text} (source={source})`` with *i* 1-based. Scores are
    left out -- they guide retrieval, but in the prompt they are noise.
    """
    return "\n".join(
        f"[{position}] {chunk['text']} (source={chunk['source']})"
        for position, (chunk, _) in enumerate(retrieved, start=1)
    )


def truncate_context(context: str, max_chars: int) -> str:
    """Cut a context block to ``max_chars``, preferring a whitespace boundary.

    Searches back from the limit for the last whitespace so words are not
    chopped in half, and falls back to a hard cut when the budget contains
    none.
    """
    if len(context) <= max_chars:
        return context

    # One past the limit, so a boundary sitting exactly at max_chars counts.
    window = context[: max_chars + 1]
    cut = max(window.rfind(character) for character in " \n\t")

    if cut == -1:
        return context[:max_chars]

    return context[:cut]


def add_system_instruction(prompt: str) -> str:
    """Prepend the grounding instruction, separated by a blank line."""
    return f"{SYSTEM_INSTRUCTION}\n\n{prompt}"


def load_generator(model_name: str = DEFAULT_GENERATOR):
    """Load a causal LM and its tokenizer as a ``(model, tokenizer)`` pair.

    GPT-style tokenizers often ship without a pad token, which breaks batched
    generation, so it is aliased to the end-of-sequence token when missing.
    """
    # Imported here so the rest of this module works without torch installed.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int = 64) -> str:
    """Generate a continuation for ``prompt``, with the prompt stripped off.

    Greedy decoding and a fixed seed, so the same prompt gives the same answer.
    The prompt is removed by slicing in *token* space -- decoding the whole
    output and stripping the prompt text back off is not reliable, because
    tokenisation does not round-trip character for character.
    """
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


def rag_answer(
    query: str,
    chunks: list,
    embeddings,
    embed_model,
    generator,
    tokenizer,
    k: int = 5,
    max_new_tokens: int = 64,
) -> dict:
    """Run the whole pipeline for one query: retrieve, prompt, generate.

    Returns ``{"answer", "sources", "query"}``. ``sources`` is the retrieved
    chunk dicts in ranked order, without their scores, so the answer can be
    audited against what was actually shown to the model.
    """
    query_vector = embed_text(embed_model, query)
    scores = cosine_similarity_search(query_vector, embeddings)
    retrieved = top_k_chunks(scores, chunks, k)

    prompt = build_prompt_template().format(
        context=format_context(retrieved),
        question=query,
    )
    answer = generate_answer(generator, tokenizer, prompt, max_new_tokens)

    return {
        "answer": answer,
        "sources": [chunk for chunk, _ in retrieved],
        "query": query,
    }


def track_source_chunk_ids(chunks: list) -> list:
    """Collect each chunk's ``id``, in retrieval order.

    A list rather than a set: order carries the ranking, and duplicates are
    meaningful when two retrieved windows came from the same chunk. Chunks
    without an ``id`` are skipped rather than raising.
    """
    return [chunk["id"] for chunk in chunks if "id" in chunk]


def append_source_references(answer: str, chunks: list) -> str:
    """Append a ``Sources: [id1, id2]`` line to an answer.

    Ids are rendered bare, not quoted, so the output does not change shape
    depending on whether they are strings or integers.
    """
    ids = track_source_chunk_ids(chunks)

    return f"{answer}\nSources: [{', '.join(str(chunk_id) for chunk_id in ids)}]"
