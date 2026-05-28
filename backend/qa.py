"""Ask My Docs — hybrid retrieval Q&A with citation enforcement."""

from __future__ import annotations

import logging
import os

from citations import (
    build_numbered_sources,
    enforce_citations,
    passages_to_legacy_sources,
)
from graph_builder import GROQ_MODEL, _get_groq_client
from retrieval import hybrid_retrieve

logger = logging.getLogger(__name__)

PRODUCT_NAME = os.getenv("ASK_MY_DOCS_NAME", "Ask My Docs")
DOMAIN_HINT = os.getenv(
    "ASK_MY_DOCS_DOMAIN",
    "the user's uploaded document (résumé, report, paper, manual, or similar)",
)


def _expand_query(question: str, history: list[dict]) -> str:
    """Include recent user context for vague follow-ups."""
    q = question.strip()
    if not history:
        return q
    if len(q.split()) > 4:
        return q
    prior = history[-1].get("question", "").strip()
    if prior:
        return f"{q} (context: {prior})"
    return q


def _build_history_section(history: list[dict]) -> str:
    if not history:
        return ""
    lines = ["## Previous conversation"]
    for turn in history[-4:]:
        lines.append(f"User: {turn.get('question', '')}")
        lines.append(f"Assistant: {turn.get('answer', '')}")
    lines.append(
        "Resolve pronouns using the conversation above, but cite only numbered sources below."
    )
    return "\n".join(lines)


def answer_question(
    graph: dict,
    question: str,
    chunks: list[dict] | None = None,
    history: list[dict] | None = None,
    retrieval_index: dict | None = None,
    *,
    offline: bool = False,
    skip_rerank: bool = False,
) -> dict:
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty")

    history = history or []
    if retrieval_index is None:
        raise ValueError("Retrieval index not built. Upload a PDF first.")

    passages = retrieval_index.get("passages") or []
    embeddings = retrieval_index.get("embeddings")
    if not len(passages):
        raise ValueError("No passages in retrieval index. Re-upload your document.")

    query = _expand_query(question, history)
    retrieved = hybrid_retrieve(
        query,
        passages,
        embeddings,
        offline=offline,
        skip_rerank=skip_rerank,
    )

    numbered_sources, sources_block = build_numbered_sources(retrieved)
    history_block = _build_history_section(history)

    context_parts = [
        f"## {PRODUCT_NAME}",
        f"Document scope: {DOMAIN_HINT}",
    ]
    if history_block:
        context_parts.append(history_block)
    context_parts.append(sources_block)

    context = "\n\n".join(context_parts)
    source_count = len(numbered_sources)

    client = _get_groq_client()
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are {PRODUCT_NAME}, a document Q&A assistant for {DOMAIN_HINT}. "
                        "Answer ONLY using the numbered sources provided. "
                        "Every factual claim MUST include at least one citation like [1] or [2]. "
                        "Prefer the highest-ranked sources when they conflict. "
                        "If sources are insufficient, say so in one sentence and do not cite. "
                        "Be direct and concise (1–4 sentences or short bullets). "
                        "Do not explain document type or your reasoning unless asked."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"{context}\n\n"
                        f"Question: {question}\n\n"
                        "Answer with mandatory [n] citations for each factual claim."
                    ),
                },
            ],
            temperature=0.2,
        )
    except Exception as exc:
        logger.error("Groq Q&A error: %s", exc)
        raise

    answer = (response.choices[0].message.content or "").strip()
    if not answer:
        answer = "I could not generate an answer from the available sources."

    answer, citation_check = enforce_citations(answer, source_count)
    legacy_sources = passages_to_legacy_sources(retrieved, numbered_sources)

    return {
        "answer": answer,
        "citations": numbered_sources,
        "citation_valid": citation_check["valid"],
        "sources": legacy_sources,
    }
