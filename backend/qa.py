import logging
import re

from graph_builder import GROQ_MODEL, _get_groq_client

logger = logging.getLogger(__name__)

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "to", "of",
    "in", "for", "on", "with", "at", "by", "from", "as", "into", "about",
    "what", "which", "who", "whom", "this", "that", "these", "those", "how",
    "why", "when", "where", "tell", "explain", "describe", "give", "show",
    "they", "them", "their", "you", "your", "he", "she", "his", "her", "its",
    "also", "just", "very", "more", "some", "any", "all", "and", "but", "or",
}

MAX_NODES = 35
MAX_EDGES = 45
MAX_CHUNKS = 8
CHUNK_SOURCE_EXCERPT = 400

FOLLOWUP_PHRASES = (
    "more specifically", "tell me more", "what about", "elaborate", "go on",
    "more detail", "expand on", "and what", "anything else",
)

OVERVIEW_PHRASES = (
    "what is this", "what is the document", "what is this pdf", "what is this paper",
    "summarize", "summary", "overview", "what is it about", "tell me about this document",
)


def _extract_keywords(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        if len(t) > 2 and t not in STOPWORDS and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _score_text(text: str, keywords: list[str]) -> int:
    if not text or not keywords:
        return 0
    lower = text.lower()
    return sum(1 for kw in keywords if kw in lower)


def _node_map(graph: dict) -> dict:
    return {n["id"]: n for n in graph.get("nodes", []) if isinstance(n, dict) and n.get("id")}


def _is_followup(question: str, history: list[dict]) -> bool:
    if not history:
        return False
    if any(p in question.lower() for p in FOLLOWUP_PHRASES):
        return True
    return len(_extract_keywords(question)) <= 2


def _is_overview_question(question: str) -> bool:
    q = question.lower()
    return any(p in q for p in OVERVIEW_PHRASES)


def _search_keywords(question: str, history: list[dict]) -> list[str]:
    """Keywords from the current question; for vague follow-ups, also pull from recent user turns."""
    keywords = _extract_keywords(question)
    q_lower = question.lower()
    vague_followup = len(keywords) <= 1 or any(p in q_lower for p in FOLLOWUP_PHRASES)
    if history and vague_followup:
        for turn in history[-2:]:
            for kw in _extract_keywords(turn.get("question", "")):
                if kw not in keywords:
                    keywords.append(kw)
    return keywords


def _overview_sample(graph: dict) -> tuple[list[dict], list[dict]]:
    """Diverse slice of the graph for document-level questions."""
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []

    by_type: dict[str, list[dict]] = {}
    for n in nodes:
        by_type.setdefault(str(n.get("type", "OTHER")), []).append(n)

    sampled: list[dict] = []
    for type_nodes in by_type.values():
        sampled.extend(type_nodes[:8])
    sampled = sampled[:MAX_NODES]

    sampled_ids = {n["id"] for n in sampled}
    scored_edges: list[tuple[int, dict]] = []
    for e in edges:
        score = len(str(e.get("evidence", "")))
        if e.get("source") in sampled_ids or e.get("target") in sampled_ids:
            score += 50
        scored_edges.append((score, e))
    scored_edges.sort(key=lambda x: x[0], reverse=True)
    matched_edges = [e for _, e in scored_edges[:MAX_EDGES]]

    logger.info("Q&A overview: %d nodes, %d edges", len(sampled), len(matched_edges))
    return sampled, matched_edges


def _retrieve_context(
    graph: dict, question: str, history: list[dict]
) -> tuple[list[dict], list[dict]]:
    """
    Document-agnostic retrieval: score nodes/edges by keyword overlap,
    expand around top matches, include conversation context for follow-ups.
    """
    keywords = _search_keywords(question, history)
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    node_by_id = _node_map(graph)

    if _is_overview_question(question) and not _is_followup(question, history):
        return _overview_sample(graph)

    if not keywords and history:
        keywords = _extract_keywords(history[-1].get("question", ""))

    logger.info("Q&A keywords: %s", keywords or "(none)")

    scored_nodes: list[tuple[int, dict]] = []
    for node in nodes:
        text = f"{node.get('label', '')} {node.get('id', '')} {node.get('type', '')}"
        score = _score_text(text, keywords)
        if score > 0:
            scored_nodes.append((score, node))

    scored_edges: list[tuple[int, dict]] = []
    for edge in edges:
        src = node_by_id.get(edge.get("source"), {}).get("label", "")
        tgt = node_by_id.get(edge.get("target"), {}).get("label", "")
        text = f"{src} {tgt} {edge.get('label', '')} {edge.get('evidence', '')}"
        score = _score_text(text, keywords)
        if len(str(edge.get("evidence", ""))) > 40:
            score += 1
        if score > 0:
            scored_edges.append((score, edge))

    scored_nodes.sort(key=lambda x: x[0], reverse=True)
    scored_edges.sort(key=lambda x: x[0], reverse=True)

    matched_nodes = [n for _, n in scored_nodes[:MAX_NODES]]
    matched_edges = [e for _, e in scored_edges[:MAX_EDGES]]
    matched_node_ids = {n["id"] for n in matched_nodes}

    # Expand: include all edges touching top-scoring nodes
    for node in matched_nodes[:10]:
        for e in edges:
            if e.get("source") == node["id"] or e.get("target") == node["id"]:
                if e not in matched_edges and len(matched_edges) < MAX_EDGES:
                    matched_edges.append(e)
                other = e["target"] if e["source"] == node["id"] else e["source"]
                if other not in matched_node_ids and other in node_by_id:
                    matched_nodes.append(node_by_id[other])
                    matched_node_ids.add(other)

    # Fallback: substantive edges (long evidence) if keyword match is thin
    if len(matched_edges) < 5 and edges:
        by_evidence = sorted(
            edges,
            key=lambda e: len(str(e.get("evidence", ""))),
            reverse=True,
        )
        for e in by_evidence:
            if e not in matched_edges:
                matched_edges.append(e)
            if len(matched_edges) >= MAX_EDGES:
                break
        for e in matched_edges:
            for nid in (e.get("source"), e.get("target")):
                if nid and nid not in matched_node_ids and nid in node_by_id:
                    matched_nodes.append(node_by_id[nid])
                    matched_node_ids.add(nid)

    logger.info("Q&A retrieved: %d nodes, %d edges", len(matched_nodes), len(matched_edges))
    return matched_nodes[:MAX_NODES], matched_edges[:MAX_EDGES]


def _retrieve_chunks(
    chunks: list[dict],
    question: str,
    history: list[dict],
    keywords: list[str] | None = None,
) -> list[dict]:
    """Retrieve raw text chunks by keyword overlap with chunk text."""
    if not chunks:
        return []

    keywords = keywords if keywords is not None else _search_keywords(question, history)
    if not keywords and history:
        keywords = _extract_keywords(history[-1].get("question", ""))

    scored: list[tuple[int, dict]] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        text = str(chunk.get("text", ""))
        score = _score_text(text, keywords)
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    matched = [c for _, c in scored[:MAX_CHUNKS]]

    if not matched and chunks:
        # Fallback: longest chunks when keyword match is thin
        matched = sorted(
            [c for c in chunks if isinstance(c, dict) and str(c.get("text", "")).strip()],
            key=lambda c: len(str(c.get("text", ""))),
            reverse=True,
        )[:MAX_CHUNKS]

    logger.info("Q&A retrieved: %d chunks", len(matched))
    return matched


def _build_graph_summary(graph: dict) -> str:
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    by_type: dict[str, list[str]] = {}
    for n in nodes:
        by_type.setdefault(str(n.get("type", "OTHER")), []).append(str(n.get("label", "")))
    lines = [
        "## Graph summary",
        f"Entities: {len(nodes)}, relationships: {len(edges)}",
    ]
    for t in sorted(by_type.keys()):
        labels = by_type[t][:12]
        extra = f" (+{len(by_type[t]) - 12} more)" if len(by_type[t]) > 12 else ""
        lines.append(f"- {t}: {', '.join(labels)}{extra}")
    return "\n".join(lines)


def _build_history_section(history: list[dict]) -> str:
    if not history:
        return ""
    lines = ["## Previous conversation"]
    for turn in history[-4:]:
        lines.append(f"User: {turn.get('question', '')}")
        lines.append(f"Assistant: {turn.get('answer', '')}")
    lines.append(
        "Resolve pronouns and follow-ups using the conversation above, "
        "but answer ONLY from graph evidence below — not from prior assistant guesses."
    )
    return "\n".join(lines)


def _build_evidence_section(edges: list[dict]) -> str:
    quotes: list[str] = []
    seen: set[str] = set()
    for e in edges:
        ev = str(e.get("evidence", "")).strip()
        if len(ev) >= 15 and ev not in seen:
            seen.add(ev)
            quotes.append(ev)
    if not quotes:
        return ""
    lines = ["## Source quotes (cite these when answering)"]
    for i, q in enumerate(quotes[:25], 1):
        lines.append(f'{i}. "{q}"')
    return "\n".join(lines)


def _build_chunks_section(chunks: list[dict]) -> str:
    if not chunks:
        return ""
    lines = ["## Raw text chunks (use to fill gaps not covered by graph evidence)"]
    for chunk in chunks:
        chunk_id = str(chunk.get("id", ""))
        page = chunk.get("page")
        page_label = f"page {page}" if page is not None else "page unknown"
        text = str(chunk.get("text", "")).strip()
        lines.append(f"- [{chunk_id}, {page_label}] {text}")
    return "\n".join(lines)


def _build_context(
    graph: dict,
    nodes: list[dict],
    edges: list[dict],
    chunks: list[dict],
    history: list[dict],
) -> str:
    node_by_id = _node_map(graph)
    parts = [_build_graph_summary(graph)]

    history_section = _build_history_section(history)
    if history_section:
        parts.append(history_section)

    evidence_section = _build_evidence_section(edges)
    if evidence_section:
        parts.append(evidence_section)

    chunks_section = _build_chunks_section(chunks)
    if chunks_section:
        parts.append(chunks_section)

    entity_lines = ["\n## Matched entities"]
    if nodes:
        for n in nodes:
            entity_lines.append(f"- [{n.get('type', 'OTHER')}] {n.get('label', n.get('id'))}")
    else:
        entity_lines.append("- (none)")

    rel_lines = ["\n## Matched relationships"]
    if edges:
        for e in edges:
            src = node_by_id.get(e.get("source"), {}).get("label", e.get("source"))
            tgt = node_by_id.get(e.get("target"), {}).get("label", e.get("target"))
            rel_lines.append(f"- {src} --[{e.get('label', 'related_to')}]--> {tgt}")
            ev = e.get("evidence", "")
            if ev:
                rel_lines.append(f'  Evidence: "{ev}"')
    else:
        rel_lines.append("- (none)")

    return "\n".join(parts + entity_lines + rel_lines)


def _edges_to_sources(edges: list[dict], node_by_id: dict) -> list[dict]:
    return [
        {
            "edge_id": str(e.get("id", "")),
            "source": node_by_id.get(e.get("source"), {}).get("label", e.get("source")),
            "target": node_by_id.get(e.get("target"), {}).get("label", e.get("target")),
            "relationship": str(e.get("label", "")),
            "evidence": str(e.get("evidence", "")),
        }
        for e in edges
    ]


def _chunks_to_sources(chunks: list[dict]) -> list[dict]:
    sources: list[dict] = []
    for chunk in chunks:
        text = str(chunk.get("text", "")).strip()
        if len(text) > CHUNK_SOURCE_EXCERPT:
            text = text[:CHUNK_SOURCE_EXCERPT].rstrip() + "…"
        sources.append(
            {
                "chunk_id": str(chunk.get("id", "")),
                "page": chunk.get("page") if isinstance(chunk.get("page"), int) else None,
                "text": text,
            }
        )
    return sources


def answer_question(
    graph: dict,
    question: str,
    chunks: list[dict] | None = None,
    history: list[dict] | None = None,
) -> dict:
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty")

    history = history or []
    chunks = chunks or []
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    if not nodes and not edges and not chunks:
        raise ValueError("Graph is empty. Upload a PDF first.")

    keywords = _search_keywords(question, history)
    matched_nodes, matched_edges = _retrieve_context(graph, question, history)
    matched_chunks = _retrieve_chunks(chunks, question, history, keywords=keywords)
    context = _build_context(graph, matched_nodes, matched_edges, matched_chunks, history)
    graph_sources = _edges_to_sources(matched_edges, _node_map(graph))
    chunk_sources = _chunks_to_sources(matched_chunks)

    logger.info(
        "Q&A context=%d chars, %d graph sources, %d chunk sources, history=%d",
        len(context),
        len(graph_sources),
        len(chunk_sources),
        len(history),
    )

    client = _get_groq_client()
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You answer questions using a hybrid knowledge graph and raw text chunks "
                        "extracted from an uploaded PDF. "
                        "Use ONLY the provided graph entities, relationships, evidence quotes, and raw text chunks. "
                        "Prefer graph evidence when it directly answers the question; use raw chunks to fill gaps. "
                        "For follow-up questions, resolve pronouns from Previous conversation, "
                        "but base factual claims on graph/chunk evidence — not prior assistant guesses. "
                        "If context is insufficient, say so briefly. Do not invent or speculate.\n\n"
                        "Style rules:\n"
                        "- Be direct and concise. Default to 1–3 short sentences.\n"
                        "- Use bullet points only when listing multiple distinct items.\n"
                        "- Do NOT explain your reasoning, document-type inference, or why you believe something.\n"
                        "- Do NOT say things like 'this appears to be a résumé because…' unless the user asked what type of document it is.\n"
                        "- Just answer the question with the facts from the context."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Hybrid context (graph + raw chunks):\n\n{context}\n\n"
                        f"Question: {question}\n\n"
                        "Answer using only the context above."
                    ),
                },
            ],
            temperature=0.3,
        )
    except Exception as exc:
        logger.error("Groq Q&A error: %s", exc)
        raise

    answer = (response.choices[0].message.content or "").strip()
    if not answer:
        answer = "I could not generate an answer from the available graph context."

    return {
        "answer": answer,
        "sources": {
            "graph_edges": graph_sources,
            "chunks": chunk_sources,
        },
    }
