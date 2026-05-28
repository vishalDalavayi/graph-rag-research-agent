"""Build unified retrieval passages from chunks and graph edges."""

from __future__ import annotations


def build_passages(chunks: list[dict], graph: dict) -> list[dict]:
    """Passages are the atomic units for BM25, vector search, and reranking."""
    passages: list[dict] = []
    seen_text: set[str] = set()

    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        text = str(chunk.get("text", "")).strip()
        if len(text) < 20:
            continue
        norm = text[:200]
        if norm in seen_text:
            continue
        seen_text.add(norm)
        passages.append(
            {
                "passage_id": f"passage_{len(passages):04d}",
                "source_type": "chunk",
                "source_id": str(chunk.get("id", "")),
                "page": chunk.get("page") if isinstance(chunk.get("page"), int) else None,
                "text": text,
            }
        )

    node_by_id = {
        n["id"]: n
        for n in graph.get("nodes", [])
        if isinstance(n, dict) and n.get("id")
    }
    for edge in graph.get("edges", []) or []:
        if not isinstance(edge, dict):
            continue
        evidence = str(edge.get("evidence", "")).strip()
        if len(evidence) < 15:
            continue
        src = node_by_id.get(edge.get("source"), {}).get("label", edge.get("source", ""))
        tgt = node_by_id.get(edge.get("target"), {}).get("label", edge.get("target", ""))
        rel = str(edge.get("label", "related_to"))
        text = f"{src} --[{rel}]--> {tgt}. Evidence: {evidence}"
        norm = evidence[:200]
        if norm in seen_text:
            continue
        seen_text.add(norm)
        passages.append(
            {
                "passage_id": f"passage_{len(passages):04d}",
                "source_type": "graph_edge",
                "source_id": str(edge.get("id", "")),
                "page": None,
                "text": text,
                "graph_source": src,
                "graph_target": tgt,
                "graph_relationship": rel,
                "graph_evidence": evidence,
            }
        )

    return passages
