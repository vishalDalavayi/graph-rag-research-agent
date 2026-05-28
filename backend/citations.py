"""Numbered source citations and enforcement."""

from __future__ import annotations

import re

CITATION_RE = re.compile(r"\[(\d+)\]")


def build_numbered_sources(passages: list[dict]) -> tuple[list[dict], str]:
    """Assign [1]..[N] to passages and build LLM context block."""
    sources: list[dict] = []
    lines = ["## Numbered sources (cite as [n] in your answer)"]

    for i, passage in enumerate(passages, start=1):
        entry = {
            "citation_id": i,
            "passage_id": passage.get("passage_id"),
            "source_type": passage.get("source_type"),
            "source_id": passage.get("source_id"),
            "page": passage.get("page"),
            "text": passage.get("text", ""),
        }
        if passage.get("source_type") == "graph_edge":
            entry.update(
                {
                    "graph_source": passage.get("graph_source"),
                    "graph_target": passage.get("graph_target"),
                    "graph_relationship": passage.get("graph_relationship"),
                    "graph_evidence": passage.get("graph_evidence"),
                }
            )
        sources.append(entry)
        page = f", page {entry['page']}" if entry.get("page") else ""
        lines.append(f"[{i}] ({entry['source_type']}{page}) {entry['text'][:1200]}")

    return sources, "\n".join(lines)


def extract_citation_ids(answer: str) -> list[int]:
    return [int(m) for m in CITATION_RE.findall(answer)]


def validate_citations(answer: str, source_count: int) -> dict:
    """Check that citations reference valid source numbers."""
    ids = extract_citation_ids(answer)
    valid = [i for i in ids if 1 <= i <= source_count]
    invalid = [i for i in ids if i < 1 or i > source_count]
    unique_valid = sorted(set(valid))

    return {
        "valid": len(invalid) == 0 and (source_count == 0 or len(unique_valid) > 0),
        "cited_ids": unique_valid,
        "invalid_ids": sorted(set(invalid)),
        "missing_citations": source_count > 0 and len(unique_valid) == 0,
    }


def enforce_citations(answer: str, source_count: int) -> tuple[str, dict]:
    """
    Validate citations; if invalid or missing, append a short compliance note.
    Returns (possibly adjusted answer, validation dict).
    """
    validation = validate_citations(answer, source_count)

    if source_count == 0:
        return answer, validation

    if validation["invalid_ids"]:
        answer = (
            f"{answer}\n\n"
            f"(Note: removed invalid citation markers {validation['invalid_ids']}; "
            f"only [1]–[{source_count}] are valid.)"
        )
        validation = validate_citations(answer, source_count)

    if validation["missing_citations"] and not answer.lower().startswith("i don't") and "insufficient" not in answer.lower():
        answer = (
            f"{answer}\n\n"
            "(Note: response lacked required [n] citations; claims should be tied to numbered sources.)"
        )
        validation["missing_citations"] = True
        validation["valid"] = False

    return answer, validation


def passages_to_legacy_sources(passages: list[dict], numbered: list[dict]) -> dict:
    """Map numbered passages back to graph_edges + chunks for API compatibility."""
    graph_edges = []
    chunks = []
    seen_chunk: set[str] = set()
    seen_edge: set[str] = set()

    for entry in numbered:
        if entry["source_type"] == "chunk" and entry["source_id"] not in seen_chunk:
            seen_chunk.add(entry["source_id"])
            text = entry["text"]
            if len(text) > 400:
                text = text[:400].rstrip() + "…"
            chunks.append(
                {
                    "chunk_id": entry["source_id"],
                    "page": entry.get("page"),
                    "text": text,
                    "citation_id": entry["citation_id"],
                }
            )
        elif entry["source_type"] == "graph_edge" and entry["source_id"] not in seen_edge:
            seen_edge.add(entry["source_id"])
            graph_edges.append(
                {
                    "edge_id": entry["source_id"],
                    "source": entry.get("graph_source", ""),
                    "target": entry.get("graph_target", ""),
                    "relationship": entry.get("graph_relationship", ""),
                    "evidence": entry.get("graph_evidence", entry.get("text", "")),
                    "citation_id": entry["citation_id"],
                }
            )

    return {"graph_edges": graph_edges, "chunks": chunks}
