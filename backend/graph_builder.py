import logging
import os
import re
import uuid

from groq import Groq

from json_parser import parse_extraction_json

logger = logging.getLogger(__name__)

GROQ_MODEL = "llama-3.3-70b-versatile"
ENTITY_TYPES = {"PERSON", "ORG", "PRODUCT", "TECH", "CONCEPT", "LOCATION", "OTHER"}

EXTRACTION_PROMPT = """Extract entities and relationships from the text chunk below.

Return ONLY valid JSON in this exact format:
{
  "entities": [
    {"id": "unique_snake_case_id", "label": "Display Name", "type": "PERSON|ORG|PRODUCT|TECH|CONCEPT|LOCATION|OTHER"}
  ],
  "relationships": [
    {"source": "entity_id", "target": "entity_id", "type": "relationship_type", "evidence": "short quote from chunk"}
  ]
}

Rules:
- Use concise snake_case ids derived from labels.
- Only include entities clearly mentioned in the text.
- Relationship source/target must reference entity ids from this response.
- evidence must be a direct quote from the chunk (prefer full sentences or bullet points).
- Capture ALL meaningful relationships: who did what, what uses what, who authored what, etc.
- For lists or bullet points, create one relationship per bullet with the full bullet as evidence.
- Choose relationship types that fit the text (e.g. authored, uses, developed, cites, located_in, part_of).
- Output JSON only. No markdown, no commentary.

Text chunk:
"""


def _normalize_label(label: str) -> str:
    return re.sub(r"\s+", " ", label.strip().lower())


def _slugify(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return slug or f"entity_{uuid.uuid4().hex[:8]}"


def validate_groq_api_key() -> None:
    """Raise if GROQ_API_KEY is missing or still a placeholder."""
    from pathlib import Path
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

    api_key = (os.getenv("GROQ_API_KEY") or "").strip().strip('"').strip("'")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Add it to backend/.env and restart the server."
        )
    if api_key == "gsk_your-key-here" or len(api_key) < 20:
        raise ValueError(
            "GROQ_API_KEY in backend/.env still looks like a placeholder. "
            "Save your full gsk_... key and restart uvicorn."
        )


def _get_groq_client() -> Groq:
    validate_groq_api_key()
    api_key = (os.getenv("GROQ_API_KEY") or "").strip().strip('"').strip("'")
    return Groq(api_key=api_key)


def extract_from_chunk(client: Groq, chunk: str, chunk_index: int) -> dict:
    logger.info("Chunk %d: calling Groq model=%s (%d chars)", chunk_index, GROQ_MODEL, len(chunk))

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract knowledge graph data from text. "
                        "Respond with a single valid JSON object only. No markdown."
                    ),
                },
                {"role": "user", "content": EXTRACTION_PROMPT + chunk},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        err = str(exc).lower()
        logger.error("Chunk %d: Groq API error — %s", chunk_index, exc)
        if "invalid_api_key" in err or "401" in err or "invalid api key" in err:
            raise ValueError(
                "Invalid GROQ_API_KEY. Get a key at https://console.groq.com/keys "
                "and set it in backend/.env"
            ) from exc
        return {"entities": [], "relationships": [], "_failed": True}

    content = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)
    if usage:
        logger.info(
            "Chunk %d: tokens prompt=%s completion=%s",
            chunk_index,
            getattr(usage, "prompt_tokens", "?"),
            getattr(usage, "completion_tokens", "?"),
        )

    if len(content) > 300:
        logger.debug("Chunk %d raw preview: %s...", chunk_index, content[:300])
    else:
        logger.debug("Chunk %d raw response: %s", chunk_index, content)

    return parse_extraction_json(content, chunk_index=chunk_index)


def merge_and_build_graph(chunk_results: list[dict]) -> dict:
    """Merge entities by normalized label and produce React Flow-ready graph."""
    label_to_id: dict[str, str] = {}
    nodes_by_id: dict[str, dict] = {}
    id_aliases: dict[str, str] = {}
    edges: list[dict] = []
    seen_edge_keys: set[tuple[str, str, str]] = set()

    for result in chunk_results:
        entities = result.get("entities") or []
        relationships = result.get("relationships") or []

        chunk_id_map: dict[str, str] = {}

        for entity in entities:
            if not isinstance(entity, dict):
                continue
            raw_id = str(entity.get("id", "")).strip()
            label = str(entity.get("label", "")).strip()
            if not label:
                continue

            entity_type = str(entity.get("type", "OTHER")).upper()
            if entity_type not in ENTITY_TYPES:
                entity_type = "OTHER"

            norm = _normalize_label(label)
            if norm in label_to_id:
                canonical_id = label_to_id[norm]
            else:
                canonical_id = _slugify(label)
                base = canonical_id
                counter = 1
                while canonical_id in nodes_by_id and nodes_by_id[canonical_id]["label"] != label:
                    canonical_id = f"{base}_{counter}"
                    counter += 1
                label_to_id[norm] = canonical_id
                nodes_by_id[canonical_id] = {
                    "id": canonical_id,
                    "label": label,
                    "type": entity_type,
                }

            if raw_id:
                chunk_id_map[raw_id] = canonical_id
                id_aliases[raw_id] = canonical_id

        for rel in relationships:
            if not isinstance(rel, dict):
                continue
            source_raw = str(rel.get("source", "")).strip()
            target_raw = str(rel.get("target", "")).strip()
            rel_type = str(rel.get("type", "related_to")).strip() or "related_to"
            evidence = str(rel.get("evidence", "")).strip()

            source = chunk_id_map.get(source_raw) or id_aliases.get(source_raw) or source_raw
            target = chunk_id_map.get(target_raw) or id_aliases.get(target_raw) or target_raw

            if source not in nodes_by_id or target not in nodes_by_id:
                continue

            edge_key = (source, target, rel_type)
            if edge_key in seen_edge_keys:
                continue
            seen_edge_keys.add(edge_key)

            edges.append(
                {
                    "id": f"e_{uuid.uuid4().hex[:8]}",
                    "source": source,
                    "target": target,
                    "label": rel_type,
                    "evidence": evidence,
                }
            )

    logger.info("Merged graph: %d nodes, %d edges", len(nodes_by_id), len(edges))
    return {
        "nodes": list(nodes_by_id.values()),
        "edges": edges,
    }


def build_graph_from_chunks(chunks: list[str]) -> dict:
    validate_groq_api_key()
    client = _get_groq_client()
    results: list[dict] = []
    failed_chunks = 0
    total = len([c for c in chunks if c.strip()])

    logger.info("Starting extraction for %d non-empty chunks", total)

    chunk_num = 0
    for chunk in chunks:
        if not chunk.strip():
            continue
        chunk_num += 1
        extracted = extract_from_chunk(client, chunk, chunk_num)
        if extracted.get("_failed"):
            failed_chunks += 1
        results.append(extracted)

    if not results:
        logger.warning("No chunk results produced")
        raise ValueError("No text chunks could be processed.")

    graph = merge_and_build_graph(results)

    if not graph["nodes"] and not graph["edges"]:
        if failed_chunks == len(results):
            raise ValueError(
                "Graph extraction failed for all chunks. Check backend logs and your GROQ_API_KEY."
            )
        raise ValueError(
            "No entities or relationships were extracted. Try a longer or more text-rich PDF."
        )

    return graph
