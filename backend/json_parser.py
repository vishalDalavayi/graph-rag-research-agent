import json
import logging
import re

logger = logging.getLogger(__name__)

EMPTY_EXTRACTION: dict = {"entities": [], "relationships": []}


def _try_parse(text: str) -> dict | None:
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        return None
    return None


def _strip_code_fence(text: str) -> str:
    """Remove markdown ```json ... ``` wrappers if present."""
    fenced = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return text


def _extract_json_object(text: str) -> str | None:
    """Find the first balanced {...} block in text."""
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _normalize_extraction(data: dict) -> dict:
    entities = data.get("entities")
    relationships = data.get("relationships")

    if not isinstance(entities, list):
        entities = []
    if not isinstance(relationships, list):
        relationships = []

    return {"entities": entities, "relationships": relationships}


def parse_extraction_json(raw: str, chunk_index: int = 0) -> dict:
    """
    Parse model output into {entities, relationships}.
    Falls back to empty extraction on malformed JSON.
    """
    if not raw or not raw.strip():
        logger.warning("Chunk %d: empty model response — using empty extraction", chunk_index)
        return dict(EMPTY_EXTRACTION)

    text = raw.strip()

    for attempt, candidate in enumerate(
        [
            ("direct", text),
            ("code_fence", _strip_code_fence(text)),
            ("json_object", _extract_json_object(text) or ""),
        ],
        start=1,
    ):
        name, payload = candidate
        if not payload:
            continue
        parsed = _try_parse(payload)
        if parsed is not None:
            result = _normalize_extraction(parsed)
            logger.info(
                "Chunk %d: parsed JSON via %s (%d entities, %d relationships)",
                chunk_index,
                name,
                len(result["entities"]),
                len(result["relationships"]),
            )
            return result

    preview = text[:250].replace("\n", " ")
    logger.warning(
        "Chunk %d: could not parse JSON — using empty extraction. Preview: %s",
        chunk_index,
        preview,
    )
    return dict(EMPTY_EXTRACTION)
