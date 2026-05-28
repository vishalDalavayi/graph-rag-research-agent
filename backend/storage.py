import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
GRAPH_PATH = DATA_DIR / "graph.json"
CHUNKS_PATH = DATA_DIR / "chunks.json"


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def save_graph(graph: dict) -> None:
    ensure_data_dir()
    with open(GRAPH_PATH, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)
    logger.info("Saved graph to %s", GRAPH_PATH)


def load_graph() -> dict | None:
    if not GRAPH_PATH.exists():
        logger.debug("No graph file at %s", GRAPH_PATH)
        return None
    with open(GRAPH_PATH, encoding="utf-8") as f:
        graph = json.load(f)
    logger.debug("Loaded graph from %s", GRAPH_PATH)
    return graph


def save_chunks(chunks: list[dict]) -> None:
    ensure_data_dir()
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d chunks to %s", len(chunks), CHUNKS_PATH)


def load_chunks() -> list[dict] | None:
    if not CHUNKS_PATH.exists():
        logger.debug("No chunks file at %s", CHUNKS_PATH)
        return None
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)
    if not isinstance(chunks, list):
        logger.warning("Invalid chunks file format at %s", CHUNKS_PATH)
        return None
    logger.debug("Loaded %d chunks from %s", len(chunks), CHUNKS_PATH)
    return chunks


def clear_document_data() -> None:
    """Delete persisted graph, chunks, and retrieval index."""
    for path in (GRAPH_PATH, CHUNKS_PATH):
        if path.exists():
            path.unlink()
            logger.info("Removed %s", path)
    from retrieval_index import EMBEDDINGS_PATH, PASSAGES_PATH

    for path in (PASSAGES_PATH, EMBEDDINGS_PATH):
        if path.exists():
            path.unlink()
            logger.info("Removed %s", path)
