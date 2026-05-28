"""Persist and load retrieval index (passages + embeddings)."""

from __future__ import annotations

import json
import logging

import numpy as np

from embeddings import embed_texts
from passages import build_passages
from storage import DATA_DIR, ensure_data_dir

logger = logging.getLogger(__name__)

PASSAGES_PATH = DATA_DIR / "passages.json"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npz"


def build_retrieval_index(chunks: list[dict], graph: dict, *, offline: bool = False) -> dict:
    passages = build_passages(chunks, graph)
    texts = [p["text"] for p in passages]
    matrix = embed_texts(texts, offline=offline)
    save_retrieval_index(passages, matrix)
    return {"passage_count": len(passages), "embedding_dim": int(matrix.shape[1]) if matrix.size else 0}


def save_retrieval_index(passages: list[dict], matrix: np.ndarray) -> None:
    ensure_data_dir()
    with open(PASSAGES_PATH, "w", encoding="utf-8") as f:
        json.dump({"passages": passages}, f, indent=2, ensure_ascii=False)
    np.savez_compressed(
        EMBEDDINGS_PATH,
        embeddings=matrix.astype(np.float32),
        passage_ids=np.array([p["passage_id"] for p in passages]),
    )
    logger.info("Saved retrieval index: %d passages", len(passages))


def load_retrieval_index() -> dict | None:
    if not PASSAGES_PATH.exists() or not EMBEDDINGS_PATH.exists():
        return None

    with open(PASSAGES_PATH, encoding="utf-8") as f:
        data = json.load(f)
    passages = data.get("passages") or []

    archive = np.load(EMBEDDINGS_PATH, allow_pickle=False)
    matrix = archive["embeddings"]
    ids = [str(x) for x in archive["passage_ids"].tolist()]
    id_to_row = {pid: i for i, pid in enumerate(ids)}

    ordered = np.zeros((len(passages), matrix.shape[1]), dtype=np.float32)
    for i, passage in enumerate(passages):
        row = id_to_row.get(passage["passage_id"])
        if row is not None:
            ordered[i] = matrix[row]

    return {"passages": passages, "embeddings": ordered}
