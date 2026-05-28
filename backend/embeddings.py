"""Local text embeddings via sentence-transformers (Groq has no embedding API)."""

from __future__ import annotations

import hashlib
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

# Groq does not host embedding models; use local MiniLM (see Groq RAG cookbook).
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = 384

_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _encoder = SentenceTransformer(EMBEDDING_MODEL)
    return _encoder


def embed_texts(texts: list[str], *, offline: bool = False) -> np.ndarray:
    """Return (n, dim) float32 embedding matrix."""
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

    if offline:
        return _offline_embeddings(texts, dim=EMBEDDING_DIM)

    matrix = _get_encoder().encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(matrix, dtype=np.float32)


def _offline_embeddings(texts: list[str], dim: int = EMBEDDING_DIM) -> np.ndarray:
    """Deterministic pseudo-embeddings for CI offline eval."""
    rows = []
    for text in texts:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        rng = np.frombuffer(digest * (dim // 32 + 1), dtype=np.uint8)[:dim].astype(np.float32)
        rng = rng / 255.0 - 0.5
        rows.append(rng)
    return _l2_normalize(np.stack(rows, axis=0))


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


def cosine_top_k(query_vec: np.ndarray, matrix: np.ndarray, k: int) -> list[tuple[int, float]]:
    if matrix.size == 0:
        return []
    scores = matrix @ query_vec
    k = min(k, len(scores))
    top_idx = np.argpartition(-scores, k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return [(int(i), float(scores[i])) for i in top_idx]
