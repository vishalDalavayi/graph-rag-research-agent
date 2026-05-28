"""Hybrid retrieval: BM25 + vector search, RRF fusion, cross-encoder reranking."""

from __future__ import annotations

import logging
import re

import numpy as np
from rank_bm25 import BM25Okapi

from embeddings import cosine_top_k, embed_texts

logger = logging.getLogger(__name__)

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

BM25_TOP_K = 30
VECTOR_TOP_K = 30
RRF_K = 60
RERANK_POOL = 25
FINAL_TOP_K = 10
RRF_CONSTANT = 60


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def bm25_search(query: str, passages: list[dict], k: int = BM25_TOP_K) -> list[tuple[str, float]]:
    if not passages:
        return []
    corpus = [tokenize(p["text"]) for p in passages]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
    return [(passages[i]["passage_id"], float(s)) for i, s in ranked if s > 0]


def vector_search(
    query: str,
    passages: list[dict],
    embedding_matrix: np.ndarray,
    k: int = VECTOR_TOP_K,
    *,
    offline: bool = False,
) -> list[tuple[str, float]]:
    if not passages or embedding_matrix.size == 0:
        return []
    query_vec = embed_texts([query], offline=offline)[0]
    top = cosine_top_k(query_vec, embedding_matrix, k)
    return [(passages[i]["passage_id"], score) for i, score in top]


def reciprocal_rank_fusion(
    rankings: list[list[tuple[str, float]]],
    k: int = RERANK_POOL,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (pid, _) in enumerate(ranking):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (RRF_CONSTANT + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]


_reranker = None


def _get_cross_encoder():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        logger.info("Loading cross-encoder: %s", model_name)
        _reranker = CrossEncoder(model_name)
    return _reranker


def cross_encoder_rerank(
    query: str,
    passages_by_id: dict[str, dict],
    candidate_ids: list[str],
    top_k: int = FINAL_TOP_K,
    *,
    skip_rerank: bool = False,
) -> list[tuple[str, float]]:
    if skip_rerank or not candidate_ids:
        return [(pid, 1.0) for pid in candidate_ids[:top_k]]

    pairs = []
    ids = []
    for pid in candidate_ids:
        passage = passages_by_id.get(pid)
        if passage:
            pairs.append([query, passage["text"]])
            ids.append(pid)

    if not pairs:
        return []

    reranker = _get_cross_encoder()
    scores = reranker.predict(pairs)
    ranked = sorted(zip(ids, scores), key=lambda x: float(x[1]), reverse=True)
    return [(pid, float(score)) for pid, score in ranked[:top_k]]


def hybrid_retrieve(
    query: str,
    passages: list[dict],
    embedding_matrix: np.ndarray,
    *,
    offline: bool = False,
    skip_rerank: bool = False,
) -> list[dict]:
    """BM25 + vector → RRF → cross-encoder rerank → passage dicts."""
    if not passages:
        return []

    passages_by_id = {p["passage_id"]: p for p in passages}
    bm25_hits = bm25_search(query, passages)
    vector_hits = vector_search(query, passages, embedding_matrix, offline=offline)

    logger.info("Retrieval: BM25=%d, vector=%d", len(bm25_hits), len(vector_hits))

    fused = reciprocal_rank_fusion([bm25_hits, vector_hits])
    candidate_ids = [pid for pid, _ in fused]
    reranked = cross_encoder_rerank(
        query,
        passages_by_id,
        candidate_ids,
        top_k=FINAL_TOP_K,
        skip_rerank=skip_rerank or offline,
    )

    result = []
    for pid, score in reranked:
        passage = dict(passages_by_id[pid])
        passage["retrieval_score"] = score
        result.append(passage)

    logger.info("Retrieval final: %d passages after rerank", len(result))
    return result
