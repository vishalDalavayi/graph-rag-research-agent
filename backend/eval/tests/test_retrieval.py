import json
from pathlib import Path

import numpy as np

from embeddings import embed_texts
from passages import build_passages
from retrieval import bm25_search, hybrid_retrieve, reciprocal_rank_fusion

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "sample_document.json"


def _load_fixture():
    data = json.loads(FIXTURE.read_text())
    passages = build_passages(data["chunks"], data["graph"])
    matrix = embed_texts([p["text"] for p in passages], offline=True)
    return passages, matrix, data


def test_bm25_finds_founder():
    passages, _, data = _load_fixture()
    hits = bm25_search("Who founded Acme Corp?", passages, k=5)
    assert hits
    hit_ids = {h[0] for h in hits}
    matched_text = " ".join(p["text"] for p in passages if p["passage_id"] in hit_ids)
    assert "Alice" in matched_text or "founded" in matched_text.lower()


def test_rrf_combines_rankings():
    fused = reciprocal_rank_fusion(
        [
            [("p0", 1.0), ("p1", 0.5)],
            [("p1", 1.0), ("p0", 0.3)],
        ],
        k=2,
    )
    assert len(fused) == 2
    assert fused[0][0] in {"p0", "p1"}


def test_hybrid_retrieve_offline():
    passages, matrix, data = _load_fixture()
    results = hybrid_retrieve(
        data["cases"][0]["question"],
        passages,
        matrix,
        offline=True,
        skip_rerank=True,
    )
    assert results
    combined = " ".join(r["text"] for r in results).lower()
    assert "alice" in combined or "acme" in combined
