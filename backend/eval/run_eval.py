#!/usr/bin/env python3
"""CI-gated evaluation for Ask My Docs retrieval and citations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from citations import build_numbered_sources, validate_citations
from embeddings import embed_texts
from passages import build_passages
from retrieval import hybrid_retrieve

FIXTURE = Path(__file__).parent / "fixtures" / "sample_document.json"


def run_offline_eval(min_score: float = 0.75) -> int:
    data = json.loads(FIXTURE.read_text())
    passages = build_passages(data["chunks"], data["graph"])
    matrix = embed_texts([p["text"] for p in passages], offline=True)

    passed = 0
    total = len(data["cases"])

    for case in data["cases"]:
        question = case["question"]
        retrieved = hybrid_retrieve(
            question, passages, matrix, offline=True, skip_rerank=True
        )
        text_blob = " ".join(r["text"] for r in retrieved).lower()
        ok = all(s.lower() in text_blob for s in case.get("expected_substrings", []))

        if ok and case.get("must_cite"):
            numbered, _ = build_numbered_sources(retrieved[:3])
            fake_answer = f"Answer with cite [{numbered[0]['citation_id']}]."
            cite_ok = validate_citations(fake_answer, len(numbered))["valid"]
            ok = ok and cite_ok

        if ok:
            passed += 1
        else:
            print(f"FAIL: {question}")

    score = passed / max(total, 1)
    print(f"Eval score: {passed}/{total} ({score:.0%})")
    if score < min_score:
        print(f"Below threshold {min_score:.0%}")
        return 1
    print("Eval passed")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", default=True)
    parser.add_argument("--min-score", type=float, default=0.75)
    args = parser.parse_args()
    raise SystemExit(run_offline_eval(min_score=args.min_score))
