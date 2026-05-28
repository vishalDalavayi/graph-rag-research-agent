#!/usr/bin/env python3
"""CI-gated evaluation for Ask My Docs retrieval, citations, and regression baselines."""

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
BASELINES = Path(__file__).parent / "baselines.json"
DEFAULT_REPORT = Path(__file__).parent / "last_report.json"


def run_offline_eval(
    *,
    min_score: float = 0.75,
    check_regression: bool = True,
    report_path: Path | None = DEFAULT_REPORT,
) -> int:
    data = json.loads(FIXTURE.read_text())
    passages = build_passages(data["chunks"], data["graph"])
    matrix = embed_texts([p["text"] for p in passages], offline=True)

    case_results: list[dict] = []
    retrieval_hits = 0
    citation_passes = 0
    citation_checks = 0

    for case in data["cases"]:
        question = case["question"]
        retrieved = hybrid_retrieve(
            question, passages, matrix, offline=True, skip_rerank=True
        )
        text_blob = " ".join(r["text"] for r in retrieved).lower()
        recall_ok = all(
            s.lower() in text_blob for s in case.get("expected_substrings", [])
        )
        retrieval_hit = bool(retrieved)
        if retrieval_hit:
            retrieval_hits += 1

        cite_ok = True
        if case.get("must_cite") and retrieved:
            citation_checks += 1
            numbered, _ = build_numbered_sources(retrieved[:3])
            fake_answer = f"Answer with cite [{numbered[0]['citation_id']}]."
            cite_ok = validate_citations(fake_answer, len(numbered))["valid"]
            if cite_ok:
                citation_passes += 1

        passed = recall_ok and (not case.get("must_cite") or cite_ok)
        case_results.append(
            {
                "question": question,
                "passed": passed,
                "retrieval_hit": retrieval_hit,
                "recall_ok": recall_ok,
                "citation_ok": cite_ok if case.get("must_cite") else None,
            }
        )
        if not passed:
            print(f"FAIL: {question}")

    total = len(data["cases"])
    passed_count = sum(1 for c in case_results if c["passed"])
    score = passed_count / max(total, 1)
    retrieval_hit_rate = retrieval_hits / max(total, 1)
    citation_fixture_pass_rate = (
        citation_passes / citation_checks if citation_checks else 1.0
    )

    report = {
        "fixture": str(FIXTURE.name),
        "total_cases": total,
        "passed": passed_count,
        "score": round(score, 4),
        "retrieval_hit_rate": round(retrieval_hit_rate, 4),
        "citation_fixture_pass_rate": round(citation_fixture_pass_rate, 4),
        "cases": case_results,
        "thresholds": {"min_score": min_score},
    }

    print(f"Eval score: {passed_count}/{total} ({score:.0%})")
    print(f"Retrieval hit rate: {retrieval_hit_rate:.0%}")
    if citation_checks:
        print(f"Citation fixture pass rate: {citation_fixture_pass_rate:.0%}")

    if report_path is not None:
        report_path.write_text(json.dumps(report, indent=2))
        print(f"Wrote report: {report_path}")

    failures: list[str] = []
    if score < min_score:
        failures.append(f"score {score:.0%} < min_score {min_score:.0%}")

    if check_regression and BASELINES.exists():
        baselines = json.loads(BASELINES.read_text())
        bl_min = baselines.get("min_score", min_score)
        if score < bl_min:
            failures.append(f"score {score:.0%} < baseline min_score {bl_min:.0%}")
        bl_retrieval = baselines.get("min_retrieval_hit_rate")
        if bl_retrieval is not None and retrieval_hit_rate < bl_retrieval:
            failures.append(
                f"retrieval_hit_rate {retrieval_hit_rate:.0%} "
                f"< baseline {bl_retrieval:.0%}"
            )
        bl_cite = baselines.get("min_citation_fixture_pass_rate")
        if bl_cite is not None and citation_fixture_pass_rate < bl_cite:
            failures.append(
                f"citation_fixture_pass_rate {citation_fixture_pass_rate:.0%} "
                f"< baseline {bl_cite:.0%}"
            )
        report["baselines"] = baselines
        report["regression_passed"] = len(failures) == 0

    if failures:
        print("Regression gate FAILED:")
        for f in failures:
            print(f"  - {f}")
        if report_path is not None:
            report_path.write_text(json.dumps(report, indent=2))
        return 1

    print("Eval passed (regression gate OK)")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", default=True)
    parser.add_argument("--min-score", type=float, default=0.75)
    parser.add_argument(
        "--no-regression",
        action="store_true",
        help="Skip comparison against eval/baselines.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="Write JSON report (use /dev/null style omit with empty string)",
    )
    args = parser.parse_args()
    report_path = args.report if str(args.report) else None
    raise SystemExit(
        run_offline_eval(
            min_score=args.min_score,
            check_regression=not args.no_regression,
            report_path=report_path,
        )
    )
