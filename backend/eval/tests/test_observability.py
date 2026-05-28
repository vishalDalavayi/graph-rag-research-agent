"""Tests for in-process metrics (p50/p95) and Langfuse graceful degradation."""

from __future__ import annotations

import os

from observability import (
    GroqUsage,
    MetricsStore,
    RequestMetric,
    _percentile,
    document_session_id,
    estimate_groq_cost_usd,
    langfuse_enabled,
)


def test_percentile_empty():
    assert _percentile([], 50) is None


def test_percentile_p50_p95():
    values = [10.0, 20.0, 30.0, 40.0, 100.0]
    assert _percentile(values, 50) == 30.0
    assert _percentile(values, 95) == 100.0


def test_groq_cost_estimate():
    usage = GroqUsage(prompt_tokens=1_000_000, completion_tokens=1_000_000)
    cost = estimate_groq_cost_usd(usage)
    assert cost > 0


def test_metrics_store_summary_latency_and_cost():
    store = MetricsStore()
    for ms in [100.0, 200.0, 300.0, 400.0, 500.0]:
        store.record(
            RequestMetric(
                operation="ask",
                total_ms=ms,
                cost_usd=0.001,
                quality={"citation_valid": True},
            )
        )
    summary = store.summary()
    assert summary["latency_ms"]["ask"]["count"] == 5
    assert summary["latency_ms"]["ask"]["p50_ms"] is not None
    assert summary["latency_ms"]["ask"]["p95_ms"] is not None
    assert summary["cost_usd"]["ask"]["avg_usd"] == 0.001
    assert summary["quality"]["citation_valid_rate"] == 1.0


def test_langfuse_disabled_without_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert langfuse_enabled() is False


def test_document_session_id_stable():
    graph = {"nodes": [{"id": "a"}], "edges": [{"id": "e1"}]}
    chunks = [{"id": "chunk_001", "text": "hello"}]
    s1 = document_session_id(graph, chunks)
    s2 = document_session_id(graph, chunks)
    assert s1 == s2
    assert s1.startswith("doc-")
