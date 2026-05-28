"""Langfuse tracing, latency percentiles, cost tracking, and quality metrics."""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import os
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable, Generator

logger = logging.getLogger(__name__)

GROQ_MODEL_DEFAULT = "llama-3.3-70b-versatile"
GROQ_PROVIDER = "groq"
# Groq llama-3.3-70b-versatile approximate pricing (USD per 1M tokens)
GROQ_INPUT_USD_PER_1M = float(os.getenv("GROQ_INPUT_USD_PER_1M", "0.59"))
GROQ_OUTPUT_USD_PER_1M = float(os.getenv("GROQ_OUTPUT_USD_PER_1M", "0.79"))

MAX_METRIC_SAMPLES = 500
TRACE_OUTPUT_MAX_CHARS = 4000
GENERATION_OUTPUT_MAX_CHARS = 8000


def langfuse_enabled() -> bool:
    pk = (os.getenv("LANGFUSE_PUBLIC_KEY") or "").strip()
    sk = (os.getenv("LANGFUSE_SECRET_KEY") or "").strip()
    return bool(pk and sk and "your-" not in pk.lower() and "pk-lf-your" not in pk.lower())


def _get_langfuse():
    if not langfuse_enabled():
        return None
    try:
        from langfuse import get_client

        return get_client()
    except Exception as exc:
        logger.warning("Langfuse client unavailable: %s", exc)
        return None


def flush_langfuse() -> None:
    client = _get_langfuse()
    if client is not None:
        try:
            client.flush()
        except Exception as exc:
            logger.debug("Langfuse flush: %s", exc)


atexit.register(flush_langfuse)


def _truncate(value: Any, limit: int = TRACE_OUTPUT_MAX_CHARS) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return value if len(value) <= limit else value[: limit - 3] + "..."
    if isinstance(value, dict):
        return {k: _truncate(v, limit=500) for k, v in value.items()}
    if isinstance(value, list):
        return [_truncate(v, limit=500) for v in value[:20]]
    return value


def _coerce_metadata(metadata: dict | None) -> dict[str, str] | None:
    """Langfuse metadata must be dict[str, str] with values <= 200 chars."""
    if not metadata:
        return None
    out: dict[str, str] = {}
    for key, val in metadata.items():
        s = str(val)
        out[str(key)[:64]] = s[:200]
    return out or None


def document_session_id(graph: dict, chunks: list[dict] | None = None) -> str:
    """Stable session id so multi-turn Q&A on the same upload groups in Langfuse Sessions."""
    first_chunk = (chunks or [{}])[0].get("id", "none")
    fingerprint = json.dumps(
        {
            "first_chunk": first_chunk,
            "nodes": len(graph.get("nodes", [])),
            "edges": len(graph.get("edges", [])),
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(fingerprint.encode()).hexdigest()[:16]
    return f"doc-{digest}"


@dataclass
class GroqUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_response(cls, response: Any) -> GroqUsage:
        usage = getattr(response, "usage", None)
        if usage is None:
            return cls()
        return cls(
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
        )


def estimate_groq_cost_usd(usage: GroqUsage) -> float:
    return (
        (usage.prompt_tokens / 1_000_000) * GROQ_INPUT_USD_PER_1M
        + (usage.completion_tokens / 1_000_000) * GROQ_OUTPUT_USD_PER_1M
    )


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    sorted_vals = sorted(values)
    idx = int(round((pct / 100) * (len(sorted_vals) - 1)))
    return round(sorted_vals[idx], 2)


@dataclass
class RequestMetric:
    operation: str
    total_ms: float
    cost_usd: float = 0.0
    stages: dict[str, float] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    groq_usage: dict[str, int] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class MetricsStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._samples: list[RequestMetric] = []
        self._groq_by_operation: dict[str, list[GroqUsage]] = {}

    def record(self, metric: RequestMetric) -> None:
        with self._lock:
            self._samples.append(metric)
            if len(self._samples) > MAX_METRIC_SAMPLES:
                self._samples = self._samples[-MAX_METRIC_SAMPLES:]

    def record_groq(self, operation: str, usage: GroqUsage) -> float:
        cost = estimate_groq_cost_usd(usage)
        with self._lock:
            self._groq_by_operation.setdefault(operation, []).append(usage)
        return cost

    def summary(self) -> dict:
        with self._lock:
            samples = list(self._samples)

        by_op: dict[str, list[RequestMetric]] = {}
        for s in samples:
            by_op.setdefault(s.operation, []).append(s)

        latency: dict[str, dict] = {}
        cost: dict[str, dict] = {}
        quality: dict[str, Any] = {}

        for op, op_samples in by_op.items():
            totals = [s.total_ms for s in op_samples]
            costs = [s.cost_usd for s in op_samples]
            latency[op] = {
                "count": len(totals),
                "p50_ms": _percentile(totals, 50),
                "p95_ms": _percentile(totals, 95),
                "avg_ms": round(sum(totals) / len(totals), 2) if totals else None,
            }
            cost[op] = {
                "count": len(costs),
                "total_usd": round(sum(costs), 6),
                "avg_usd": round(sum(costs) / len(costs), 6) if costs else None,
            }

        cite_valid = [s.quality.get("citation_valid") for s in samples if s.operation == "ask"]
        cite_valid = [v for v in cite_valid if v is not None]
        if cite_valid:
            quality["citation_valid_rate"] = round(
                sum(1 for v in cite_valid if v) / len(cite_valid), 4
            )
        quality["total_requests"] = len(samples)

        groq_totals = GroqUsage()
        with self._lock:
            for usages in self._groq_by_operation.values():
                for u in usages:
                    groq_totals.prompt_tokens += u.prompt_tokens
                    groq_totals.completion_tokens += u.completion_tokens
                    groq_totals.total_tokens += u.total_tokens

        return {
            "langfuse_enabled": langfuse_enabled(),
            "langfuse_host": os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            "tracing_environment": os.getenv("LANGFUSE_TRACING_ENVIRONMENT", "development"),
            "latency_ms": latency,
            "cost_usd": cost,
            "quality": quality,
            "groq_tokens": {
                "prompt": groq_totals.prompt_tokens,
                "completion": groq_totals.completion_tokens,
                "total": groq_totals.total_tokens,
                "estimated_total_usd": round(estimate_groq_cost_usd(groq_totals), 6),
            },
        }


metrics_store = MetricsStore()


@contextmanager
def trace_request(
    operation: str,
    *,
    input_data: Any = None,
    metadata: dict | None = None,
    session_id: str | None = None,
    tags: list[str] | None = None,
    build_output: Callable[[dict[str, Any]], Any] | None = None,
) -> Generator[dict[str, Any], None, None]:
    """Top-level trace with Langfuse propagate_attributes + nested stage spans."""
    ctx: dict[str, Any] = {"stages": {}, "cost_usd": 0.0, "groq_usage": GroqUsage()}
    start = time.perf_counter()
    langfuse = _get_langfuse()
    trace_name = f"ask-my-docs/{operation}"
    all_tags = ["ask-my-docs", operation, *(tags or [])]
    meta = _coerce_metadata(metadata)
    trace_input = _truncate(input_data)

    propagate_cm = nullcontext()
    if langfuse is not None:
        try:
            from langfuse import propagate_attributes

            propagate_cm = propagate_attributes(
                trace_name=trace_name,
                session_id=session_id,
                tags=all_tags,
                metadata=meta,
            )
        except Exception as exc:
            logger.debug("Langfuse propagate_attributes skipped: %s", exc)

    span_cm = nullcontext()
    if langfuse is not None:
        try:
            span_cm = langfuse.start_as_current_observation(
                as_type="span",
                name=operation,
                input=trace_input,
            )
        except Exception as exc:
            logger.debug("Langfuse root span skipped: %s", exc)

    with propagate_cm:
        with span_cm as span:
            try:
                yield ctx
            except Exception as exc:
                if span is not None:
                    try:
                        span.update(level="ERROR", status_message=str(exc)[:500])
                    except Exception:
                        pass
                raise
            finally:
                total_ms = (time.perf_counter() - start) * 1000
                default_output = {
                    "total_ms": round(total_ms, 2),
                    "cost_usd": round(ctx.get("cost_usd", 0.0), 6),
                    "stages_ms": ctx.get("stages", {}),
                    "quality": _truncate(ctx.get("quality", {})),
                }
                trace_output = (
                    _truncate(build_output(ctx))
                    if build_output is not None
                    else default_output
                )
                if span is not None:
                    try:
                        span.update(output=trace_output)
                    except Exception:
                        pass

                metrics_store.record(
                    RequestMetric(
                        operation=operation,
                        total_ms=round(total_ms, 2),
                        cost_usd=ctx.get("cost_usd", 0.0),
                        stages=ctx.get("stages", {}),
                        quality=ctx.get("quality", {}),
                        groq_usage={
                            "prompt": ctx.get("groq_usage", GroqUsage()).prompt_tokens,
                            "completion": ctx.get("groq_usage", GroqUsage()).completion_tokens,
                        },
                    )
                )
                flush_langfuse()


@contextmanager
def trace_stage(
    ctx: dict,
    name: str,
    *,
    input_data: Any = None,
) -> Generator[None, None, None]:
    start = time.perf_counter()
    langfuse = _get_langfuse()
    child_cm = nullcontext()
    if langfuse is not None:
        try:
            child_cm = langfuse.start_as_current_observation(
                as_type="span",
                name=name,
                input=_truncate(input_data) if input_data is not None else None,
            )
        except Exception:
            child_cm = nullcontext()

    with child_cm as child:
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            ctx.setdefault("stages", {})[name] = round(elapsed, 2)
            if child is not None:
                try:
                    child.update(output={"duration_ms": round(elapsed, 2)})
                except Exception:
                    pass


@contextmanager
def trace_groq_generation(
    ctx: dict,
    *,
    operation: str,
    model: str,
    messages: list[dict] | None = None,
) -> Generator[None, None, None]:
    langfuse = _get_langfuse()
    gen_cm = nullcontext()
    start = time.perf_counter()

    gen_input: dict[str, Any] | None = None
    if messages:
        # Trace user-facing input only; omit system prompts with full source text.
        user_msgs = [m for m in messages if m.get("role") == "user"]
        last_user = user_msgs[-1]["content"] if user_msgs else None
        gen_input = {
            "operation": operation,
            "message_count": len(messages),
            "last_user_message": _truncate(last_user, limit=1500),
        }

    if langfuse is not None:
        try:
            gen_cm = langfuse.start_as_current_observation(
                as_type="generation",
                name=operation,
                model=model,
                input=gen_input,
                metadata={"provider": GROQ_PROVIDER},
            )
        except Exception:
            gen_cm = nullcontext()

    with gen_cm:
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            ctx.setdefault("stages", {})[f"groq_{operation}"] = round(elapsed, 2)


def record_groq_response(
    ctx: dict,
    response: Any,
    *,
    operation: str,
    model: str,
) -> GroqUsage:
    usage = GroqUsage.from_response(response)
    cost = metrics_store.record_groq(operation, usage)
    ctx["cost_usd"] = ctx.get("cost_usd", 0.0) + cost
    existing: GroqUsage = ctx.get("groq_usage", GroqUsage())
    ctx["groq_usage"] = GroqUsage(
        prompt_tokens=existing.prompt_tokens + usage.prompt_tokens,
        completion_tokens=existing.completion_tokens + usage.completion_tokens,
        total_tokens=existing.total_tokens + usage.total_tokens,
    )

    content = ""
    try:
        content = (response.choices[0].message.content or "").strip()
    except (AttributeError, IndexError, TypeError):
        pass

    langfuse = _get_langfuse()
    if langfuse is not None:
        try:
            langfuse.update_current_generation(
                model=model,
                output=_truncate(content, limit=GENERATION_OUTPUT_MAX_CHARS),
                usage={
                    "input": usage.prompt_tokens,
                    "output": usage.completion_tokens,
                    "total": usage.total_tokens,
                },
                cost=cost,
                metadata={
                    "provider": GROQ_PROVIDER,
                    "estimated_cost_usd": str(round(cost, 8)),
                },
            )
        except Exception:
            pass
    return usage


def score_langfuse_eval(
    *,
    name: str,
    value: float,
    comment: str | None = None,
) -> None:
    """Attach quality score to current trace (for Langfuse dashboards)."""
    client = _get_langfuse()
    if client is None:
        return
    try:
        client.score_current_trace(name=name, value=value, comment=comment)
    except Exception as exc:
        logger.debug("Langfuse score skipped: %s", exc)
