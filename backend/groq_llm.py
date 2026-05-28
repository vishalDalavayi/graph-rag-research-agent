"""Groq chat completions with Langfuse tracing and cost attribution."""

from __future__ import annotations

from typing import Any

from groq_client import GROQ_MODEL, get_groq_client
from observability import record_groq_response, trace_groq_generation


def chat_completion(
    messages: list[dict],
    ctx: dict,
    *,
    operation: str,
    temperature: float = 0.2,
    response_format: dict | None = None,
    model: str = GROQ_MODEL,
) -> Any:
    with trace_groq_generation(
        ctx,
        operation=operation,
        model=model,
        messages=messages,
    ):
        kwargs: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        response = get_groq_client().chat.completions.create(**kwargs)
        record_groq_response(ctx, response, operation=operation, model=model)
        return response
