"""Shared Groq client configuration (avoids circular imports with groq_llm / graph_builder)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def validate_groq_api_key() -> None:
    """Raise if GROQ_API_KEY is missing or still a placeholder."""
    load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

    api_key = (os.getenv("GROQ_API_KEY") or "").strip().strip('"').strip("'")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Add it to backend/.env and restart the server."
        )
    if api_key == "gsk_your-key-here" or len(api_key) < 20:
        raise ValueError(
            "GROQ_API_KEY in backend/.env still looks like a placeholder. "
            "Save your full gsk_... key and restart uvicorn."
        )


def get_groq_client() -> Groq:
    validate_groq_api_key()
    api_key = (os.getenv("GROQ_API_KEY") or "").strip().strip('"').strip("'")
    return Groq(api_key=api_key)
