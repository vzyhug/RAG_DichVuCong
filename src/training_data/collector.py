"""Build and persist normalized chatbot conversation records."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any, Iterable, Mapping
from uuid import uuid4

from .firebase_store import save_chat_log


logger = logging.getLogger(__name__)


def _normalize_contexts(
    contexts: Iterable[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for context in contexts or []:
        if not isinstance(context, Mapping):
            continue
        metadata = context.get("metadata", {})
        normalized.append(
            {
                "text": context.get("text", ""),
                "metadata": dict(metadata) if isinstance(metadata, Mapping) else {},
                "score": context.get("score"),
            }
        )
    return normalized


def build_chat_record(
    user_query: str,
    assistant_answer: str,
    response_type: str,
    contexts: Iterable[Mapping[str, Any]] | None = None,
    entities: Mapping[str, Any] | None = None,
    model: str | None = None,
    provider: str | None = None,
    session_id: str | None = None,
    turn_id: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build the stable Firestore schema for one chatbot turn."""
    return {
        "session_id": session_id or str(uuid4()),
        "turn_id": turn_id or str(uuid4()),
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "user_query": user_query,
        "assistant_answer": assistant_answer,
        "response_type": response_type,
        "contexts": _normalize_contexts(contexts),
        "entities": dict(entities) if isinstance(entities, Mapping) else {},
        "model": model,
        "provider": provider,
        "review_status": "raw",
        "feedback": None,
        "corrected_answer": None,
    }


def collect_chat_log(
    user_query: str,
    assistant_answer: str,
    response_type: str,
    contexts: Iterable[Mapping[str, Any]] | None = None,
    entities: Mapping[str, Any] | None = None,
    model: str | None = None,
    provider: str | None = None,
    session_id: str | None = None,
    turn_id: str | None = None,
) -> str | None:
    """Build and persist one chat record without affecting the chatbot flow."""
    try:
        record = build_chat_record(
            user_query=user_query,
            assistant_answer=assistant_answer,
            response_type=response_type,
            contexts=contexts,
            entities=entities,
            model=model,
            provider=provider,
            session_id=session_id,
            turn_id=turn_id,
        )
        return save_chat_log(record)
    except Exception:
        logger.warning("Could not collect chat log", exc_info=True)
        return None
