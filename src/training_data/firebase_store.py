"""Firestore persistence for chatbot conversation records."""

from __future__ import annotations

import logging
import os
from threading import Lock
from typing import Any


logger = logging.getLogger(__name__)

_client_lock = Lock()
_firestore_client: Any | None = None

_REVIEW_STATUSES = {"raw", "approved", "rejected", "edited"}


def _is_enabled() -> bool:
    return os.getenv("ENABLE_FIREBASE_LOGGING", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _get_firestore_client(credentials_path: str) -> Any | None:
    """Return a cached Firestore client, initializing Firebase at most once."""
    global _firestore_client

    with _client_lock:
        if _firestore_client is not None:
            return _firestore_client

        try:
            import firebase_admin
            from firebase_admin import credentials, firestore

            try:
                app = firebase_admin.get_app()
            except ValueError:
                app = firebase_admin.initialize_app(
                    credentials.Certificate(credentials_path)
                )

            _firestore_client = firestore.client(app)
            return _firestore_client
        except Exception:
            logger.warning("Firebase initialization failed", exc_info=True)
            return None


def _get_chat_collection() -> Any | None:
    """Return the configured chat-log collection, or None when unavailable."""
    if not _is_enabled():
        logger.warning("Firebase chat logging is disabled")
        return None

    credentials_path = os.getenv(
        "FIREBASE_CREDENTIALS_PATH", "firebase-service-account.json"
    ).strip()
    if not credentials_path:
        logger.warning("Firebase credentials path is not configured")
        return None

    if not os.path.isfile(credentials_path):
        logger.warning("Firebase credentials file is missing: %s", credentials_path)
        return None

    collection_name = os.getenv("FIREBASE_COLLECTION", "chat_logs").strip()
    if not collection_name:
        logger.warning("Firebase collection is not configured")
        return None

    client = _get_firestore_client(credentials_path)
    if client is None:
        return None
    return client.collection(collection_name)


def _find_chat_document(collection: Any, turn_id: str) -> Any | None:
    """Find the first chat log for a turn ID."""
    documents = collection.where("turn_id", "==", turn_id).limit(1).stream()
    return next(iter(documents), None)


def save_chat_log(record: dict) -> str | None:
    """Save a chat record and return its Firestore document ID.

    Logging is deliberately best-effort so Firebase outages do not affect chat.
    """
    try:
        collection = _get_chat_collection()
        if collection is None:
            return None

        result = collection.add(record)
        document = result[1] if isinstance(result, tuple) else result
        document_id = getattr(document, "id", None)
        if not document_id:
            logger.warning("Firebase returned no document ID for chat log")
            return None
        return str(document_id)
    except Exception:
        logger.warning("Firebase chat log save failed", exc_info=True)
        return None


def update_chat_feedback(turn_id: str, feedback: dict) -> bool:
    """Update feedback for a chat turn without interrupting the chatbot."""
    if not turn_id or not isinstance(feedback, dict):
        return False

    try:
        collection = _get_chat_collection()
        if collection is None:
            return False

        document = _find_chat_document(collection, turn_id)
        if document is None:
            return False
        document.reference.update({"feedback": dict(feedback)})
        return True
    except Exception:
        logger.warning("Firebase chat feedback update failed", exc_info=True)
        return False


def list_chat_logs(
    limit: int = 100,
    review_status: str | None = None,
    rating: str | None = None,
) -> list[dict]:
    """Return recent chat logs, optionally filtered by status and rating."""
    try:
        limit = max(0, int(limit))
    except (TypeError, ValueError):
        return []
    if limit == 0:
        return []

    try:
        collection = _get_chat_collection()
        if collection is None:
            return []

        query = collection
        if review_status is not None:
            query = query.where("review_status", "==", review_status)
        if rating is not None:
            query = query.where("feedback.rating", "==", rating)

        logs: list[dict] = []
        for document in query.limit(limit).stream():
            if hasattr(document, "to_dict"):
                record = document.to_dict() or {}
            elif isinstance(document, dict):
                record = dict(document)
            else:
                continue
            record["document_id"] = str(getattr(document, "id", ""))
            logs.append(record)
        return logs
    except Exception:
        logger.warning("Firebase chat log listing failed", exc_info=True)
        return []


def update_review_status(
    turn_id: str,
    review_status: str,
    corrected_answer: str | None = None,
    error_type: str | None = None,
) -> bool:
    """Save an admin review decision for a chat turn."""
    if not turn_id or review_status not in _REVIEW_STATUSES:
        return False

    try:
        collection = _get_chat_collection()
        if collection is None:
            return False

        document = _find_chat_document(collection, turn_id)
        if document is None:
            return False
        document.reference.update(
            {
                "review_status": review_status,
                "corrected_answer": corrected_answer,
                "error_type": error_type,
            }
        )
        return True
    except Exception:
        logger.warning("Firebase review status update failed", exc_info=True)
        return False
