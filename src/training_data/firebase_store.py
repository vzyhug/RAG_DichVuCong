"""Firestore persistence for chatbot conversation records."""

from __future__ import annotations

import logging
import os
import json
from threading import Lock
from typing import Any
from datetime import datetime, timezone


logger = logging.getLogger(__name__)

_client_lock = Lock()
_firestore_client: Any | None = None

_REVIEW_STATUSES = {"raw", "approved", "rejected", "edited"}


def _get_secret_value(key: str, default: str = "") -> str:
    """Read a config value from env or Streamlit secrets."""
    value = os.getenv(key)
    if value:
        return str(value)

    try:
        import streamlit as st

        if key in st.secrets:
            return str(st.secrets[key])
        for section_name in ("configs", "firebase", "github", "gemini"):
            if section_name not in st.secrets:
                continue
            section = st.secrets.get(section_name)
            if hasattr(section, "get") and section.get(key):
                return str(section.get(key))
    except Exception:
        pass
    return default


def _is_enabled() -> bool:
    return _get_secret_value("ENABLE_FIREBASE_LOGGING", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _has_inline_credentials() -> bool:
    return bool(_get_secret_value("FIREBASE_SERVICE_ACCOUNT_JSON").strip())


def _get_firestore_client(credentials_path: str | None = None) -> Any | None:
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
                service_account_json = _get_secret_value(
                    "FIREBASE_SERVICE_ACCOUNT_JSON"
                ).strip()
                if service_account_json:
                    service_account_info = json.loads(service_account_json)
                    firebase_credentials = credentials.Certificate(
                        service_account_info
                    )
                else:
                    firebase_credentials = credentials.Certificate(credentials_path)
                app = firebase_admin.initialize_app(firebase_credentials)

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

    credentials_path = _get_secret_value(
        "FIREBASE_CREDENTIALS_PATH", "firebase-service-account.json"
    ).strip()
    if not _has_inline_credentials() and not credentials_path:
        logger.warning("Firebase credentials path is not configured")
        return None

    if not _has_inline_credentials() and not os.path.isfile(credentials_path):
        logger.warning("Firebase credentials file is missing: %s", credentials_path)
        return None

    collection_name = _get_secret_value("FIREBASE_COLLECTION", "chat_logs").strip()
    if not collection_name:
        logger.warning("Firebase collection is not configured")
        return None

    client = _get_firestore_client(credentials_path)
    if client is None:
        return None
    return client.collection(collection_name)


def _get_data_update_collection() -> Any | None:
    """Return the configured data-update request collection."""
    if not _is_enabled():
        logger.warning("Firebase data update request logging is disabled")
        return None

    credentials_path = _get_secret_value(
        "FIREBASE_CREDENTIALS_PATH", "firebase-service-account.json"
    ).strip()
    if not _has_inline_credentials() and not credentials_path:
        logger.warning("Firebase credentials path is not configured")
        return None

    if not _has_inline_credentials() and not os.path.isfile(credentials_path):
        logger.warning("Firebase credentials file is missing: %s", credentials_path)
        return None

    collection_name = _get_secret_value(
        "FIREBASE_DATA_UPDATE_COLLECTION", "data_update_requests"
    ).strip()
    if not collection_name:
        logger.warning("Firebase data update collection is not configured")
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
        feedback_payload = dict(feedback)
        feedback_payload.setdefault(
            "feedback_at", datetime.now(timezone.utc).isoformat()
        )
        document.reference.update({"feedback": feedback_payload})
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
        filter_raw_locally = review_status == "raw"
        if review_status is not None and not filter_raw_locally:
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
            record.setdefault("review_status", "raw")
            if filter_raw_locally and record.get("review_status") != "raw":
                continue
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
    needs_data_update: bool | None = None,
    data_update_note: str | None = None,
    data_update_request_id: str | None = None,
    github_issue_url: str | None = None,
    github_issue_number: int | None = None,
    github_project_item_id: str | None = None,
    github_project_error: str | None = None,
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
        update_payload = {
            "review_status": review_status,
            "corrected_answer": corrected_answer,
            "error_type": error_type,
        }
        if needs_data_update is not None:
            update_payload["needs_data_update"] = needs_data_update
        if data_update_note is not None:
            update_payload["data_update_note"] = data_update_note
        if data_update_request_id is not None:
            update_payload["data_update_request_id"] = data_update_request_id
        if github_issue_url is not None:
            update_payload["github_issue_url"] = github_issue_url
        if github_issue_number is not None:
            update_payload["github_issue_number"] = github_issue_number
        if github_project_item_id is not None:
            update_payload["github_project_item_id"] = github_project_item_id
        if github_project_error is not None:
            update_payload["github_project_error"] = github_project_error
        document.reference.update(update_payload)
        return True
    except Exception:
        logger.warning("Firebase review status update failed", exc_info=True)
        return False


def create_data_update_request(payload: dict) -> str | None:
    """Persist a reviewed data-update request for developer follow-up."""
    if not isinstance(payload, dict):
        return None

    try:
        collection = _get_data_update_collection()
        if collection is None:
            return None

        request_payload = dict(payload)
        request_payload.setdefault("status", "open")
        request_payload.setdefault("priority", "normal")
        request_payload.setdefault(
            "created_at", datetime.now(timezone.utc).isoformat()
        )
        result = collection.add(request_payload)
        document = result[1] if isinstance(result, tuple) else result
        document_id = getattr(document, "id", None)
        if not document_id:
            logger.warning("Firebase returned no document ID for data update request")
            return None
        return str(document_id)
    except Exception:
        logger.warning("Firebase data update request save failed", exc_info=True)
        return None
