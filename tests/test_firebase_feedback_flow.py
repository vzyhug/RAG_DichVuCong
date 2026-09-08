from __future__ import annotations

from typing import Any

from src.training_data import firebase_store
from src.training_data.collector import build_chat_record


class _Document:
    def __init__(self, document_id: str, data: dict[str, Any]):
        self.id = document_id
        self.data = data
        self.reference = self

    def to_dict(self) -> dict[str, Any]:
        return dict(self.data)

    def update(self, values: dict[str, Any]) -> None:
        self.data.update(values)


class _Query:
    def __init__(self, documents: list[_Document]):
        self.documents = documents
        self.filters: list[tuple[str, Any]] = []
        self.limit_value: int | None = None

    def where(self, field: str, _operator: str, value: Any) -> "_Query":
        self.filters.append((field, value))
        return self

    def limit(self, value: int) -> "_Query":
        self.limit_value = value
        return self

    def stream(self) -> list[_Document]:
        matched = self.documents
        for field, expected in self.filters:
            if field == "feedback.rating":
                matched = [
                    document
                    for document in matched
                    if (document.data.get("feedback") or {}).get("rating") == expected
                ]
            else:
                matched = [
                    document for document in matched if document.data.get(field) == expected
                ]
        if self.limit_value is not None:
            matched = matched[: self.limit_value]
        return matched


class _Collection(_Query):
    def __init__(self):
        super().__init__([])

    def where(self, field: str, operator: str, value: Any) -> _Query:
        query = _Query(self.documents)
        return query.where(field, operator, value)

    def limit(self, value: int) -> _Query:
        query = _Query(self.documents)
        return query.limit(value)

    def add(self, record: dict[str, Any]) -> tuple[None, _Document]:
        document = _Document(f"doc-{len(self.documents) + 1}", dict(record))
        self.documents.append(document)
        return None, document


def test_feedback_and_review_states_are_persisted_and_filterable(monkeypatch):
    collection = _Collection()
    monkeypatch.setattr(firebase_store, "_get_chat_collection", lambda: collection)

    for turn_id in ("approved-turn", "edited-turn", "rejected-turn"):
        record = build_chat_record(
            user_query="same question",
            assistant_answer=f"generated {turn_id}",
            response_type="normal",
            turn_id=turn_id,
        )
        assert firebase_store.save_chat_log(record) == f"doc-{len(collection.documents)}"

    assert firebase_store.update_chat_feedback("approved-turn", {"rating": "up"})
    assert firebase_store.update_review_status("approved-turn", "approved")
    assert firebase_store.update_review_status(
        "edited-turn", "edited", corrected_answer="reviewed answer"
    )
    assert firebase_store.update_review_status("rejected-turn", "rejected")

    approved = firebase_store.list_chat_logs(review_status="approved")
    edited = firebase_store.list_chat_logs(review_status="edited")
    rejected = firebase_store.list_chat_logs(review_status="rejected")
    rated = firebase_store.list_chat_logs(rating="up")

    assert [row["turn_id"] for row in approved] == ["approved-turn"]
    assert edited[0]["corrected_answer"] == "reviewed answer"
    assert [row["turn_id"] for row in rejected] == ["rejected-turn"]
    assert [row["turn_id"] for row in rated] == ["approved-turn"]
