"""Export reviewed chat logs for RAG chunking and re-index analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
load_dotenv(ROOT_DIR / ".env")

from src.training_data.anonymizer import redact_record  # noqa: E402
from src.training_data.firebase_store import list_chat_logs  # noqa: E402


DEFAULT_OUTPUT_PATH = (
    ROOT_DIR / "data" / "training" / "exports" / "chunk_review_dataset.jsonl"
)


def _text(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field, "")
    return value.strip() if isinstance(value, str) else ""


def _normalize_context(context: Any) -> dict[str, Any] | None:
    if not isinstance(context, Mapping):
        return None
    metadata = context.get("metadata", {})
    metadata = metadata if isinstance(metadata, Mapping) else {}
    return {
        "text": context.get("text", ""),
        "score": context.get("score"),
        "metadata": {
            "filename": metadata.get("filename"),
            "source_id": metadata.get("source_id"),
            "intent_code": metadata.get("intent_code"),
            "chunk_index": metadata.get("chunk_index"),
        },
    }


def build_chunk_review_rows(logs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build rows that explain which chunks helped or failed for each query."""
    rows: list[dict[str, Any]] = []
    for record in logs:
        if not isinstance(record, dict):
            continue
        review_status = record.get("review_status") or "raw"
        if review_status not in {"approved", "edited", "rejected"}:
            continue

        redacted = redact_record(record)
        contexts = [
            context
            for context in (
                _normalize_context(context) for context in redacted.get("contexts", [])
            )
            if context is not None
        ]
        rows.append(
            {
                "turn_id": redacted.get("turn_id"),
                "created_at": redacted.get("created_at"),
                "review_status": review_status,
                "response_type": redacted.get("response_type"),
                "error_type": redacted.get("error_type"),
                "feedback": redacted.get("feedback"),
                "user_query": _text(redacted, "user_query"),
                "assistant_answer": _text(redacted, "assistant_answer"),
                "corrected_answer": _text(redacted, "corrected_answer"),
                "retrieved_contexts": contexts,
                "chunking_action_hint": infer_chunking_action(redacted, contexts),
            }
        )
    return rows


def infer_chunking_action(
    record: Mapping[str, Any],
    contexts: list[dict[str, Any]],
) -> str:
    """Return a coarse action hint for improving the RAG chunk/index pipeline."""
    error_type = _text(record, "error_type").lower()
    response_type = _text(record, "response_type").lower()
    review_status = _text(record, "review_status").lower()

    if response_type == "no_data" or error_type in {"no_context", "missing_context"}:
        return "add_source_or_rechunk_missing_coverage"
    if error_type == "wrong_context":
        return "improve_chunk_metadata_or_retrieval_filtering"
    if error_type == "incomplete":
        return "split_or_merge_chunks_for_complete_answer"
    if error_type == "outdated_info":
        return "refresh_source_document_and_reindex"
    if review_status == "approved":
        return "keep_as_positive_retrieval_example"
    if review_status == "edited":
        return "compare_corrected_answer_with_retrieved_chunks"
    if not contexts:
        return "inspect_empty_retrieval"
    return "manual_chunk_review"


def write_jsonl(rows: Iterable[dict[str, Any]], output_path: Path) -> int:
    """Write rows to JSONL and return the number of rows written."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def export_chunk_review_dataset(
    output_path: Path = DEFAULT_OUTPUT_PATH,
    limit: int = 10_000,
) -> int:
    """Read Firebase logs and write the chunk review export."""
    logs = list_chat_logs(limit=limit)
    return write_jsonl(build_chunk_review_rows(logs), Path(output_path))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args(argv)

    count = export_chunk_review_dataset(output_path=args.output, limit=args.limit)
    print(f"Wrote {count} chunk review rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
