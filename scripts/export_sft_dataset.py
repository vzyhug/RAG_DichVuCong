"""Export reviewed Firebase chat logs as a supervised fine-tuning dataset.

The exporter follows the current Firestore schema: ``user_query`` is the
question, ``assistant_answer`` is the generated answer, ``feedback.rating`` is
the user rating, ``review_status`` stores raw/approved/rejected/edited, and
``corrected_answer`` stores a human edit. Ratings are retained as review
metadata but do not approve a sample by themselves.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
load_dotenv(ROOT_DIR / ".env")

from src.training_data.anonymizer import redact_record  # noqa: E402
from src.training_data.firebase_store import list_chat_logs  # noqa: E402


DEFAULT_OUTPUT_PATH = ROOT_DIR / "data" / "training" / "exports" / "sft_dataset.jsonl"
SYSTEM_MESSAGE = (
    "You are the official virtual assistant for An Vien commune police. "
    "Answer citizens accurately using the provided legal and administrative knowledge."
)
_EXPORTABLE_REVIEW_STATUSES = {"approved", "edited"}


def _text(record: dict[str, Any], field: str) -> str:
    value = record.get(field, "")
    return value.strip() if isinstance(value, str) else ""


def _record_sort_key(record: dict[str, Any]) -> tuple[str, ...]:
    """Return stable ordering keys for records returned in arbitrary Firestore order."""
    return (
        _text(record, "created_at"),
        _text(record, "turn_id"),
        _text(record, "document_id"),
        _text(record, "user_query"),
        _text(record, "assistant_answer"),
        _text(record, "corrected_answer"),
        json.dumps(record, ensure_ascii=False, sort_keys=True, default=str),
    )


def build_sft_examples(logs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build deterministic SFT rows from reviewed Firebase records.

    The review implementation stores approval/rejection as ``review_status``.
    An ``edited`` record is usable only when its ``corrected_answer`` is
    present; an ``approved`` record uses its original ``assistant_answer``.
    Raw, rejected, malformed, and empty records are excluded.
    """
    examples: list[dict[str, Any]] = []
    records = sorted(
        (record for record in logs if isinstance(record, dict)),
        key=_record_sort_key,
    )
    for record in records:
        review_status = record.get("review_status")
        if review_status not in _EXPORTABLE_REVIEW_STATUSES:
            continue

        redacted = redact_record(record)
        answer_field = "corrected_answer" if review_status == "edited" else "assistant_answer"
        user_query = _text(redacted, "user_query")
        assistant_answer = _text(redacted, answer_field)
        if not user_query or not assistant_answer:
            continue

        examples.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": user_query},
                    {"role": "assistant", "content": assistant_answer},
                ]
            }
        )
    return examples


def write_jsonl(rows: Iterable[dict[str, Any]], output_path: Path) -> int:
    """Write rows to JSONL and return the number of rows written."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def export_sft_dataset(
    output_path: Path = DEFAULT_OUTPUT_PATH,
    limit: int = 10_000,
) -> int:
    """Read Firebase logs and write the SFT export."""
    logs = list_chat_logs(limit=limit)
    return write_jsonl(build_sft_examples(logs), Path(output_path))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args(argv)

    count = export_sft_dataset(output_path=args.output, limit=args.limit)
    print(f"Wrote {count} SFT examples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
