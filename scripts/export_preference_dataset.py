"""Export corrected Firebase chat logs as a preference-learning dataset."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any, Iterable, MutableMapping

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
load_dotenv(ROOT_DIR / ".env")

from src.training_data.anonymizer import redact_record  # noqa: E402
from src.training_data.firebase_store import list_chat_logs  # noqa: E402


DEFAULT_OUTPUT_PATH = (
    ROOT_DIR / "data" / "training" / "exports" / "preference_dataset.jsonl"
)
DEFAULT_STATS_PATH = (
    ROOT_DIR / "data" / "training" / "exports" / "preference_dataset.stats.json"
)


def _text(record: dict[str, Any], field: str) -> str:
    value = record.get(field, "")
    return value.strip() if isinstance(value, str) else ""


@dataclass
class PreferenceBuildStats:
    """Explain which reviewed records produced preference pairs."""

    source_records: int = 0
    approved_records: int = 0
    rejected_records: int = 0
    unpaired_approved_records: int = 0
    unpaired_rejected_records: int = 0
    edited_pairs: int = 0
    approved_rejected_pairs: int = 0
    skipped_without_evidence: int = 0
    skipped_missing_fields: int = 0
    skipped_identical_answers: int = 0
    pairs_written: int = 0


def _append_pair(
    examples: list[dict[str, str]],
    seen: set[tuple[str, str, str]],
    prompt: str,
    chosen: str,
    rejected: str,
    stats: PreferenceBuildStats,
    pair_type: str,
) -> None:
    """Append an evidence-backed pair once, tracking unusable candidates."""
    if not prompt or not chosen or not rejected:
        stats.skipped_missing_fields += 1
        return
    if chosen == rejected:
        stats.skipped_identical_answers += 1
        return
    key = (prompt, chosen, rejected)
    if key in seen:
        return
    seen.add(key)
    examples.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    if pair_type == "edited":
        stats.edited_pairs += 1
    else:
        stats.approved_rejected_pairs += 1


def build_preference_examples(
    logs: Iterable[dict[str, Any]],
    stats: PreferenceBuildStats | None = None,
) -> list[dict[str, Any]]:
    """Build pairs only from explicit edited or approved/rejected reviews.

    Ratings are deliberately ignored because they do not identify a preferred
    response or establish a pair by themselves.
    """
    build_stats = stats or PreferenceBuildStats()
    examples: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    reviewed_by_prompt: MutableMapping[str, dict[str, set[str]]] = defaultdict(
        lambda: {"approved": set(), "rejected": set()}
    )

    for record in logs:
        build_stats.source_records += 1
        if not isinstance(record, dict):
            build_stats.skipped_missing_fields += 1
            continue

        redacted = redact_record(record)
        prompt = _text(redacted, "user_query")
        status = redacted.get("review_status")
        assistant_answer = _text(redacted, "assistant_answer")

        if status == "edited":
            _append_pair(
                examples,
                seen,
                prompt,
                _text(redacted, "corrected_answer"),
                assistant_answer,
                build_stats,
                "edited",
            )
            continue

        if status == "approved":
            approved_answer = _text(redacted, "corrected_answer") or assistant_answer
            if prompt and approved_answer:
                build_stats.approved_records += 1
                reviewed_by_prompt[prompt]["approved"].add(approved_answer)
            else:
                build_stats.skipped_missing_fields += 1
            continue

        if status == "rejected":
            if prompt and assistant_answer:
                build_stats.rejected_records += 1
                reviewed_by_prompt[prompt]["rejected"].add(assistant_answer)
            else:
                build_stats.skipped_missing_fields += 1
            continue

        build_stats.skipped_without_evidence += 1

    for prompt, answers in reviewed_by_prompt.items():
        if answers["approved"] and not answers["rejected"]:
            build_stats.unpaired_approved_records += len(answers["approved"])
        if answers["rejected"] and not answers["approved"]:
            build_stats.unpaired_rejected_records += len(answers["rejected"])
        for chosen in sorted(answers["approved"]):
            for rejected in sorted(answers["rejected"]):
                _append_pair(
                    examples,
                    seen,
                    prompt,
                    chosen,
                    rejected,
                    build_stats,
                    "approved_rejected",
                )

    build_stats.pairs_written = len(examples)
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


def export_preference_dataset(
    output_path: Path = DEFAULT_OUTPUT_PATH,
    limit: int = 10_000,
    stats_path: Path = DEFAULT_STATS_PATH,
) -> int:
    """Read Firebase logs and write the preference export."""
    logs = list_chat_logs(limit=limit)
    stats = PreferenceBuildStats()
    count = write_jsonl(
        build_preference_examples(logs, stats=stats),
        Path(output_path),
    )
    Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
    Path(stats_path).write_text(
        json.dumps(asdict(stats), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS_PATH)
    args = parser.parse_args(argv)

    count = export_preference_dataset(
        output_path=args.output,
        limit=args.limit,
        stats_path=args.stats,
    )
    print(f"Wrote {count} preference examples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
