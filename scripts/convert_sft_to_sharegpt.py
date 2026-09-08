"""Convert validated OpenAI Messages SFT JSONL to ShareGPT JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.validate_training_exports import (  # noqa: E402
    validate_sft_sample,
    validate_jsonl,
    write_report,
)


ROLE_MAP = {"system": "system", "user": "human", "assistant": "gpt"}
DEFAULT_INPUT_PATH = ROOT_DIR / "data" / "training" / "exports" / "sft_dataset.jsonl"
DEFAULT_OUTPUT_PATH = (
    ROOT_DIR / "data" / "training" / "exports" / "sft_dataset_sharegpt.jsonl"
)


def to_sharegpt_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Convert one validated OpenAI Messages sample."""
    errors = validate_sft_sample(sample)
    if errors:
        raise ValueError("Invalid SFT sample: " + "; ".join(errors))
    return {
        "conversations": [
            {"from": ROLE_MAP[message["role"].strip()], "value": message["content"]}
            for message in sample["messages"]
        ]
    }


def convert_sft_dataset(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    report_path: Path | None = None,
) -> int:
    """Validate and convert an SFT file without discarding bad records."""
    summary = validate_jsonl(Path(input_path), "sft")
    if report_path is not None:
        write_report(summary, Path(report_path))
    if summary.errors:
        raise ValueError(
            f"Refusing to convert {input_path}: "
            f"{summary.invalid_samples} invalid sample(s) found"
        )

    converted: list[dict[str, Any]] = []
    with Path(input_path).open("r", encoding="utf-8") as input_file:
        for line_number, raw_line in enumerate(input_file, start=1):
            if not raw_line.strip():
                continue
            sample = json.loads(raw_line)
            try:
                converted.append(to_sharegpt_sample(sample))
            except ValueError as exc:
                raise ValueError(f"Invalid SFT sample at line {line_number}: {exc}") from exc

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as output_file:
        for sample in converted:
            output_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return len(converted)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--validation-report",
        type=Path,
        default=DEFAULT_INPUT_PATH.with_name("sft_dataset.validation.json"),
        help="JSON validation report path.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        count = convert_sft_dataset(
            input_path=args.input,
            output_path=args.output,
            report_path=args.validation_report,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Conversion failed: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {count} ShareGPT examples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
