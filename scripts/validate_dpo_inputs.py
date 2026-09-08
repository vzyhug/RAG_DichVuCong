"""Validate the non-empty preference input required before a DPO run."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.validate_training_exports import validate_preference_file  # noqa: E402


DEFAULT_PREFERENCE_PATH = (
    ROOT_DIR / "data" / "training" / "exports" / "preference_dataset.jsonl"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preference", type=Path, default=DEFAULT_PREFERENCE_PATH)
    args = parser.parse_args(argv)

    summary = validate_preference_file(args.preference, missing_is_error=True)
    if summary.errors:
        for message in summary.messages:
            print(message, file=sys.stderr)
        return 1
    if summary.valid_samples == 0:
        print(
            f"ERROR: {args.preference} contains no valid preference rows. "
            "Add reviewed chosen/rejected pairs before DPO training.",
            file=sys.stderr,
        )
        return 1

    print(f"DPO input is valid: {summary.valid_samples} preference rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
