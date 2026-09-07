"""Validate SFT and preference JSONL exports before they are used for training."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Iterable


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SFT_PATH = ROOT_DIR / "data" / "training" / "exports" / "sft_dataset.jsonl"
DEFAULT_PREFERENCE_PATH = (
    ROOT_DIR / "data" / "training" / "exports" / "preference_dataset.jsonl"
)

_EMAIL_RE = re.compile(
    r"(?<![\w.+-])[\w.!#$%&'*+/=?^`{|}~-]+@"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+(?![\w-])"
)
_PHONE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"(?:\+?84[ .-]?(?:3|5|7|8|9)(?:[ .-]?\d){8})"
    r"|(?:0(?:3|5|7|8|9)(?:[ .-]?\d){8})"
    r"|(?:02\d(?:[ .-]?\d){8,9})"
    r")(?![A-Za-z0-9])"
)
# This intentionally errs on the side of warning: CCCD/CMND values can be
# written with spaces or separators, and exports should be reviewed manually.
_IDENTITY_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:\d[\s.-]?){9,12}(?![A-Za-z0-9])"
)


@dataclass
class ValidationSummary:
    """Counts and messages produced while validating one or more files."""

    valid_samples: int = 0
    errors: int = 0
    warnings: int = 0
    messages: list[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        self.errors += 1
        self.messages.append(f"ERROR: {message}")

    def add_warning(self, message: str) -> None:
        self.warnings += 1
        self.messages.append(f"WARNING: {message}")

    def merge(self, other: "ValidationSummary") -> None:
        self.valid_samples += other.valid_samples
        self.errors += other.errors
        self.warnings += other.warnings
        self.messages.extend(other.messages)


def _check_pii(text: str, label: str, line_number: int, summary: ValidationSummary) -> None:
    """Warn once per PII pattern found in a JSONL row."""
    checks = (
        (_EMAIL_RE, "email address"),
        (_PHONE_RE, "phone number"),
        (_IDENTITY_NUMBER_RE, "possible CCCD/CMND number"),
    )
    for pattern, description in checks:
        if pattern.search(text):
            summary.add_warning(f"{label}:{line_number} contains a raw {description}")


def validate_sft_sample(sample: Any) -> list[str]:
    """Return structural errors for one SFT sample."""
    errors: list[str] = []
    if not isinstance(sample, dict):
        return ["sample must be a JSON object"]

    messages = sample.get("messages")
    if not isinstance(messages, list) or not messages:
        return ["messages must be a non-empty list"]

    roles: set[str] = set()
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            errors.append(f"messages[{index}] must be an object")
            continue
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role.strip():
            errors.append(f"messages[{index}].role must be a non-empty string")
        else:
            roles.add(role.strip())
        if not isinstance(content, str) or not content.strip():
            errors.append(f"messages[{index}].content must be a non-empty string")

    for required_role in ("user", "assistant"):
        if required_role not in roles:
            errors.append(f"messages must contain a {required_role} role")
    return errors


def validate_preference_sample(sample: Any) -> list[str]:
    """Return structural errors for one preference sample."""
    if not isinstance(sample, dict):
        return ["sample must be a JSON object"]

    errors: list[str] = []
    for field_name in ("prompt", "chosen", "rejected"):
        value = sample.get(field_name)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{field_name} must be a non-empty string")
    return errors


def validate_jsonl(path: Path, dataset_type: str) -> ValidationSummary:
    """Validate JSONL syntax, structure, and common raw PII patterns."""
    summary = ValidationSummary()
    label = str(path)
    if not path.is_file():
        summary.add_error(f"{label}: file does not exist")
        return summary

    validator = (
        validate_sft_sample if dataset_type == "sft" else validate_preference_sample
    )
    try:
        input_file = path.open("r", encoding="utf-8")
    except OSError as exc:
        summary.add_error(f"{label}: cannot read file ({exc})")
        return summary

    with input_file:
        for line_number, raw_line in enumerate(input_file, start=1):
            if not raw_line.strip():
                continue

            _check_pii(raw_line, label, line_number, summary)
            try:
                sample = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                summary.add_error(
                    f"{label}:{line_number}: invalid JSON ({exc.msg} at column {exc.colno})"
                )
                continue

            sample_errors = validator(sample)
            if sample_errors:
                for error in sample_errors:
                    summary.add_error(f"{label}:{line_number}: {error}")
            else:
                summary.valid_samples += 1
    return summary


def validate_sft_file(path: Path) -> ValidationSummary:
    """Validate an SFT export."""
    return validate_jsonl(Path(path), "sft")


def validate_preference_file(path: Path) -> ValidationSummary:
    """Validate a preference export."""
    return validate_jsonl(Path(path), "preference")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sft",
        type=Path,
        default=DEFAULT_SFT_PATH,
        help=f"SFT JSONL path (default: {DEFAULT_SFT_PATH})",
    )
    parser.add_argument(
        "--preference",
        type=Path,
        default=DEFAULT_PREFERENCE_PATH,
        help=f"Preference JSONL path (default: {DEFAULT_PREFERENCE_PATH})",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    summary = ValidationSummary()
    summary.merge(validate_sft_file(args.sft))
    summary.merge(validate_preference_file(args.preference))

    for message in summary.messages:
        print(message)
    print("\nValidation summary")
    print(f"Valid samples: {summary.valid_samples}")
    print(f"Errors: {summary.errors}")
    print(f"Warnings: {summary.warnings}")
    return 1 if summary.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
