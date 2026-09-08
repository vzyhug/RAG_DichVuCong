"""Validate SFT and preference JSONL exports before they are used for training."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from collections import Counter
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

    total_samples: int = 0
    valid_samples: int = 0
    invalid_samples: int = 0
    duplicate_count: int = 0
    errors: int = 0
    warnings: int = 0
    messages: list[str] = field(default_factory=list)
    rejected_reasons: Counter[str] = field(default_factory=Counter)
    rejected_samples: list[dict[str, Any]] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        self.errors += 1
        self.messages.append(f"ERROR: {message}")

    def add_warning(self, message: str) -> None:
        self.warnings += 1
        self.messages.append(f"WARNING: {message}")

    def add_rejection(self, line_number: int, reasons: list[str]) -> None:
        """Record one invalid sample and preserve every reason for rejection."""
        self.invalid_samples += 1
        self.rejected_samples.append({"line": line_number, "reasons": reasons})
        self.rejected_reasons.update(reasons)

    def merge(self, other: "ValidationSummary") -> None:
        self.total_samples += other.total_samples
        self.valid_samples += other.valid_samples
        self.invalid_samples += other.invalid_samples
        self.duplicate_count += other.duplicate_count
        self.errors += other.errors
        self.warnings += other.warnings
        self.messages.extend(other.messages)
        self.rejected_reasons.update(other.rejected_reasons)
        self.rejected_samples.extend(other.rejected_samples)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable validation report."""
        return {
            "total_samples": self.total_samples,
            "valid_samples": self.valid_samples,
            "invalid_samples": self.invalid_samples,
            "duplicate_count": self.duplicate_count,
            "rejected_reasons": dict(self.rejected_reasons),
            "rejected_samples": self.rejected_samples,
            "warnings": self.warnings,
        }


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
    supported_roles = {"system", "user", "assistant"}
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
            normalized_role = role.strip()
            if normalized_role not in supported_roles:
                errors.append(
                    f"messages[{index}].role must be one of "
                    f"{sorted(supported_roles)}"
                )
            roles.add(normalized_role)
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
    if not errors and sample["chosen"].strip() == sample["rejected"].strip():
        errors.append("chosen and rejected must be different")
    return errors


def _sample_key(sample: Any) -> str:
    """Build a stable key for duplicate detection of parsed JSON records."""
    return json.dumps(sample, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def validate_jsonl(
    path: Path,
    dataset_type: str,
    missing_is_error: bool = True,
) -> ValidationSummary:
    """Validate JSONL syntax, structure, and common raw PII patterns."""
    summary = ValidationSummary()
    label = str(path)
    if not path.is_file():
        message = f"{label}: file does not exist"
        if missing_is_error:
            summary.add_error(message)
        else:
            summary.add_warning(message)
        return summary

    validator = (
        validate_sft_sample if dataset_type == "sft" else validate_preference_sample
    )
    seen_samples: set[str] = set()
    try:
        input_file = path.open("r", encoding="utf-8")
    except OSError as exc:
        summary.add_error(f"{label}: cannot read file ({exc})")
        return summary

    with input_file:
        for line_number, raw_line in enumerate(input_file, start=1):
            if not raw_line.strip():
                continue

            summary.total_samples += 1
            _check_pii(raw_line, label, line_number, summary)
            try:
                sample = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                reason = f"invalid JSON ({exc.msg} at column {exc.colno})"
                summary.add_error(f"{label}:{line_number}: {reason}")
                summary.add_rejection(line_number, [reason])
                continue

            sample_key = _sample_key(sample)
            if sample_key in seen_samples:
                summary.duplicate_count += 1
                summary.add_warning(f"{label}:{line_number}: duplicate record")
            seen_samples.add(sample_key)

            sample_errors = validator(sample)
            if sample_errors:
                summary.add_rejection(line_number, sample_errors)
                for error in sample_errors:
                    summary.add_error(f"{label}:{line_number}: {error}")
            else:
                summary.valid_samples += 1
    return summary


def validate_sft_file(path: Path) -> ValidationSummary:
    """Validate an SFT export."""
    return validate_jsonl(Path(path), "sft")


def validate_preference_file(
    path: Path,
    missing_is_error: bool = False,
) -> ValidationSummary:
    """Validate a preference export."""
    return validate_jsonl(Path(path), "preference", missing_is_error=missing_is_error)


def write_report(summary: ValidationSummary, path: Path) -> None:
    """Write a machine-readable validation report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


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
    parser.add_argument(
        "--require-preference",
        action="store_true",
        help="Fail if the preference dataset file is missing.",
    )
    parser.add_argument(
        "--sft-report",
        type=Path,
        default=DEFAULT_SFT_PATH.with_name("sft_dataset.validation.json"),
        help="JSON report path for the SFT validation.",
    )
    parser.add_argument(
        "--preference-report",
        type=Path,
        default=DEFAULT_PREFERENCE_PATH.with_name("preference_dataset.validation.json"),
        help="JSON report path for preference validation.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    sft_summary = validate_sft_file(args.sft)
    preference_summary = validate_preference_file(
        args.preference,
        missing_is_error=args.require_preference,
    )
    if args.sft_report:
        write_report(sft_summary, args.sft_report)
    if args.preference_report:
        write_report(preference_summary, args.preference_report)

    summary = ValidationSummary()
    summary.merge(sft_summary)
    summary.merge(preference_summary)

    for message in summary.messages:
        print(message)
    print("\nValidation summary")
    print(f"Total samples: {summary.total_samples}")
    print(f"Valid samples: {summary.valid_samples}")
    print(f"Invalid samples: {summary.invalid_samples}")
    print(f"Duplicate records: {summary.duplicate_count}")
    print(f"Errors: {summary.errors}")
    print(f"Warnings: {summary.warnings}")
    return 1 if summary.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
