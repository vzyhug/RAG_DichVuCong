"""Redact common personally identifiable information from training records."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


_EMAIL_RE = re.compile(
    r"(?<![\w.+-])[\w.!#$%&'*+/=?^`{|}~-]+@"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+(?![\w-])"
)

# Vietnamese mobile prefixes, with support for +84 and common separators.
_MOBILE_PREFIX = r"(?:3|5|7|8|9)"
_PHONE_RE = re.compile(
    rf"(?<![A-Za-z0-9])(?:"
    rf"(?:\+?84[ .-]?{_MOBILE_PREFIX}(?:[ .-]?\d){{8}})"
    rf"|(?:0{_MOBILE_PREFIX}(?:[ .-]?\d){{8}})"
    # Domestic landlines have a 02 area-code prefix and 10 or 11 digits.
    rf"|(?:02\d(?:[ .-]?\d){{8,9}})"
    rf")(?![A-Za-z0-9])"
)

# Case/file IDs are intentionally restricted to a single token.  Requiring
# both letters and digits prevents ordinary words from being redacted.
_CASE_ID_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?=[A-Za-z0-9._/-]{8,}(?![A-Za-z0-9]))"
    r"(?=[A-Za-z0-9._/-]*[A-Za-z])"
    r"(?=[A-Za-z0-9._/-]*\d)"
    r"[A-Za-z0-9](?:[A-Za-z0-9._/-]*[A-Za-z0-9])?"
    r"(?![A-Za-z0-9])"
)

_ID_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:\d[\s.-]?){9,12}(?![A-Za-z0-9])"
)


def _redact_id_number(match: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", match.group(0))
    return "[ID_NUMBER]" if 9 <= len(digits) <= 12 else match.group(0)


def redact_pii(text: str) -> str:
    """Replace common Vietnamese PII patterns with stable placeholders."""
    if not isinstance(text, str):
        return text

    redacted = _EMAIL_RE.sub("[EMAIL]", text)
    redacted = _PHONE_RE.sub("[PHONE]", redacted)
    redacted = _CASE_ID_RE.sub("[CASE_ID]", redacted)
    return _ID_NUMBER_RE.sub(_redact_id_number, redacted)


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_pii(value)
    if isinstance(value, Mapping):
        return {key: _redact_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(item) for item in value)
    return value


def redact_record(record: dict) -> dict:
    """Return a redacted copy of a record without mutating the input."""
    if not isinstance(record, dict):
        raise TypeError("record must be a dict")
    return _redact_value(record)
