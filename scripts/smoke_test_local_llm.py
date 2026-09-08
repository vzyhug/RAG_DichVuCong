"""Run one independent smoke test against the configured local LLM provider."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import inspect
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any
from urllib.parse import urlparse


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DEFAULT_PROMPT = "Reply with exactly: LOCAL_LLM_OK"


class MalformedAPIResponseError(RuntimeError):
    """Raised when the provider returns no usable chat completion content."""


@dataclass
class DiagnosticFailure:
    category: str
    message: str
    hint: str


def _is_model_error(message: str) -> bool:
    text = message.lower()
    return "model" in text and any(
        marker in text
        for marker in ("not found", "does not exist", "unknown", "invalid")
    )


def classify_error(exc: BaseException) -> DiagnosticFailure:
    """Map common OpenAI-compatible client failures to actionable diagnostics."""
    name = type(exc).__name__
    message = str(exc).strip() or repr(exc)
    status_code = getattr(exc, "status_code", None)
    related_errors = [message]
    for related in (getattr(exc, "__cause__", None), getattr(exc, "__context__", None)):
        if related is not None:
            related_errors.append(str(related))
    lowered = " ".join(related_errors).lower()

    if name in {"APITimeoutError", "TimeoutError", "ReadTimeout", "ConnectTimeout"}:
        return DiagnosticFailure(
            "timeout",
            message,
            "The local server did not respond before the timeout. Check that it is running and increase --timeout if needed.",
        )

    if name in {"APIConnectionError", "ConnectError", "ConnectionError"}:
        if "refused" in lowered:
            hint = "Connection was refused. Start the configured local LLM server, then rerun this smoke test."
        else:
            hint = "The endpoint could not be reached. Check the host, port, firewall, and server status."
        return DiagnosticFailure("connection_failure", message, hint)

    if status_code == 404:
        if _is_model_error(message):
            return DiagnosticFailure(
                "model_not_found",
                message,
                "The requested model is not loaded or does not exist on the local server. Check LOCAL_LLM_MODEL.",
            )
        return DiagnosticFailure(
            "incorrect_endpoint",
            message,
            "The server responded with 404. Check that LOCAL_LLM_URL points to the OpenAI-compatible /v1 endpoint.",
        )

    if status_code in {401, 403}:
        return DiagnosticFailure(
            "invalid_configuration",
            message,
            "The local server rejected authentication or permissions. Check the endpoint and any required API key configuration.",
        )

    if status_code == 400 and _is_model_error(message):
        return DiagnosticFailure(
            "model_not_found",
            message,
            "The local server rejected the requested model. Check LOCAL_LLM_MODEL and the server's model list.",
        )

    if name in {
        "APIResponseValidationError",
        "JSONDecodeError",
        "MalformedAPIResponseError",
    } or "json" in lowered and "response" in lowered:
        return DiagnosticFailure(
            "malformed_api_response",
            message,
            "The endpoint did not return a valid OpenAI-compatible response. Check the server API mode and response format.",
        )

    if name in {"ModuleNotFoundError", "ValueError", "TypeError", "KeyError"}:
        return DiagnosticFailure(
            "invalid_configuration",
            message,
            "Check LLM_PROVIDER, LOCAL_LLM_URL, LOCAL_LLM_MODEL, and the installed project dependencies.",
        )

    return DiagnosticFailure(
        "request_failure",
        message,
        "Inspect the local server logs and verify that it supports chat.completions.create.",
    )


def _validate_endpoint(endpoint: Any) -> str:
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("LOCAL_LLM_URL is empty")
    parsed = urlparse(endpoint.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            "LOCAL_LLM_URL must be an absolute http(s) URL, for example http://localhost:11434/v1"
        )
    return endpoint.strip()


def _extract_response_text(response: Any) -> str:
    """Read the first completion while explicitly rejecting malformed payloads."""
    choices = getattr(response, "choices", None)
    if not isinstance(choices, list) or not choices:
        raise MalformedAPIResponseError("response.choices is missing or empty")

    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise MalformedAPIResponseError(
            "response.choices[0].message.content is missing or empty"
        )
    return content.strip()


async def _close_client(client: Any) -> None:
    close = getattr(client, "close", None)
    if close is None:
        return
    result = close()
    if inspect.isawaitable(result):
        await result


async def run_smoke_test(prompt: str, timeout: float) -> int:
    """Load project settings, call LLMFactory, and print a diagnostic report."""
    settings = None
    client = None
    started = perf_counter()

    try:
        from configs.settings import settings as loaded_settings
        from src.llm.model_factory import LLMFactory

        settings = loaded_settings
        provider = getattr(settings, "LLM_PROVIDER", "<missing>")
        model = getattr(settings, "LOCAL_LLM_MODEL", "<missing>")
        endpoint = getattr(settings, "LOCAL_LLM_URL", "<missing>")
        print("Local LLM smoke test")
        print(f"provider: {provider}")
        print(f"model: {model or '<missing>'}")
        print(f"endpoint: {endpoint or '<missing>'}")

        if provider != "local":
            raise ValueError(
                f"LLM_PROVIDER={provider!r}; this diagnostic requires LLM_PROVIDER=local"
            )
        endpoint = _validate_endpoint(endpoint)
        if not isinstance(model, str) or not model.strip():
            raise ValueError("LOCAL_LLM_MODEL is empty")
        settings.validate()

        client = LLMFactory.get_llm()
        request_client = client
        with_options = getattr(client, "with_options", None)
        if with_options is not None:
            # Disable the OpenAI client's default retries so --timeout is predictable.
            request_client = with_options(timeout=timeout, max_retries=0)
        response = await request_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=16,
        )
        result = _extract_response_text(response)
        latency_ms = (perf_counter() - started) * 1000
        print("connection: success")
        print("status: success")
        print(f"response: {result}")
        print(f"latency_ms: {latency_ms:.1f}")
        return 0
    except Exception as exc:  # The CLI must fail gracefully for offline diagnostics.
        failure = classify_error(exc)
        if settings is None:
            print("Local LLM smoke test")
            print(f"provider: {os.getenv('LLM_PROVIDER', '<unavailable>')}")
            print(f"model: {os.getenv('LOCAL_LLM_MODEL', '<unavailable>')}")
            print(f"endpoint: {os.getenv('LOCAL_LLM_URL', '<unavailable>')}")
        print("connection: failure")
        print("status: failure")
        print(f"error_category: {failure.category}")
        print(f"error: {failure.message}")
        print(f"hint: {failure.hint}")
        return 1
    finally:
        if client is not None:
            try:
                await _close_client(client)
            except Exception as exc:
                print(f"warning: failed to close LLM client: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Minimal prompt sent to the local provider.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="Request timeout in seconds (default: 15).",
    )
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    return asyncio.run(run_smoke_test(args.prompt, args.timeout))


if __name__ == "__main__":
    raise SystemExit(main())
