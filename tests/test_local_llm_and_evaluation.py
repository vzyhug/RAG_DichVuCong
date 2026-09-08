import asyncio
import json
from pathlib import Path

from configs.settings import settings
from scripts.evaluate_base_vs_sft import assert_held_out, main as evaluate_main
from scripts.evaluate_base_vs_sft import score_response
from src.llm.model_factory import LLMFactory
from src.llm.prompt_templates import build_prompt
from src.llm.service import LLMRequestError, LLMService


class _Delta:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.delta = _Delta(content)


class _Chunk:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)

        async def stream():
            yield _Chunk("answer")

        return stream()


class _Client:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": _Completions()})()


def test_local_factory_selects_configured_model_and_endpoint(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "local")
    monkeypatch.setattr(settings, "LOCAL_LLM_URL", "http://localhost:11434/v1/")
    monkeypatch.setattr(settings, "LOCAL_LLM_MODEL", "qwen2.5:7b-instruct")

    client = LLMFactory.get_llm()

    assert str(client.base_url).rstrip("/").endswith("localhost:11434/v1")
    assert LLMFactory.get_model_name() == "qwen2.5:7b-instruct"


def test_service_sends_prompt_with_retrieved_context(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "local")
    monkeypatch.setattr(settings, "LOCAL_LLM_MODEL", "qwen2.5:7b-instruct")
    client = _Client()
    service = LLMService(client=client)
    prompt = build_prompt("retrieved RAG context", "citizen question")

    values = asyncio.run(_collect(service.stream_chat([{"role": "user", "content": prompt}])))

    assert values == ["answer"]
    request = client.chat.completions.calls[0]
    assert request["model"] == "qwen2.5:7b-instruct"
    assert "retrieved RAG context" in request["messages"][0]["content"]


async def _collect(stream):
    return [value async for value in stream]


def test_service_exposes_provider_diagnostics_on_connection_error(monkeypatch):
    class FailingCompletions:
        async def create(self, **kwargs):
            raise ConnectionError("connection refused")

    monkeypatch.setattr(settings, "LLM_PROVIDER", "local")
    monkeypatch.setattr(settings, "LOCAL_LLM_URL", "http://localhost:11434/v1")
    monkeypatch.setattr(settings, "LOCAL_LLM_MODEL", "qwen2.5:7b-instruct")
    client = type(
        "Client",
        (),
        {"chat": type("Chat", (), {"completions": FailingCompletions()})()},
    )()

    try:
        asyncio.run(_collect(LLMService(client=client).stream_chat([])))
    except LLMRequestError as error:
        assert "provider=local" in str(error)
        assert "qwen2.5:7b-instruct" in str(error)
        assert "localhost:11434/v1" in str(error)
    else:
        raise AssertionError("Expected LLMRequestError")


def test_evaluation_scores_refusal_and_keeps_hallucination_signal():
    metrics = score_response(
        "Xin lỗi, chưa có thông tin trong dữ liệu để xác định.",
        {"category": "no_data", "expected_points": []},
        "",
        ["Xin lỗi, chưa có thông tin trong dữ liệu để xác định."],
    )

    assert metrics["refusal_expected"] is True
    assert metrics["refusal_behavior"] == 1.0
    assert metrics["hallucination_tendency"] == 0.0


def test_evaluation_does_not_write_results_without_sft_model(tmp_path):
    config_path = tmp_path / "evaluation.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset": "data/eval/eval_questions.jsonl",
                "base": {"model": "base", "base_url": "http://localhost:1/v1"},
                "sft": {"model": None, "base_url": "http://localhost:1/v1"},
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "results.jsonl"

    exit_code = evaluate_main(
        ["--config", str(config_path), "--output", str(output_path)]
    )

    assert exit_code == 2
    assert not output_path.exists()


def test_evaluation_holdout_guard_reads_dpo_preference_prompts(tmp_path):
    preference_path = tmp_path / "preference.jsonl"
    preference_path.write_text(
        json.dumps(
            {
                "prompt": "held out question",
                "chosen": "grounded",
                "rejected": "unsupported",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        assert_held_out(
            [{"question": "held out question"}],
            [preference_path],
        )
    except ValueError as error:
        assert "overlaps training data" in str(error)
    else:
        raise AssertionError("Expected DPO preference overlap to be rejected")
