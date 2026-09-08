"""Evaluate Base, SFT, and optional DPO checkpoints on the same RAG prompts.

The input JSONL under ``data/eval`` is deliberately separate from the SFT
exports. Retrieval is performed once per question, and the resulting context
and fully rendered prompt are sent unchanged to every configured model. The
runner produces raw responses plus heuristic, machine-readable metrics and a
comparison report; it does not assume that fine-tuning improves the model.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import unicodedata
from typing import Any, Iterable

from openai import AsyncOpenAI


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.llm.prompt_templates import build_prompt  # noqa: E402
from src.rag_flow.context_retriever import ContextRetriever  # noqa: E402


DEFAULT_CONFIG = ROOT_DIR / "configs" / "evaluation.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "evaluation"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "combined_results.jsonl"
DEFAULT_SUMMARY = DEFAULT_OUTPUT_DIR / "summary.json"
DEFAULT_REPORT = DEFAULT_OUTPUT_DIR / "comparison_report.md"

_TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)
_NUMBER_RE = re.compile(r"(?<!\w)\d+(?:[.,/]\d+)*(?!\w)")
_REFUSAL_PHRASES = (
    "không có thông tin",
    "chưa có thông tin",
    "không đủ dữ liệu",
    "không thể xác định",
    "chưa được cung cấp",
    "liên hệ cơ quan có thẩm quyền",
)
_STOPWORDS = {
    "và", "là", "có", "cho", "của", "một", "các", "những", "để", "trong",
    "khi", "với", "về", "người", "tôi", "cần", "nêu", "phải", "hoặc", "theo",
    "thì", "được", "này", "ra", "sao", "gì", "nào", "hướng", "dẫn",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model: str
    base_url: str | None
    api_key: str


MODEL_OUTPUT_NAMES = {
    "base": "base_results.jsonl",
    "sft": "sft_results.jsonl",
    "dpo": "dpo_results.jsonl",
}


def _tokens(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFC", text.lower())
    return {
        token
        for token in _TOKEN_RE.findall(normalized)
        if token not in _STOPWORDS and len(token) > 1
    }


def _accentless(text: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFD", text.lower())
        if unicodedata.category(character) != "Mn"
    )


def _context_tokens(context_text: str) -> set[str]:
    return _tokens(context_text)


def _expected_point_coverage(answer: str, expected_points: Iterable[str]) -> float:
    answer_tokens = _tokens(answer)
    points = list(expected_points)
    if not points:
        return 1.0
    covered = 0
    for point in points:
        point_tokens = _tokens(point)
        if point_tokens and len(point_tokens & answer_tokens) / len(point_tokens) >= 0.2:
            covered += 1
    return covered / len(points)


def _response_structure_score(answer: str) -> float:
    if not answer.strip():
        return 0.0
    features = (
        bool(re.search(r"(?m)^\s{0,3}#{1,6}\s+\S+", answer)),
        bool(re.search(r"(?m)^\s*(?:[-*+] |\d+[.)] )", answer)),
        len(answer.split()) >= 20,
    )
    return sum(features) / len(features)


def _formatting_score(answer: str) -> float:
    if not answer.strip():
        return 0.0
    defects = 0
    if answer.count("```") % 2:
        defects += 1
    if answer.count("**") % 2:
        defects += 1
    if re.search(r"(?m)^\s*[-*+]\s*$", answer):
        defects += 1
    return max(0.0, 1.0 - defects / 3)


def _context_adherence_score(answer: str, context_text: str) -> float:
    answer_tokens = _tokens(answer)
    context_tokens = _context_tokens(context_text)
    if not answer_tokens:
        return 0.0
    if not context_tokens:
        return 1.0 if _has_refusal(answer) else 0.0
    return min(1.0, len(answer_tokens & context_tokens) / max(1, len(answer_tokens) * 0.35))


def _context_ignore_tendency(answer: str, context_text: str) -> float:
    """Return a risk signal where 1 means the supplied context was ignored."""
    if not context_text.strip():
        return 0.0 if _has_refusal(answer) else 1.0
    return 1.0 - _context_adherence_score(answer, context_text)


def _hallucination_tendency(answer: str, context_text: str) -> float:
    """Estimate unsupported numeric claims; this is a risk signal, not a judge."""
    numbers = set(_NUMBER_RE.findall(answer))
    if not numbers:
        return 0.0
    context_numbers = set(_NUMBER_RE.findall(context_text))
    unsupported = numbers - context_numbers
    return len(unsupported) / len(numbers)


def _has_refusal(answer: str) -> bool:
    normalized = _accentless(answer)
    return any(_accentless(phrase) in normalized for phrase in _REFUSAL_PHRASES)


def _stability_score(responses: list[str]) -> float:
    nonempty = [response.strip() for response in responses if response.strip()]
    if not nonempty:
        return 0.0
    return sum(response == nonempty[0] for response in nonempty) / len(responses)


def _deployment_score(metrics: dict[str, float | bool]) -> float:
    """Weighted screening score used only to rank comparable responses.

    This is deliberately exposed in every result record so report examples can
    be traced back to per-case evidence. It is not a substitute for review.
    """
    positive_weights = {
        "vietnamese_response_quality": 0.15,
        "instruction_following": 0.20,
        "response_structure": 0.10,
        "formatting_consistency": 0.10,
        "contextual_adherence": 0.15,
        "rag_context_adherence": 0.10,
        "answer_stability": 0.10,
        "refusal_behavior": 0.10,
    }
    score = sum(
        float(metrics.get(name, 0.0)) * weight
        for name, weight in positive_weights.items()
    )
    score -= float(metrics.get("hallucination_tendency", 0.0)) * 0.10
    score -= float(metrics.get("context_ignore_tendency", 0.0)) * 0.10
    return max(0.0, min(1.0, score))


def _qualitative_evaluation(
    case: dict[str, Any], metrics: dict[str, float | bool]
) -> dict[str, Any]:
    """Store a structured rubric record without inventing human judgments."""
    feedback = case.get("reviewed_feedback")
    feedback_status = "available" if isinstance(feedback, dict) else "not_available"
    return {
        "status": "heuristic_screening_pending_human_review",
        "rubric": [
            "Vietnamese language quality",
            "instruction following and response format",
            "contextual and RAG-context adherence",
            "unsupported claims and context ignoring",
            "consistency across repeated runs",
            "behavior preferred by reviewed user feedback",
        ],
        "reviewed_feedback_alignment": {
            "status": feedback_status,
            "evidence": feedback if isinstance(feedback, dict) else {},
        },
        "evidence": {
            "deployment_screening_score": metrics["deployment_screening_score"],
            "instruction_following": metrics["instruction_following"],
            "rag_context_adherence": metrics["rag_context_adherence"],
            "context_ignore_tendency": metrics["context_ignore_tendency"],
            "unsupported_claim_rate": metrics["unsupported_claim_rate"],
            "answer_stability": metrics["answer_stability"],
        },
    }


def score_response(
    answer: str,
    case: dict[str, Any],
    context_text: str,
    responses: list[str],
) -> dict[str, float | bool]:
    """Return comparable heuristic scores for one model/question pair."""
    category = str(case.get("category", "")).lower()
    is_refusal_case = category == "no_data" or not context_text.strip()
    vietnamese_letters = sum(
        character.lower() in "àáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ"
        for character in answer
    )
    letters = sum(character.isalpha() for character in answer)
    vietnamese_quality = 0.0 if not answer.strip() else min(
        1.0,
        0.5 + (vietnamese_letters / max(1, letters)) * 0.5,
    )
    refusal = _has_refusal(answer)
    metrics: dict[str, float | bool] = {
        "vietnamese_response_quality": vietnamese_quality,
        "instruction_following": _expected_point_coverage(
            answer, case.get("expected_points", [])
        ),
        "response_structure": _response_structure_score(answer),
        "formatting_consistency": _formatting_score(answer),
        "contextual_adherence": _context_adherence_score(answer, context_text),
        "rag_context_adherence": _context_adherence_score(answer, context_text),
        "hallucination_tendency": _hallucination_tendency(answer, context_text),
        "unsupported_claim_rate": _hallucination_tendency(answer, context_text),
        "context_ignore_tendency": _context_ignore_tendency(answer, context_text),
        "answer_stability": _stability_score(responses),
        "refusal_expected": is_refusal_case,
        "refusal_behavior": float(refusal == is_refusal_case),
    }
    metrics["deployment_screening_score"] = _deployment_score(metrics)
    return metrics


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, raw_line in enumerate(input_file, start=1):
            if not raw_line.strip():
                continue
            value = json.loads(raw_line)
            if not isinstance(value, dict) or not str(value.get("question", "")).strip():
                raise ValueError(f"{path}:{line_number} must contain a question")
            cases.append(value)
    if not cases:
        raise ValueError(f"Evaluation dataset is empty: {path}")
    return cases


def _normalized_question(question: str) -> str:
    return " ".join(unicodedata.normalize("NFC", question).split()).casefold()


def _training_questions(path: Path) -> set[str]:
    questions: set[str] = set()
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, raw_line in enumerate(input_file, start=1):
            if not raw_line.strip():
                continue
            sample = json.loads(raw_line)
            if not isinstance(sample, dict):
                continue
            preference_prompt = sample.get("prompt")
            if isinstance(preference_prompt, str) and preference_prompt.strip():
                questions.add(_normalized_question(preference_prompt))
                continue
            messages = sample.get("messages") or sample.get("conversations") or []
            for message in messages:
                if not isinstance(message, dict):
                    continue
                role = message.get("role") or message.get("from")
                if role in {"user", "human"}:
                    content = message.get("content") or message.get("value")
                    if isinstance(content, str) and content.strip():
                        questions.add(_normalized_question(content))
                    break
            else:
                raise ValueError(
                    f"{path}:{line_number} has no user message for hold-out validation"
                )
    return questions


def assert_held_out(cases: list[dict[str, Any]], training_paths: Iterable[Path]) -> None:
    """Fail closed if an evaluation question is present in training exports."""
    evaluation_questions = {
        _normalized_question(str(case["question"])) for case in cases
    }
    for training_path in training_paths:
        if not training_path.is_file():
            raise ValueError(
                f"Configured training data for hold-out validation does not exist: "
                f"{training_path}"
            )
        overlap = evaluation_questions & _training_questions(training_path)
        if overlap:
            raise ValueError(
                f"Evaluation dataset overlaps training data in {training_path}: "
                f"{len(overlap)} question(s)"
            )


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Evaluation config must be a JSON object: {path}")
    return config


def _model_spec(
    config: dict[str, Any],
    name: str,
    model_override: str | None,
    base_url_override: str | None,
) -> ModelSpec:
    section = config.get(name) or {}
    model = model_override or section.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(
            f"No {name} model is configured. Set {name}.model in the evaluation "
            "config or pass the corresponding CLI option."
        )
    base_url = base_url_override or section.get("base_url")
    api_key_env = section.get("api_key_env") or "OPENAI_API_KEY"
    api_key = os.getenv(api_key_env, "dummy")
    return ModelSpec(name=name, model=model, base_url=base_url, api_key=api_key)


def _optional_model_spec(
    config: dict[str, Any],
    name: str,
    model_override: str | None,
    base_url_override: str | None,
) -> ModelSpec | None:
    section = config.get(name) or {}
    configured_model = model_override or section.get("model")
    if not isinstance(configured_model, str) or not configured_model.strip():
        return None
    return _model_spec(config, name, model_override, base_url_override)


def _client(spec: ModelSpec) -> AsyncOpenAI:
    kwargs: dict[str, Any] = {"api_key": spec.api_key or "dummy"}
    if spec.base_url:
        kwargs["base_url"] = str(spec.base_url).rstrip("/")
    return AsyncOpenAI(**kwargs)


async def _complete(
    client: AsyncOpenAI,
    spec: ModelSpec,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    response = await client.chat.completions.create(
        model=spec.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
    )
    content = response.choices[0].message.content if response.choices else ""
    return content or ""


def _context_text(contexts: list[dict[str, Any]]) -> str:
    return "\n---\n".join(
        re.sub(r"\.{3,}", "", str(context.get("text", "")))
        for context in contexts
    )


async def evaluate(
    cases: list[dict[str, Any]],
    specs: list[ModelSpec],
    *,
    use_retrieval: bool,
    temperature: float,
    max_tokens: int,
    stability_runs: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from src.rag_flow.context_retriever import ContextRetriever

    retriever = ContextRetriever() if use_retrieval else None
    clients = {spec.name: _client(spec) for spec in specs}
    rows: list[dict[str, Any]] = []
    score_totals: dict[str, Counter[str]] = {
        spec.name: Counter() for spec in specs
    }
    score_counts: Counter[tuple[str, str]] = Counter()

    try:
        for index, case in enumerate(cases, start=1):
            question = str(case["question"])
            context_result = retriever.get_context(question) if retriever else {}
            contexts = context_result.get("contexts", []) if context_result else []
            context_text = _context_text(contexts)
            prompt = build_prompt(context_text, question)
            row: dict[str, Any] = {
                "case_id": case.get("id", f"eval-{index:04d}"),
                "question": question,
                "category": case.get("category"),
                "expected_points": case.get("expected_points", []),
                "retrieved_contexts": contexts,
                "prompt": prompt,
                "models": {},
            }

            for spec in specs:
                responses: list[str] = []
                errors: list[str] = []
                for _ in range(stability_runs):
                    try:
                        responses.append(
                            await _complete(
                                clients[spec.name],
                                spec,
                                prompt,
                                temperature,
                                max_tokens,
                            )
                        )
                    except Exception as exc:  # preserve failures per model/case
                        errors.append(f"{type(exc).__name__}: {exc}")
                        responses.append("")
                metrics = score_response(
                    responses[0] if responses else "",
                    case,
                    context_text,
                    responses,
                )
                qualitative = _qualitative_evaluation(case, metrics)
                row["models"][spec.name] = {
                    "model": spec.model,
                    "base_url": spec.base_url,
                    "responses": responses,
                    "errors": errors,
                    "metrics": metrics,
                    "qualitative_evaluation": qualitative,
                }
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        score_totals[spec.name][metric] += float(value)
                        score_counts[(spec.name, metric)] += 1
            rows.append(row)
    finally:
        await asyncio.gather(*(client.close() for client in clients.values()))

    summary: dict[str, Any] = {
        "cases": len(rows),
        "models": {},
        "metric_semantics": {
            "hallucination_tendency": "Heuristic ratio of answer numbers absent from retrieved context; inspect raw outputs.",
            "instruction_following": "Lexical expected-point coverage, not a semantic judge.",
            "contextual_adherence": "Lexical answer-token overlap with retrieved context.",
            "context_ignore_tendency": "One minus lexical RAG-context adherence; a risk signal, not a factuality judge.",
            "deployment_screening_score": "Weighted screening score for comparable per-case ranking; not a release gate.",
            "reviewed_feedback_alignment": "Only scored when the held-out case supplies structured reviewed-feedback labels; otherwise pending human review.",
        },
    }
    for spec in specs:
        summary["models"][spec.name] = {
            "model": spec.model,
            "mean_metrics": {
                metric: total / score_counts[(spec.name, metric)]
                for metric, total in score_totals[spec.name].items()
                if score_counts[(spec.name, metric)]
            },
        }
    return rows, summary


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _model_rows(
    rows: Iterable[dict[str, Any]], model_name: str, metadata: dict[str, Any]
) -> Iterable[dict[str, Any]]:
    for row in rows:
        model_result = row["models"].get(model_name)
        if model_result is None:
            continue
        yield {
            "metadata": metadata,
            "case_id": row["case_id"],
            "question": row["question"],
            "category": row.get("category"),
            "expected_points": row.get("expected_points", []),
            "retrieved_contexts": row.get("retrieved_contexts", []),
            "prompt": row.get("prompt", ""),
            "model_name": model_name,
            "model": model_result["model"],
            "responses": model_result["responses"],
            "errors": model_result["errors"],
            "metrics": model_result["metrics"],
            "qualitative_evaluation": model_result["qualitative_evaluation"],
        }


def _execution_status(rows: list[dict[str, Any]], model_name: str) -> str:
    results = [row["models"].get(model_name) for row in rows]
    results = [result for result in results if result is not None]
    if not results:
        return "pending"
    errors = sum(len(result.get("errors", [])) for result in results)
    responses = sum(
        bool(response.strip())
        for result in results
        for response in result.get("responses", [])
    )
    if errors and not responses:
        return "failed"
    if errors:
        return "completed_with_errors"
    return "completed"


def _short_text(value: Any, limit: int = 320) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _mean_score(summary: dict[str, Any], model_name: str) -> float | None:
    metrics = summary.get("models", {}).get(model_name, {}).get("mean_metrics", {})
    value = metrics.get("deployment_screening_score")
    return float(value) if isinstance(value, (int, float)) else None


def _comparison_examples(
    rows: list[dict[str, Any]], left: str, right: str, improved: bool
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows:
        left_result = row["models"].get(left)
        right_result = row["models"].get(right)
        if not left_result or not right_result:
            continue
        left_score = float(left_result["metrics"].get("deployment_screening_score", 0.0))
        right_score = float(right_result["metrics"].get("deployment_screening_score", 0.0))
        delta = right_score - left_score
        if (improved and delta <= 0.05) or (not improved and delta >= -0.05):
            continue
        examples.append(
            {
                "case_id": row["case_id"],
                "category": row.get("category"),
                "question": row["question"],
                "score_delta": round(delta, 4),
                "left_metrics": left_result["metrics"],
                "right_metrics": right_result["metrics"],
                "left_response": _short_text(left_result["responses"][0] if left_result["responses"] else ""),
                "right_response": _short_text(right_result["responses"][0] if right_result["responses"] else ""),
            }
        )
    return sorted(examples, key=lambda item: abs(item["score_delta"]), reverse=True)[:3]


def _render_examples(
    title: str, examples: list[dict[str, Any]], left_label: str, right_label: str
) -> list[str]:
    lines = [f"### {title}"]
    if not examples:
        lines.append("No cases crossed the 0.05 screening-score threshold.")
        return lines
    for example in examples:
        lines.extend(
            [
                f"- **{example['case_id']} ({example.get('category') or 'uncategorized'})**: score delta `{example['score_delta']:+.4f}` ({right_label} minus {left_label}).",
                f"  Question: {example['question']}",
                f"  {left_label}: {example['left_response']}",
                f"  {right_label}: {example['right_response']}",
                f"  Evidence: {json.dumps({'left': example['left_metrics'], 'right': example['right_metrics']}, ensure_ascii=False, sort_keys=True)}",
            ]
        )
    return lines


def build_comparison_report(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    metadata: dict[str, Any],
    model_status: dict[str, str],
) -> str:
    """Render a report entirely from structured evaluation records."""
    lines = [
        "# Base vs SFT vs DPO Evaluation",
        "",
        f"- Test set size: **{metadata['test_set_size']} cases**",
        f"- Dataset: `{metadata['dataset']}`",
        f"- Retrieval enabled: **{metadata['retrieval_enabled']}**",
        f"- Stability runs per model/case: **{metadata['stability_runs']}**",
        "- Evaluation method: one held-out question set, one retrieval pass and one rendered prompt shared by all available models; repeated deterministic requests are recorded; heuristic metrics and structured qualitative-review records are emitted per case.",
        "- Hold-out check: evaluation questions were checked against the configured SFT, ShareGPT, and DPO preference training exports.",
        "",
        "## Results",
        "",
        "| Model | Status | Mean screening score | Vietnamese | Instruction | RAG adherence | Ignore tendency | Unsupported claims | Stability |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_name, label in (("base", "Base"), ("sft", "SFT"), ("dpo", "DPO")):
        model_summary = summary.get("models", {}).get(model_name, {})
        metrics = model_summary.get("mean_metrics", {})
        if model_name not in summary.get("models", {}):
            values = ["pending", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a"]
        else:
            values = [
                f"{float(metrics.get('deployment_screening_score', 0.0)):.3f}",
                f"{float(metrics.get('vietnamese_response_quality', 0.0)):.3f}",
                f"{float(metrics.get('instruction_following', 0.0)):.3f}",
                f"{float(metrics.get('rag_context_adherence', 0.0)):.3f}",
                f"{float(metrics.get('context_ignore_tendency', 0.0)):.3f}",
                f"{float(metrics.get('unsupported_claim_rate', 0.0)):.3f}",
                f"{float(metrics.get('answer_stability', 0.0)):.3f}",
            ]
        lines.append(f"| {label} | {model_status.get(model_name, 'unknown')} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "Scores are screening signals. Higher is better except ignore tendency and unsupported-claim rate, where lower is better. Lexical context and expected-point metrics require human review of the raw Vietnamese answer.",
            "",
            "## Base Results",
            "",
            json.dumps(summary.get("models", {}).get("base", {}), ensure_ascii=False, indent=2),
            "",
            "## SFT Results",
            "",
            json.dumps(summary.get("models", {}).get("sft", {}), ensure_ascii=False, indent=2),
            "",
            "## DPO Results",
            "",
        ]
    )
    if model_status.get("dpo") != "completed":
        if model_status.get("dpo") == "pending":
            lines.append("DPO execution is **pending** because no DPO checkpoint/model endpoint was configured. No DPO quality conclusion is made.")
        else:
            lines.append(f"DPO execution status is **{model_status.get('dpo')}**. No DPO quality conclusion is made.")
    else:
        lines.append(json.dumps(summary.get("models", {}).get("dpo", {}), ensure_ascii=False, indent=2))

    lines.extend(["", "## Comparative Evidence", ""])
    lines.extend(_render_examples("SFT improves over Base", _comparison_examples(rows, "base", "sft", True), "Base", "SFT"))
    lines.append("")
    lines.extend(_render_examples("SFT regresses compared with Base", _comparison_examples(rows, "base", "sft", False), "Base", "SFT"))
    lines.append("")
    if model_status.get("dpo") != "completed":
        lines.append("### DPO improves over SFT")
        lines.append("Pending DPO execution.")
        lines.append("")
        lines.append("### DPO regresses compared with SFT")
        lines.append("Pending DPO execution.")
    else:
        lines.extend(_render_examples("DPO improves over SFT", _comparison_examples(rows, "sft", "dpo", True), "SFT", "DPO"))
        lines.append("")
        lines.extend(_render_examples("DPO regresses compared with SFT", _comparison_examples(rows, "sft", "dpo", False), "SFT", "DPO"))

    lines.extend(["", "## Qualitative Review Records", "", "Each JSONL record contains a `qualitative_evaluation` object with the rubric, metric evidence, and reviewed-feedback status. The current held-out cases do not include reviewed-feedback labels, so alignment with reviewed user preferences remains `not_available` and must be completed by a reviewer before deployment.", ""])

    available = [name for name in ("base", "sft", "dpo") if model_status.get(name) == "completed"]
    scores = {name: _mean_score(summary, name) for name in available}
    scores = {name: score for name, score in scores.items() if score is not None}
    if model_status.get("dpo") != "completed":
        winner = max(scores, key=scores.get) if scores else None
        recommendation = (
            f"DPO evaluation is not complete. Based on the available screening results, **{winner.upper()} is the provisional candidate**; do not deploy until DPO is evaluated and qualitative review is complete."
            if winner
            else "DPO evaluation is pending and no completed model results are available; no deployment recommendation can be made."
        )
    else:
        winner = max(scores, key=scores.get) if scores else None
        recommendation = (
            f"Select **{winner.upper()}** for deployment review because it has the highest mean screening score among completed models. This is a recommendation for review only; verify qualitative records, unsupported claims, and regression cases before deployment."
            if winner
            else "No completed model results are available; no deployment recommendation can be made."
        )
    lines.extend(["## Deployment Recommendation", "", recommendation, "", "Deployment was not performed by this task."])
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--base-model")
    parser.add_argument("--sft-model")
    parser.add_argument("--dpo-model")
    parser.add_argument("--base-url")
    parser.add_argument("--sft-base-url")
    parser.add_argument("--dpo-base-url")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--stability-runs", type=int)
    parser.add_argument("--no-rag", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config)
        dataset_path = args.dataset or ROOT_DIR / str(
            config.get("dataset", "data/eval/eval_questions.jsonl")
        )
        cases = load_jsonl(dataset_path)
        training_data = config.get(
            "training_data", ["data/training/exports/sft_dataset.jsonl"]
        )
        if isinstance(training_data, str):
            training_data = [training_data]
        assert_held_out(
            cases,
            (ROOT_DIR / str(path) for path in training_data),
        )
        specs = [
            _model_spec(config, "base", args.base_model, args.base_url),
            _model_spec(config, "sft", args.sft_model, args.sft_base_url),
        ]
        dpo_spec = _optional_model_spec(
            config, "dpo", args.dpo_model, args.dpo_base_url
        )
        model_status = {"base": "completed", "sft": "completed"}
        if dpo_spec:
            specs.append(dpo_spec)
            model_status["dpo"] = "completed"
        else:
            model_status["dpo"] = "pending"
        stability_runs = args.stability_runs or int(config.get("stability_runs", 3))
        if stability_runs < 1:
            raise ValueError("--stability-runs must be at least 1")
        rows, summary = asyncio.run(
            evaluate(
                cases,
                specs,
                use_retrieval=not args.no_rag and bool(config.get("retrieval", True)),
                temperature=float(config.get("temperature", 0.0)),
                max_tokens=int(config.get("max_tokens", 2048)),
                stability_runs=stability_runs,
            )
        )
        for model_name in ("base", "sft"):
            model_status[model_name] = _execution_status(rows, model_name)
        if dpo_spec:
            model_status["dpo"] = _execution_status(rows, "dpo")
        metadata = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset": str(dataset_path),
            "test_set_size": len(cases),
            "same_prompt_for_models": True,
            "retrieval_enabled": not args.no_rag and bool(config.get("retrieval", True)),
            "stability_runs": stability_runs,
            "model_status": model_status,
            "holdout_training_data": [str(path) for path in training_data],
        }
        output_dir = args.output_dir or args.output.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(({"metadata": metadata, **row} for row in rows), args.output)
        for model_name in ("base", "sft"):
            write_jsonl(
                _model_rows(rows, model_name, metadata),
                output_dir / MODEL_OUTPUT_NAMES[model_name],
            )
        if dpo_spec:
            write_jsonl(
                _model_rows(rows, "dpo", metadata),
                output_dir / MODEL_OUTPUT_NAMES["dpo"],
            )
        else:
            write_jsonl(
                [{
                    "metadata": metadata,
                    "model_name": "dpo",
                    "status": "pending",
                    "reason": "No DPO checkpoint/model endpoint configured.",
                }],
                output_dir / MODEL_OUTPUT_NAMES["dpo"],
            )
        summary["metadata"] = metadata
        summary["model_status"] = model_status
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        report_path = args.report
        if args.output_dir:
            report_path = args.output_dir / "comparison_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            build_comparison_report(rows, summary, metadata, model_status),
            encoding="utf-8",
        )
        print(f"Wrote {len(rows)} evaluation rows to {args.output}")
        print(f"Wrote per-model results to {output_dir}")
        print(f"Wrote evaluation summary to {args.summary}")
        print(f"Wrote comparison report to {report_path}")
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Evaluation was not run: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
