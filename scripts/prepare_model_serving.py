"""Resolve the Task 7 winner into a reproducible vLLM serving manifest.

The evaluator owns model selection. This script only consumes its summary and
refuses to guess when the summary or selected training artifact is missing.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT_DIR / "configs" / "serving.json"
DEFAULT_OUTPUT = ROOT_DIR / "deployment" / "selected_model.json"
MODEL_ORDER = ("base", "sft", "dpo")
CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")


class ServingPreparationError(ValueError):
    """Raised when deployment inputs are incomplete or inconsistent."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ServingPreparationError(f"Cannot read JSON file: {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ServingPreparationError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ServingPreparationError(f"Expected a JSON object in {path}")
    return value


def _resolve_path(value: str, *, root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _completed_winner(summary: dict[str, Any]) -> tuple[str, float]:
    statuses = summary.get("model_status", {})
    models = summary.get("models", {})
    candidates: list[tuple[str, float]] = []
    for name in MODEL_ORDER:
        if statuses.get(name) != "completed":
            continue
        score = models.get(name, {}).get("mean_metrics", {}).get(
            "deployment_screening_score"
        )
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            continue
        candidates.append((name, float(score)))
    if not candidates:
        raise ServingPreparationError(
            "Task 7 has no completed model with a deployment_screening_score; "
            "run evaluation and review its output before serving a model."
        )
    # Prefer the simpler artifact on an exact score tie. This is deterministic
    # and does not make DPO the implicit default.
    return max(candidates, key=lambda item: (item[1], -MODEL_ORDER.index(item[0])))


def _checkpoint_or_root(path: Path) -> Path:
    if (path / "adapter_config.json").is_file():
        return path
    candidates: list[tuple[int, Path]] = []
    if path.is_dir():
        for child in path.iterdir():
            match = CHECKPOINT_PATTERN.match(child.name)
            if child.is_dir() and match and (child / "adapter_config.json").is_file():
                candidates.append((int(match.group(1)), child))
    if candidates:
        return max(candidates, key=lambda item: (item[0], item[1].name))[1]
    return path


def _is_merged_model(path: Path) -> bool:
    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    weight_patterns = ("*.safetensors", "pytorch_model*.bin", "*.bin")
    return any(path.glob(pattern) for pattern in weight_patterns)


def _resolve_artifact(
    name: str, spec: dict[str, Any], *, root: Path
) -> tuple[str, Path | None]:
    if name == "base":
        return "base_model", None

    adapter_value = spec.get("adapter")
    if isinstance(adapter_value, str) and adapter_value.strip():
        adapter = _checkpoint_or_root(_resolve_path(adapter_value, root=root))
        if (adapter / "adapter_config.json").is_file():
            return "lora_adapter", adapter

    merged_value = spec.get("merged_model")
    if isinstance(merged_value, str) and merged_value.strip():
        merged = _resolve_path(merged_value, root=root)
        if _is_merged_model(merged):
            return "merged_model", merged

    raise ServingPreparationError(
        f"No deployable artifact found for selected model {name!r}. Expected "
        f"a PEFT adapter with adapter_config.json or a merged model with "
        f"config.json and weights in {spec.get('adapter')!r} / "
        f"{spec.get('merged_model')!r}."
    )


def prepare_manifest(
    *, config_path: Path = DEFAULT_CONFIG,
    output_path: Path = DEFAULT_OUTPUT,
    root: Path = ROOT_DIR,
) -> dict[str, Any]:
    config = _read_json(config_path)
    summary_path = _resolve_path(str(config.get("evaluation_summary", "")), root=root)
    if not summary_path.is_file():
        raise ServingPreparationError(
            f"Task 7 evaluation summary is missing: {summary_path}. "
            "No model will be selected without evaluation results."
        )
    summary = _read_json(summary_path)
    selected, score = _completed_winner(summary)
    artifacts = config.get("artifacts", {})
    spec = artifacts.get(selected)
    if not isinstance(spec, dict):
        raise ServingPreparationError(f"No serving artifact configuration for {selected!r}")
    artifact_kind, artifact_path = _resolve_artifact(selected, spec, root=root)

    base_model = str(config.get("base_model", "")).strip()
    if not base_model:
        raise ServingPreparationError("configs/serving.json must define base_model")
    prefix = str(config.get("served_model_prefix", "dvc-bca-rag")).strip()
    served_model = f"{prefix}-{selected}"
    host = str(config.get("host", "127.0.0.1"))
    port = int(config.get("port", 8000))
    vllm = config.get("vllm", {})
    if not isinstance(vllm, dict):
        raise ServingPreparationError("The vllm serving settings must be an object")

    model_argument = base_model if artifact_kind == "lora_adapter" else (
        str(artifact_path) if artifact_path else base_model
    )
    command = [
        "vllm",
        "serve",
        model_argument,
        "--served-model-name",
        served_model,
        "--host",
        host,
        "--port",
        str(port),
        "--dtype",
        str(vllm.get("dtype", "auto")),
        "--generation-config",
        "vllm",
        "--max-model-len",
        str(int(vllm.get("max_model_len", 2048))),
        "--gpu-memory-utilization",
        str(float(vllm.get("gpu_memory_utilization", 0.9))),
    ]
    if artifact_kind == "lora_adapter":
        command.extend(
            [
                "--enable-lora",
                "--lora-modules",
                f"{served_model}={artifact_path}",
                "--max-lora-rank",
                str(int(vllm.get("max_lora_rank", 16))),
            ]
        )

    manifest = {
        "selected_model": selected,
        "evaluation_score": score,
        "evaluation_summary": str(summary_path),
        "base_model": base_model,
        "artifact_kind": artifact_kind,
        "artifact_path": str(artifact_path) if artifact_path else None,
        "served_model": served_model,
        "endpoint": f"http://{host}:{port}/v1",
        "command": command,
        "command_display": shlex.join(command),
        "selection_policy": "highest Task 7 deployment_screening_score among status=completed; base wins exact ties",
        "original_artifacts_preserved": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        manifest = prepare_manifest(config_path=args.config, output_path=args.output)
    except (OSError, ServingPreparationError) as exc:
        parser.error(str(exc))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
