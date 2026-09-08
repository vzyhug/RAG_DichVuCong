import json
from pathlib import Path

import pytest

from scripts.prepare_model_serving import (
    ServingPreparationError,
    prepare_manifest,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _summary(*, sft_status="completed", dpo_status="pending"):
    return {
        "model_status": {"base": "completed", "sft": sft_status, "dpo": dpo_status},
        "models": {
            "base": {"mean_metrics": {"deployment_screening_score": 0.5}},
            "sft": {"mean_metrics": {"deployment_screening_score": 0.8}},
            "dpo": {"mean_metrics": {"deployment_screening_score": 0.99}},
        },
    }


def _config(root: Path) -> Path:
    config = root / "serving.json"
    _write_json(
        config,
        {
            "backend": "vllm",
            "evaluation_summary": "evaluation/summary.json",
            "base_model": "Qwen/Qwen2.5-7B-Instruct",
            "served_model_prefix": "test-model",
            "artifacts": {
                "base": {"model": "Qwen/Qwen2.5-7B-Instruct"},
                "sft": {"adapter": "outputs/sft", "merged_model": "outputs/sft-merged"},
                "dpo": {"adapter": "outputs/dpo", "merged_model": "outputs/dpo-merged"},
            },
        },
    )
    return config


def test_prepare_manifest_selects_best_completed_non_dpo_model(tmp_path):
    _write_json(tmp_path / "evaluation/summary.json", _summary())
    _write_json(tmp_path / "outputs/sft/adapter_config.json", {"base_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct"})

    manifest = prepare_manifest(
        config_path=_config(tmp_path),
        output_path=tmp_path / "deployment/selected_model.json",
        root=tmp_path,
    )

    assert manifest["selected_model"] == "sft"
    assert manifest["artifact_kind"] == "lora_adapter"
    assert manifest["served_model"] == "test-model-sft"
    assert "--enable-lora" in manifest["command"]
    assert "--lora-modules" in manifest["command"]


def test_prepare_manifest_does_not_assume_dpo(tmp_path):
    summary = _summary(sft_status="failed", dpo_status="completed")
    _write_json(tmp_path / "evaluation/summary.json", summary)
    _write_json(tmp_path / "outputs/dpo/adapter_config.json", {})

    manifest = prepare_manifest(
        config_path=_config(tmp_path),
        output_path=tmp_path / "deployment/selected_model.json",
        root=tmp_path,
    )

    assert manifest["selected_model"] == "dpo"


def test_prepare_manifest_finds_latest_adapter_checkpoint(tmp_path):
    _write_json(tmp_path / "evaluation/summary.json", _summary(dpo_status="failed"))
    _write_json(tmp_path / "outputs/sft/checkpoint-10/adapter_config.json", {})
    _write_json(tmp_path / "outputs/sft/checkpoint-20/adapter_config.json", {})

    manifest = prepare_manifest(
        config_path=_config(tmp_path),
        output_path=tmp_path / "deployment/selected_model.json",
        root=tmp_path,
    )

    assert manifest["artifact_path"].endswith("checkpoint-20")


def test_prepare_manifest_uses_separate_merged_directory(tmp_path):
    _write_json(tmp_path / "evaluation/summary.json", _summary(dpo_status="failed"))
    merged = tmp_path / "outputs/sft-merged"
    _write_json(merged / "config.json", {})
    (merged / "model.safetensors").write_bytes(b"weights")

    manifest = prepare_manifest(
        config_path=_config(tmp_path),
        output_path=tmp_path / "deployment/selected_model.json",
        root=tmp_path,
    )

    assert manifest["selected_model"] == "sft"
    assert manifest["artifact_kind"] == "merged_model"
    assert "--enable-lora" not in manifest["command"]


def test_prepare_manifest_fails_without_task7_summary(tmp_path):
    with pytest.raises(ServingPreparationError, match="evaluation summary is missing"):
        prepare_manifest(
            config_path=_config(tmp_path),
            output_path=tmp_path / "deployment/selected_model.json",
            root=tmp_path,
        )
