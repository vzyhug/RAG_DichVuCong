import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.validate_dpo_inputs import main as validate_dpo_inputs


def test_dpo_dataset_is_registered_as_a_ranking_dataset():
    dataset_info = json.loads(
        (ROOT_DIR / "training" / "data" / "dataset_info.json").read_text(
            encoding="utf-8"
        )
    )

    assert dataset_info["preference_dataset"] == {
        "file_name": "../../data/training/exports/preference_dataset.jsonl",
        "ranking": True,
        "columns": {"prompt": "prompt", "chosen": "chosen", "rejected": "rejected"},
    }


def test_dpo_config_uses_sft_adapter_and_separate_output():
    config = (ROOT_DIR / "training" / "configs" / "dpo_qwen_lora.yaml").read_text(
        encoding="utf-8"
    )

    assert "stage: dpo" in config
    assert "adapter_name_or_path: training/outputs/qwen2.5-7b-instruct-qlora" in config
    assert "dataset: preference_dataset" in config
    assert "output_dir: training/outputs/qwen2.5-7b-instruct-dpo-lora" in config
    assert "pref_beta: 0.1" in config
    assert "learning_rate: 0.00005" in config
    assert "cutoff_len: 2048" in config
    assert "overwrite_output_dir: false" in config


def test_dpo_input_guard_rejects_empty_dataset(tmp_path):
    preference_path = tmp_path / "preference.jsonl"
    preference_path.write_text("", encoding="utf-8")

    assert validate_dpo_inputs(["--preference", str(preference_path)]) == 1


def test_dpo_input_guard_accepts_valid_pair(tmp_path, capsys):
    preference_path = tmp_path / "preference.jsonl"
    preference_path.write_text(
        json.dumps(
            {"prompt": "question", "chosen": "grounded answer", "rejected": "guess"}
        )
        + "\n",
        encoding="utf-8",
    )

    assert validate_dpo_inputs(["--preference", str(preference_path)]) == 0
    assert "1 preference rows" in capsys.readouterr().out
