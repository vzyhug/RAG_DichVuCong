import json
import os
from pathlib import Path
import subprocess
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.export_sft_dataset import build_sft_examples, write_jsonl  # noqa: E402


def test_local_settings_are_loaded_and_required(tmp_path):
    child_env = os.environ.copy()
    child_env.update(
        {
            "LLM_PROVIDER": "local",
            "LOCAL_LLM_URL": "http://localhost:11434/v1",
            "LOCAL_LLM_MODEL": "qwen2.5:7b-instruct",
            "PYTHONPATH": str(ROOT_DIR),
        }
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from configs.settings import settings; print(settings.LOCAL_LLM_URL); print(settings.LOCAL_LLM_MODEL)",
        ],
        cwd=tmp_path,
        env=child_env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "http://localhost:11434/v1",
        "qwen2.5:7b-instruct",
    ]

    missing_env = dict(child_env)
    missing_env["LOCAL_LLM_MODEL"] = ""
    result = subprocess.run(
        [sys.executable, "-c", "from configs.settings import settings"],
        cwd=tmp_path,
        env=missing_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "LOCAL_LLM_MODEL" in result.stderr


def test_build_sft_examples_applies_review_rules_and_order():
    logs = [
        {
            "turn_id": "approved-turn",
            "created_at": "2026-01-02T00:00:00+00:00",
            "user_query": "Approved question",
            "assistant_answer": "Approved answer",
            "review_status": "approved",
            "feedback": {"rating": "up"},
        },
        {
            "turn_id": "edited-turn",
            "created_at": "2026-01-01T00:00:00+00:00",
            "user_query": "Edited question",
            "assistant_answer": "Bad generated answer",
            "corrected_answer": "Human answer",
            "review_status": "edited",
        },
        {
            "turn_id": "empty-edited-turn",
            "user_query": "Should be excluded",
            "assistant_answer": "Generated answer",
            "corrected_answer": "",
            "review_status": "edited",
        },
        {
            "turn_id": "raw-turn",
            "user_query": "Unreviewed question",
            "assistant_answer": "Unreviewed answer",
            "review_status": "raw",
        },
        {
            "turn_id": "rejected-turn",
            "user_query": "Rejected question",
            "assistant_answer": "Rejected answer",
            "review_status": "rejected",
        },
    ]

    examples = build_sft_examples(logs)

    assert [
        example["messages"][1]["content"] for example in examples
    ] == ["Edited question", "Approved question"]
    assert examples[0]["messages"][2]["content"] == "Human answer"
    assert examples[1]["messages"][2]["content"] == "Approved answer"
    assert all(len(example["messages"]) == 3 for example in examples)


def test_write_jsonl_writes_one_valid_object_per_line(tmp_path):
    output_path = tmp_path / "nested" / "sft_dataset.jsonl"
    rows = build_sft_examples(
        [
            {
                "created_at": "2026-01-01T00:00:00+00:00",
                "user_query": "What is my email a@example.com?",
                "assistant_answer": "Call 0901234567.",
                "review_status": "approved",
            }
        ]
    )

    assert write_jsonl(rows, output_path) == 1
    with output_path.open(encoding="utf-8") as output_file:
        parsed = [json.loads(line) for line in output_file if line.strip()]
    assert parsed[0]["messages"][1]["content"] == "What is my email [EMAIL]?"
    assert parsed[0]["messages"][2]["content"] == "Call [PHONE]."
