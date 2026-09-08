# Held-out evaluation data

`eval_questions.jsonl` is maintained separately from `data/training/` and is
not an input to the SFT export scripts. Each row contains a question, a
category, expected answer points, and source hints for human review.

Run the shared Base/SFT/DPO harness after configuring Base and SFT model
names. DPO is optional until its checkpoint is available:

```bash
python scripts/evaluate_base_vs_sft.py \
  --sft-model path-or-server-model-name \
  --output-dir evaluation
```

The default configuration uses the local OpenAI-compatible endpoint and the
base `qwen2.5:7b-instruct` model. The SFT model is intentionally `null` until
a real checkpoint or serving-model name is available. Therefore this
repository contains no fabricated evaluation results.

The runner checks the evaluation questions against both SFT exports and DPO
preference prompts before inference. Retrieval happens once per question and
the same rendered prompt is sent to every configured model. It writes
`evaluation/base_results.jsonl`, `evaluation/sft_results.jsonl`,
`evaluation/dpo_results.jsonl`, and `evaluation/comparison_report.md`; DPO is
recorded as pending when no checkpoint/model name is configured. Every result
row contains the raw repeated responses, heuristic metrics, and a structured
qualitative-review record. Numeric hallucination and lexical overlap scores
are screening signals and should be supplemented by human review.

Before inference, the runner also checks that evaluation questions do not
exactly overlap the configured SFT exports or DPO preference prompts.
