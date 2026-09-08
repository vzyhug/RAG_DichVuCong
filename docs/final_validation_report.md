# Final End-to-End Validation Report

Date: 2026-09-07

## Result

The runtime application and data/training tooling are present and importable.
The complete production pipeline is **not deployable yet** because no SFT or
DPO checkpoint, evaluation summary, preference pair, or selected-model
manifest exists in this workspace. No GPU training was rerun.

## Completed Modules

- Streamlit chatbot startup and provider-neutral LLM service.
- Gemini and OpenAI-compatible Local LLM selection through `LLMFactory`.
- FAISS retrieval, context grading, prompt construction, and RAG diagnostics.
- Firebase chat-log schema, ratings, review statuses, corrected answers, and
  best-effort persistence.
- SFT and preference dataset exporters with PII redaction and JSONL reports.
- OpenAI-messages to ShareGPT conversion.
- Reproducible LoRA/QLoRA SFT and DPO configurations and launchers.
- Held-out Base/SFT/DPO evaluation harness.
- Evaluation-gated vLLM manifest generation and selected-model serving script.

## Validation Evidence

- `27 passed` in the final `python -m pytest -q` run, including Firebase
  feedback/review persistence and the missing-intent RAG regression.
- All application, package, script, and test Python files compile successfully.
- Streamlit health endpoint returned HTTP 200 on ports 8501 and 8503.
- Gemini request succeeded using the configured model and returned non-empty
  content.
- FAISS index loaded 1,839 metadata rows; a live query returned 10 chunks with
  a top score of `0.8798425`.
- The rendered prompt contained the retrieved chunk text.
- SFT export validation: 1 total, 1 valid, 0 invalid, 0 duplicates, 0 warnings.
- SFT to ShareGPT conversion produced 1 row.
- Firebase persistence and feedback/review behavior are validated with an
  in-memory Firestore-compatible integration test; no production test write was
  performed.
- Local LLM smoke test is intentionally blocked by the current environment:
  `LLM_PROVIDER=gemini`, no local URL/model is configured, and no local server
  is listening on port 11434.
- Deployment preparation fails closed because `evaluation/summary.json` is
  missing and no training output artifact exists.

## Runtime Flow

```text
User -> Streamlit -> RAG Retriever -> Retrieved Context -> Prompt Builder
     -> LLMFactory -> Selected LLM -> Response
```

The local model receives the same RAG-rendered prompt as Gemini. The reasoning
gate was made compatible with the checked-in index: if optional intent metadata
is absent, retrieved context is allowed to proceed to grounded generation.

## Feedback and Training Flow

```text
Response -> Firebase Logging -> User Rating / Review
         -> Approved / Edited / Rejected Samples -> SFT Dataset
         -> SFT Training -> Preference Dataset -> DPO Training
         -> Model Evaluation -> Model Selection -> Deployment
```

## Final Architecture

```text
                    ┌────────────── RAG Knowledge Base
                    │
User → Streamlit → Retriever
                    │
                    ▼
              Retrieved Context
                    │
                    ▼
              Prompt Builder
                    │
                    ▼
                LLMFactory
                    │
             ┌──────┴──────┐
             ▼             ▼
          Gemini       Local LLM
                           │
                           ▼
                        Response
                           │
                           ▼
                       Firebase
                           │
                           ▼
                         Review
                      ┌────┴────┐
                      ▼         ▼
                     SFT   Preference Data
                      │         │
                      ▼         ▼
                    LoRA       DPO
                      └────┬────┘
                           ▼
                       Evaluation
                           │
                           ▼
                    Selected Model
                           │
                           ▼
                       Deployment
```

## Selected Model and Evaluation Summary

No model is selected. `configs/evaluation.json` intentionally leaves SFT and
DPO model names unset, `evaluation/summary.json` does not exist, and the
training output directories are absent. Therefore there is no defensible Base,
SFT, or DPO score to report and no deployment endpoint to verify.

The held-out set contains 10 questions across administrative, emergency,
no-data, PCCC, residence, and vehicle-registration categories. It is separate
from the training exports.

## Known Limitations and Unresolved Issues

- The current reviewed export has only one SFT sample.
- The preference export is empty: there are no explicit chosen/rejected pairs.
- SFT and DPO checkpoints have not been produced in this workspace.
- Evaluation requires a running OpenAI-compatible local endpoint and real model
  names/checkpoints; it was not fabricated or run against placeholder models.
- Firebase live writes were not performed during validation to avoid mutating
  the configured project. The application path remains best-effort, so logging
  failures do not stop chat.
- The static FastAPI client does not implement the Streamlit Firebase review
  workflow; Streamlit is the documented feedback/training entry point.
- Numeric evaluation metrics are screening signals and require human review of
  Vietnamese answers, unsupported claims, and legal correctness.

## Recommended Future Improvements

1. Collect and review a larger anonymized corpus, including explicit edited and
   rejected answers for the same prompts to create DPO pairs.
2. Run SFT and, only after non-empty preference validation, DPO on a CUDA host.
3. Run the held-out evaluation with Base/SFT/DPO, complete qualitative review,
   and retain `evaluation/summary.json` as the selection evidence.
4. Generate `deployment/selected_model.json`, serve the winner with vLLM, and
   run the local endpoint smoke test plus a RAG-context request.
5. Add authentication and audit history to the admin review surface before
   production use.

## Exact Commands

Commands below are run from the repository root.

### Install and configure

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set Gemini in the current PowerShell session:

```powershell
$env:LLM_PROVIDER = "gemini"
$env:GEMINI_API_KEY = "<your-key>"
$env:GEMINI_MODEL = "gemini-3.5-flash-lite"
```

Set Local LLM in the current PowerShell session:

```powershell
$env:LLM_PROVIDER = "local"
$env:LOCAL_LLM_URL = "http://localhost:11434/v1"
$env:LOCAL_LLM_MODEL = "qwen2.5:3b-instruct"
```

### Start the application

```powershell
python -m streamlit run streamlit_app.py
```

Switching between Gemini and Local LLM only requires changing `LLM_PROVIDER`
and the corresponding model/endpoint variables, then restarting Streamlit.

### Start Local LLM inference

For an Ollama OpenAI-compatible endpoint:

```powershell
ollama serve
ollama pull qwen2.5:3b-instruct
python scripts\smoke_test_local_llm.py
```

### Export and validate data

```powershell
python scripts\export_sft_dataset.py
python scripts\validate_training_exports.py --sft data\training\exports\sft_dataset.jsonl
python scripts\convert_sft_to_sharegpt.py
python scripts\export_preference_dataset.py
python scripts\validate_training_exports.py --require-preference
python scripts\validate_dpo_inputs.py
```

`validate_training_exports.py` reports the current JSONL structure; the final
`validate_dpo_inputs.py` guard intentionally fails until at least one valid
preference pair exists.

### Train SFT and DPO

```powershell
.\scripts\train_sft.ps1
.\scripts\train_dpo.ps1
```

These commands require a CUDA-enabled environment with the pinned
LLaMA-Factory setup described in `training/README.md`.

### Evaluate checkpoints

Set `sft.model` and, when available, `dpo.model` in
`configs/evaluation.json`, then run:

```powershell
python scripts\evaluate_base_vs_sft.py --config configs\evaluation.json --output-dir evaluation
```

### Prepare and serve the selected model

```powershell
python scripts\prepare_model_serving.py --config configs\serving.json --output deployment\selected_model.json
python scripts\serve_selected_model.py --manifest deployment\selected_model.json
```

The generated manifest supplies the final `LOCAL_LLM_URL` and
`LOCAL_LLM_MODEL` values. For the default serving config they are derived from
the selected model prefix and `http://127.0.0.1:8000/v1`.
