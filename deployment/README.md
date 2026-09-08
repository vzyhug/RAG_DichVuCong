# Selected model serving

## Current selection state

The repository checkout does not contain a Task 7 evaluation summary or model
weights. Therefore no model is selected or served by default. This is
intentional: deployment must not silently choose DPO, SFT, or a checkpoint that
has not been evaluated.

After Task 7 has produced `evaluation/summary.json` and the selected artifact
exists, prepare the deployment manifest:

```bash
python scripts/prepare_model_serving.py
```

The resolver selects the highest `deployment_screening_score` among models
whose Task 7 status is exactly `completed`. It considers Base, SFT, and DPO in
that order for deterministic exact-score ties, so DPO is not an implicit
default. The generated `deployment/selected_model.json` records the winner,
score, artifact kind, endpoint, model identifier, and complete vLLM command.

## Artifact decision

The training configs use QLoRA/LoRA and write adapters under
`training/outputs/`. The resolver first looks for the selected PEFT adapter's
`adapter_config.json`, including the highest numbered `checkpoint-*` directory
when the output root contains only checkpoints. vLLM can load that adapter
directly on top of `Qwen/Qwen2.5-7B-Instruct`, so the normal deployment is:

```text
Base Model + LoRA Adapter
```

If an explicitly configured `merged_model` directory has `config.json` and
weights but no adapter is available, the resolver serves that merged directory
instead. No merge is required for the selected vLLM path, and the original base
model and adapter directories are never changed. If merging is needed for a
different backend, do it in a separate output directory with PEFT's
`merge_and_unload()` and keep the source directories intact.

## Start the server

Run this on the CUDA host with vLLM installed:

```bash
python scripts/serve_selected_model.py
```

The generated command serves the model at `http://127.0.0.1:8000/v1` and uses
the generated model identifier, for example `dvc-bca-rag-sft` or
`dvc-bca-rag-dpo`. Set `VLLM_API_KEY` before starting if the server should
require an API key.

Install vLLM in a CUDA-compatible environment according to the version used by
the serving host. A 7B Qwen model needs a CUDA GPU and enough VRAM for the
base weights, KV cache, and adapter; start with one GPU, `--dtype auto`, a
2048-token context, and 90% GPU memory utilization. Increase those values only
after measuring the target host.

## Application configuration

Point the existing Local provider at the generated endpoint and identifier:

```env
LLM_PROVIDER=local
LOCAL_LLM_URL=http://127.0.0.1:8000/v1
LOCAL_LLM_MODEL=dvc-bca-rag-sft
```

Replace `dvc-bca-rag-sft` with the `served_model` value in
`deployment/selected_model.json`. The Streamlit app and RAG flow require no
changes; `LLMFactory` continues to create the OpenAI-compatible client.

## Verify the request path

With the server running and the three environment variables set, run:

```bash
python scripts/verify_local_llm.py
```

This sends an OpenAI-compatible chat request through
`LLMFactory -> Local provider -> vLLM endpoint` and prints the response. A
direct protocol check is also available:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"dvc-bca-rag-sft","messages":[{"role":"user","content":"Reply with exactly: LOCAL_LLM_OK"}],"temperature":0,"max_tokens":32}'
```

The vLLM server implements the OpenAI Chat Completions API and supports static
LoRA modules through `--enable-lora` and `--lora-modules`:

- https://docs.vllm.ai/en/stable/features/lora/
- https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/
