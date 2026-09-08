# Local inference and GPU SFT/DPO package

This package prepares supervised fine-tuning for `Qwen/Qwen2.5-7B-Instruct` with
LLaMA-Factory and LoRA. The checked-in YAML is configured for 4-bit QLoRA; the
same file documents the small change needed for ordinary LoRA.

## Dataset contract

The training input is the validated Task 3 artifact:

- `data/training/exports/sft_dataset.jsonl` is the validated OpenAI-messages export.
- `data/training/exports/sft_dataset_sharegpt.jsonl` is its validated ShareGPT conversion.
- `data/training/exports/preference_dataset.jsonl` contains `prompt`, `chosen`, and
  `rejected` rows for DPO.
- `training/data/dataset_info.json` registers that artifact for LLaMA-Factory.

Validate and regenerate the conversion from the repository root when the reviewed
dataset changes:

```bash
python scripts/validate_training_exports.py --sft data/training/exports/sft_dataset.jsonl
python scripts/convert_sft_to_sharegpt.py
```

The current export is structurally valid but contains only one sample. Add more
approved or edited reviewed examples before treating a training run as useful.

## Local inference environment

The local inference path is independent from training and does not install a
server or download model weights. Configure the existing application settings:

```env
LLM_PROVIDER=local
LOCAL_LLM_URL=http://localhost:11434/v1
LOCAL_LLM_MODEL=qwen2.5:3b-instruct
```

Run the independent smoke test from the repository root:

```bash
python scripts/smoke_test_local_llm.py
```

It loads `configs.settings`, constructs the existing `LLMFactory` client, sends
one minimal chat request, and reports the provider, model, endpoint, result, and
latency. A stopped server, refused connection, bad endpoint, missing model,
timeout, invalid configuration, or malformed response produces a non-zero exit
with an actionable category. The command never installs Ollama or downloads a
model.

## GPU training environment

Do not run the 7B training job in the local inference environment. Use a Linux
CUDA machine or Google Colab with enough VRAM, disk, and system RAM. QLoRA still
requires downloading the base model once in that environment.

The package uses LLaMA-Factory `v0.9.4` as the reproducible reference version.
Install a CUDA-matched PyTorch build first, then install the pinned trainer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
python -m pip install llamafactory==0.9.4
```

For a different CUDA/PyTorch combination, use the matching PyTorch index and
keep the LLaMA-Factory version recorded with the experiment. Verify the GPU
before training:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA GPU')"
```

From the repository root, launch the canonical configuration:

```bash
bash scripts/train_sft.sh
```

On PowerShell:

```powershell
.\scripts\train_sft.ps1
```

The launchers only select the YAML and pass optional extra LLaMA-Factory
arguments; training parameters live in `training/configs/sft_qwen_lora.yaml`.
The output adapter and checkpoints are written under
`training/outputs/qwen2.5-7b-instruct-qlora`.

## DPO continuation

DPO starts from the SFT LoRA adapter and keeps its output separate:

```text
Qwen/Qwen2.5-7B-Instruct
  -> SFT: training/outputs/qwen2.5-7b-instruct-qlora
  -> DPO: training/outputs/qwen2.5-7b-instruct-dpo-lora
```

The DPO config loads the base model together with the SFT adapter through
`adapter_name_or_path`, creates a new adapter, and uses `pref_beta: 0.1` with
the sigmoid DPO loss. This does not overwrite the SFT output. Validate the
preference export before launching:

```bash
python scripts/validate_dpo_inputs.py
```

The checked-in export is currently empty because no reviewed chosen/rejected
pairs have been produced yet. The launcher refuses to start until at least one
valid pair exists.

On a CUDA Linux or Colab training machine:

```bash
bash scripts/train_dpo.sh
```

On PowerShell:

```powershell
.\scripts\train_dpo.ps1
```

Both launchers require a CUDA-enabled PyTorch installation, `llamafactory-cli`,
and a non-empty preference dataset. They pass additional LLaMA-Factory
arguments through to the selected YAML. DPO is the preference optimization
method here; no reward-model or traditional RLHF stage is used.

Fine-tuning changes response behavior and style. Current and legal knowledge
remains in the existing RAG chunks and vector index and must continue to be
retrieved at inference time.

### LoRA versus QLoRA

The committed configuration is QLoRA with 4-bit bitsandbytes quantization,
`bf16`, gradient accumulation, and checkpoint saving every 100 steps. For
ordinary LoRA, remove `quantization_bit`, `quantization_method`,
`double_quantization`, and `optim: paged_adamw_8bit` from the YAML, then set
`optim: adamw_torch` or another optimizer supported by the chosen environment.
Keep the remaining rank, alpha, dropout, learning rate, sequence length, batch,
epoch, precision, and checkpoint settings in that YAML.
