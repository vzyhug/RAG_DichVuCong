#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-${ROOT_DIR}/training/configs/dpo_qwen_lora.yaml}"
if [[ $# -gt 0 ]]; then shift; fi
PREFERENCE_DATASET="${PREFERENCE_DATASET:-${ROOT_DIR}/data/training/exports/preference_dataset.jsonl}"

cd "${ROOT_DIR}"
python scripts/validate_dpo_inputs.py --preference "${PREFERENCE_DATASET}"
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
  echo "DPO requires a CUDA-enabled PyTorch environment; refusing a CPU training attempt." >&2
  exit 1
}
command -v llamafactory-cli >/dev/null 2>&1 || {
  echo "llamafactory-cli was not found. Install LLaMA-Factory in the GPU environment first." >&2
  exit 127
}

exec llamafactory-cli train "${CONFIG}" "$@"
