#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-${ROOT_DIR}/training/configs/sft_qwen_lora.yaml}"
shift || true

cd "${ROOT_DIR}"
command -v llamafactory-cli >/dev/null 2>&1 || {
  echo "llamafactory-cli was not found. Install LLaMA-Factory in the GPU environment first." >&2
  exit 127
}

exec llamafactory-cli train "${CONFIG}" "$@"
