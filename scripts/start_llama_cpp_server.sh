#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
server="${project_root}/.runtime/llama.cpp/build/bin/llama-server"
repo="mistralai/Ministral-3-3B-Instruct-2512-GGUF"
revision="eb599d408350ea2bb60452cb86be7c7b2fc28227"
quantization="${EDGEFLOW_LLAMA_CPP_QUANTIZATION:-Q4_K_M}"
case "${quantization}" in
  Q4_K_M|Q5_K_M|Q6_K|Q8_0) ;;
  *)
    echo "Unsupported EDGEFLOW_LLAMA_CPP_QUANTIZATION: ${quantization}" >&2
    exit 2
    ;;
esac
filename="Ministral-3-3B-Instruct-2512-${quantization}.gguf"
alias_quantization="$(printf '%s' "${quantization}" | tr '[:upper:]' '[:lower:]')"

if [[ ! -x "${server}" ]]; then
  echo "llama.cpp is not installed; run scripts/bootstrap_llama_cpp.sh first." >&2
  exit 2
fi
if ! command -v hf >/dev/null 2>&1; then
  echo "The Hugging Face 'hf' CLI is required to resolve the pinned model." >&2
  exit 2
fi

export HF_HUB_DISABLE_VERSION_CHECK=1
model_path="$(hf download "${repo}" "${filename}" --revision "${revision}" --format quiet)"
authentication=()
if [[ -n "${EDGEFLOW_RUNTIME_API_KEY:-}" ]]; then
  authentication=(--api-key "${EDGEFLOW_RUNTIME_API_KEY}")
fi

exec "${server}" \
  --model "${model_path}" \
  --alias "edgeflow-ministral3-3b-${alias_quantization}" \
  --host 127.0.0.1 \
  --port "${EDGEFLOW_LLAMA_CPP_PORT:-8001}" \
  --n-gpu-layers 99 \
  --ctx-size 4096 \
  --parallel 1 \
  --cont-batching \
  --no-webui \
  "${authentication[@]}"
