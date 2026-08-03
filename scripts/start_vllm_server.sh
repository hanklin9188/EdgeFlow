#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
vllm="${project_root}/.runtime/vllm/.venv/bin/vllm"
model="HuggingFaceTB/SmolLM2-360M-Instruct"
revision="a10cc1512eabd3dde888204e902eca88bddb4951"

if [[ ! -x "${vllm}" ]]; then
  echo "vLLM is not installed; run scripts/bootstrap_vllm.sh first." >&2
  exit 2
fi

# vLLM V2 needs pinned memory on WSL2. Its bundled FlashInfer sampler does not
# compile against the current cu129 CCCL headers on SM89, so use the native sampler.
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_USE_FLASHINFER_SAMPLER=0
authentication=()
if [[ -n "${EDGEFLOW_RUNTIME_API_KEY:-}" ]]; then
  authentication=(--api-key "${EDGEFLOW_RUNTIME_API_KEY}")
fi

exec "${vllm}" serve "${model}" \
  --revision "${revision}" \
  --host 127.0.0.1 \
  --port "${EDGEFLOW_VLLM_PORT:-8002}" \
  --dtype half \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.35 \
  --served-model-name smollm2-edgeflow \
  --disable-fastapi-docs \
  "${authentication[@]}"
