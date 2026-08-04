#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
vllm="${project_root}/.runtime/vllm/.venv/bin/vllm"
profile="${EDGEFLOW_VLLM_PROFILE:-smoke}"
case "${profile}" in
  smoke)
    model="HuggingFaceTB/SmolLM2-360M-Instruct"
    revision="a10cc1512eabd3dde888204e902eca88bddb4951"
    dtype="half"
    memory_utilization="0.35"
    served_model="smollm2-edgeflow"
    ;;
  llama32-3b-bf16)
    model="meta-llama/Llama-3.2-3B-Instruct"
    revision="0cb88a4f764b7a12671c53f0838cd831a0843b95"
    dtype="bfloat16"
    memory_utilization="0.75"
    served_model="llama32-3b-edgeflow"
    ;;
  *)
    echo "Unsupported EDGEFLOW_VLLM_PROFILE: ${profile}" >&2
    exit 2
    ;;
esac

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
  --dtype "${dtype}" \
  --max-model-len 4096 \
  --gpu-memory-utilization "${memory_utilization}" \
  --served-model-name "${served_model}" \
  --generation-config vllm \
  --disable-fastapi-docs \
  "${authentication[@]}"
