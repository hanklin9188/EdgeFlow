#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_root="${project_root}/.runtime/vllm"
venv="${runtime_root}/.venv"
wheel="https://github.com/vllm-project/vllm/releases/download/v0.26.0/vllm-0.26.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl#sha256=6ce4ca30616f0a35810391015622b197a7b8b267ed27f8716f0789db79ff578b"

mkdir -p "${runtime_root}"
if [[ ! -x "${venv}/bin/python" ]]; then
  uv venv --python 3.12 --seed --managed-python "${venv}"
fi

uv pip install --python "${venv}/bin/python" "${wheel}" --torch-backend=cu129
"${venv}/bin/python" -c 'import torch, vllm; assert vllm.__version__ == "0.26.0"; print(f"vLLM {vllm.__version__} / torch {torch.__version__} / CUDA {torch.version.cuda}")'
uv pip freeze --python "${venv}/bin/python" > "${runtime_root}/requirements.freeze.txt"
echo "vLLM v0.26.0 is ready at ${venv}"
