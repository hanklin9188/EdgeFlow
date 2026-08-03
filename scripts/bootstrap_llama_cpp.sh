#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_root="${project_root}/.runtime/llama.cpp"
release="b10242"
commit="96278e39fc83e1d97c881e34bcec39ac7ea98820"

mkdir -p "${project_root}/.runtime"
if [[ ! -d "${runtime_root}/.git" ]]; then
  git clone --filter=blob:none --branch "${release}" --depth 1 \
    https://github.com/ggml-org/llama.cpp.git "${runtime_root}"
fi

if [[ -n "$(git -C "${runtime_root}" status --porcelain)" ]]; then
  echo "Refusing to alter a dirty generated llama.cpp checkout: ${runtime_root}" >&2
  exit 2
fi

actual_commit="$(git -C "${runtime_root}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${commit}" ]]; then
  echo "llama.cpp checkout is ${actual_commit}; expected ${commit}" >&2
  exit 2
fi

cmake -S "${runtime_root}" -B "${runtime_root}/build" -G Ninja \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=OFF
cmake --build "${runtime_root}/build" --target llama-server llama-cli llama-bench --parallel

"${runtime_root}/build/bin/llama-server" --version
sha256sum \
  "${runtime_root}/build/bin/llama-server" \
  "${runtime_root}/build/bin/llama-cli" \
  "${runtime_root}/build/bin/llama-bench" \
  > "${runtime_root}/BUILD_CHECKSUMS.sha256"
git -C "${runtime_root}" rev-parse HEAD > "${runtime_root}/BUILD_COMMIT"
cmake -LA -N "${runtime_root}/build" > "${runtime_root}/BUILD_CMAKE_CACHE.txt"
echo "llama.cpp ${release} is ready at ${runtime_root}/build/bin"
