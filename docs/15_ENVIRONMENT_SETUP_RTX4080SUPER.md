# 15 · RTX 4080 SUPER Environment Setup and Version Strategy

## 15.1 Recommended topology

For a Windows desktop, use:

```text
Windows 11 host
├── NVIDIA Windows display driver
├── Browser / VS Code / optional native Nsight Systems
└── WSL2 Ubuntu
    ├── EdgeFlow core
    ├── PyTorch + Triton environment
    ├── separate vLLM environment
    ├── llama.cpp CUDA build
    └── model/data/cache on the Linux filesystem
```

Prefer WSL2 for the Linux-first inference ecosystem. Keep a native Ubuntu path documented as a future cross-environment validation target.

### Critical WSL rule

Install the NVIDIA display driver on Windows. Do **not** install a Linux NVIDIA display driver inside WSL. CUDA is exposed to WSL by the Windows driver. Install a toolkit-only package inside WSL only when compilation tools such as `nvcc` are required.

---

## 15.2 Filesystem layout

Do not run heavy model I/O under `/mnt/c` unless it is an explicit filesystem experiment. Use the WSL ext4 filesystem:

```text
~/edgeflow-workspace/
├── src/edgeflow/
├── envs/
├── models/
│   ├── hf/
│   └── gguf/
├── datasets/
├── caches/
│   ├── huggingface/
│   ├── torchinductor/
│   ├── triton/
│   └── vllm/
├── runs/
├── traces/
└── scratch/
```

Environment variables:

```bash
export EDGEFLOW_ROOT="$HOME/edgeflow-workspace"
export HF_HOME="$EDGEFLOW_ROOT/caches/huggingface"
export TORCHINDUCTOR_CACHE_DIR="$EDGEFLOW_ROOT/caches/torchinductor"
export TRITON_CACHE_DIR="$EDGEFLOW_ROOT/caches/triton"
export EDGEFLOW_RUNS="$EDGEFLOW_ROOT/runs"
export EDGEFLOW_TRACES="$EDGEFLOW_ROOT/traces"
```

Do not commit absolute local paths; manifests may store redacted logical roots and hashes.

---

## 15.3 Host preflight

On Windows PowerShell:

```powershell
wsl --status
wsl --version
nvidia-smi
```

Inside WSL:

```bash
uname -a
cat /etc/os-release
nvidia-smi
ls -l /usr/lib/wsl/lib/libcuda.so.1
```

Expected behavior:

- RTX 4080 SUPER visible in WSL;
- CUDA driver API available;
- no separately installed Linux NVIDIA kernel driver;
- sufficient disk space for multiple model formats and traces.

Capture exact outputs in the hardware fingerprint, but sanitize username and machine-specific private paths before publishing.

---

## 15.4 Environment separation

Do not force every runtime into one Python environment. Binary compatibility changes rapidly, especially around PyTorch, CUDA, Triton, and vLLM.

Recommended environments:

```text
.edgeflow-core   CLI, schemas, analysis, dashboard, tests
.edgeflow-torch  pinned PyTorch, Transformers, Triton
.edgeflow-vllm   vLLM-selected compatible PyTorch/CUDA stack
llama.cpp        independent CMake build and pinned commit
```

The orchestrator communicates with backend workers through subprocess JSON/JSONL or local HTTP. This isolation makes failures and version conflicts observable rather than hidden.

### Version pin policy

At implementation time:

1. Select one stable PyTorch CUDA wheel from the official selector.
2. Record Python, PyTorch, CUDA runtime, Transformers, Accelerate, and Triton versions.
3. Freeze with `uv.lock` or a hash-pinned requirements export.
4. Install vLLM in a fresh environment using its current official compatibility instructions.
5. Pin llama.cpp to an audited commit, not `master` without a hash.
6. Record compiler, CMake, CUDA toolkit, and build flags.
7. Never compare results across environment revisions without treating the revision as an experimental factor.

---

## 15.5 Core environment bootstrap

Illustrative commands; generate exact package versions at implementation time:

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake ninja-build git git-lfs pkg-config \
  python3-dev python3-venv jq sqlite3 libopenblas-dev

curl -LsSf https://astral.sh/uv/install.sh | sh
cd "$EDGEFLOW_ROOT/src/edgeflow"
uv venv .venv-core --python 3.12
source .venv-core/bin/activate
uv pip install -e '.[dev]'
```

Security rule: review install scripts before use in a public project. For a reproducible release, replace moving installer URLs with documented manual or pinned methods.

---

## 15.6 PyTorch and Triton worker

Use the command produced by the official PyTorch selector for Linux, pip, and the desired CUDA build. Then verify:

```bash
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda build', torch.version.cuda)
print('available', torch.cuda.is_available())
print('gpu', torch.cuda.get_device_name(0))
print('capability', torch.cuda.get_device_capability(0))
x = torch.randn(2048, 2048, device='cuda', dtype=torch.float16)
y = x @ x
print(float(y[0, 0]))
PY
```

Triton probe:

- vector add correctness against PyTorch;
- FP16/BF16 support;
- compile cache creation;
- repeated launch;
- record generated cache and compilation time separately.

Do not begin performance sweeps until the smoke test is represented by a validated run artifact.

---

## 15.7 llama.cpp CUDA build

Pin a commit and preserve build metadata:

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout <PINNED_COMMIT>
cmake -B build -G Ninja \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

Record:

- commit;
- CMake cache/build options;
- compiler version;
- CUDA architecture output;
- executable SHA-256;
- `llama-cli --version` and `llama-bench` output format.

Do not compare a locally converted GGUF against a different original checkpoint revision.

---

## 15.8 vLLM worker

Install vLLM in a fresh environment following the current official GPU installation page. The selected wheel constrains PyTorch and CUDA compatibility, so EdgeFlow must probe rather than assume support.

Capability probe:

1. import and version;
2. detect CUDA device;
3. load smoke model;
4. one greedy request;
5. streaming server request;
6. concurrency 2 smoke;
7. inspect supported attention backend and CUDA Graph mode;
8. capture failure as a capability artifact, not a benchmark result.

If WSL2, consumer GPU, or a model architecture is not stable in the pinned version, mark the pair `experimental` or `unsupported`; do not patch around it silently.

---

## 15.9 Profiler setup

### `torch.profiler`

Available inside the PyTorch worker. Use it for operator/CUDA activity and graph diagnosis. Never mix its overhead into production timings.

### Nsight Systems

Use for CPU/GPU timeline, CUDA API, kernel launches, copies, and CUDA Graph behavior. Depending on WSL and installed tooling, collection or viewing may be split between WSL and Windows.

### Nsight Compute

Use only for selected kernels after Nsight Systems identifies a hotspot. Check performance-counter permission before experiment registration. If counters are unavailable, return `PROFILE_UNAVAILABLE`; never synthesize bandwidth/utilization values.

Profiler artifacts may expose commands and paths. Sanitize public exports while retaining private full traces locally.

---

## 15.10 Measurement controls on a personal desktop

A workstation is not a dedicated benchmark server. EdgeFlow must explicitly manage nuisance variables:

- close games, video playback, GPU renderers, and unrelated CUDA processes;
- disable screen recording/overlays during formal runs;
- record display-attached GPU utilization;
- wait for a stable temperature band;
- record SM/memory clock and power;
- run plans in randomized blocks;
- separate AC power/battery and Windows power mode;
- avoid Windows Update or background model downloads;
- log whether browser and desktop compositor remain active.

Do not overclock for the primary result. Optional OC/undervolt experiments require separate fingerprints and safety boundaries.

---

## 15.11 Environment acceptance checklist

A formal matrix can start only when:

- [ ] Windows driver and WSL GPU path are healthy.
- [ ] Core, PyTorch, vLLM, and llama.cpp versions are pinned independently.
- [ ] PyTorch CUDA and Triton correctness pass.
- [ ] Each runtime has a capability report.
- [ ] Hardware fingerprint includes temperatures, clocks, power limit, software, and hashes.
- [ ] Timer calibration and idle/thermal preconditions pass.
- [ ] Model and dataset cache roots are on the intended filesystem.
- [ ] Profiler availability is known and permission failures are explicit.
- [ ] Clean environment recreation instructions are tested.

---

## 15.12 Never publish

- access tokens, OAuth files, Hugging Face tokens;
- complete local environment dumps containing secrets;
- gated/private dataset rows;
- Meta or other restricted model weights;
- absolute paths containing personal usernames when unnecessary;
- raw generated code execution artifacts without sandbox review;
- a profiler trace before inspecting strings and annotations.
