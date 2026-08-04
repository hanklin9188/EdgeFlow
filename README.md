<div align="center">

<img src="docs/assets/banner.svg" alt="EdgeFlow — evidence-backed autotuning for local LLM inference" width="100%">

<p>
  <a href="https://github.com/hanklin9188/EdgeFlow/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/hanklin9188/EdgeFlow/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python 3.11–3.13" src="https://img.shields.io/badge/python-3.11%20–%203.13-1e2d36?style=flat-square&logo=python&logoColor=78d5c8&labelColor=152129">
  <img alt="CUDA 13.0" src="https://img.shields.io/badge/CUDA-13.0-1e2d36?style=flat-square&logo=nvidia&logoColor=78d5c8&labelColor=152129">
  <img alt="Verified on RTX 4080 SUPER" src="https://img.shields.io/badge/verified%20on-RTX%204080%20SUPER-1e2d36?style=flat-square&labelColor=152129">
  <img alt="Control plane: localhost only" src="https://img.shields.io/badge/control%20plane-localhost%20only-1e2d36?style=flat-square&labelColor=152129">
  <a href="LICENSE"><img alt="Apache 2.0" src="https://img.shields.io/badge/license-Apache%202.0-1e2d36?style=flat-square&labelColor=152129"></a>
</p>

**English** · [繁體中文](README.zh-TW.md) · [Docs](docs/) · [Implementation status](docs/IMPLEMENTATION_STATUS.md)

</div>

---

EdgeFlow turns local LLM deployment from ad-hoc benchmark chasing into an **evidence-backed, workload-conditioned optimization process**.

It does not assume that one runtime or one quantization is always fastest. EdgeFlow preserves every per-request raw measurement, pushes it through correctness, timing, stability, statistics, quality and provenance gates, derives *falsifiable* bottleneck hypotheses from profiler observations, and only then builds a deployment policy — using nothing but runs that survived validation.

```
Observe  →  Diagnose  →  Intervene  →  Verify  →  Synthesize policy
```

> [!IMPORTANT]
> `examples/` and `ui-prototype/` are permanently labelled `demo` and may never support a performance conclusion. The shipping product is a **localhost-only, local-first web app** — models, GPU work and artifacts never touch the cloud. This README publishes **no performance numbers** until confirmatory and holdout experiments are complete.

---

## Why this is different

|                              | Typical benchmark repo             | EdgeFlow                                                              |
| ---------------------------- | ---------------------------------- | --------------------------------------------------------------------- |
| **Unit of truth**            | A summary table                    | Per-request raw JSONL, re-derived on every validation                 |
| **Workload**                 | One prompt length, one batch       | Exact-token distributions, concurrency and session length as inputs   |
| **Cold vs steady**           | Blended into one number            | Cold start, compile, capture and steady state stored separately       |
| **Cause**                    | Asserted from a chart              | `HYPOTHESIS` until a matched intervention plus mediator confirms it   |
| **Quality**                  | Footnote, or absent                | A hard gate — a quantized plan cannot buy latency with accuracy       |
| **Recommendation**           | "X is faster"                      | A scoped decision list that reports `STALE` when the fingerprint moves |

---

## The Local Control Console

<div align="center">
<img src="docs/assets/console-preview.svg" alt="EdgeFlow Local Control Console" width="100%">
</div>

One command, one browser tab, one machine:

```bash
edgeflow serve --host 127.0.0.1 --port 8787
```

The console can start and stop pinned llama.cpp / vLLM builds, define a workload, screen candidates, submit **one** controlled GPU benchmark, cancel the local worker, and read runtime capability, validation verdicts, raw artifacts, evidence and policy. The OpenAPI contract lives at `/openapi.json`, Prometheus metrics at `/metrics`.

Port `8787` is EdgeFlow's reserved default so it never collides with other localhost services in the workspace.

<details>
<summary><strong>Security posture of the control plane</strong></summary>

- One managed runtime and one GPU job at a time.
- Managed runtimes bind `127.0.0.1` with a random API key held only in process memory.
- Control writes additionally require a token that exists only in page memory.
- The server rejects non-loopback hosts, cross-origin writes, bodies over 1 MiB, and any shell, path or environment argument supplied by the browser.
- A public site is never a control plane. If GitHub Pages ever publishes results, it may carry only sanitised, validated, read-only JSON and charts.

</details>

---

## Quickstart

**Requirements** — Python 3.11–3.13, an NVIDIA driver, and CUDA-enabled PyTorch. WSL2 Ubuntu or native Ubuntu recommended.

```bash
uv sync --extra dev
source .venv/bin/activate
edgeflow doctor
pytest -q
```

The GPU data plane needs an existing CUDA PyTorch/Transformers environment, or:

```bash
uv sync --extra dev --extra gpu
```

### 1 · Inspect

*Fingerprint the machine before anything is measured.*

```bash
edgeflow inspect
edgeflow inspect --json --output artifacts/hardware_fingerprint.json
edgeflow doctor
```

### 2 · Define a workload

*Prompt length, output length, concurrency and session length are explicit inputs — not defaults.*

```bash
edgeflow workload create \
  --model smollm2-360m-instruct \
  --profile local-agent \
  --prompt-distribution 32 \
  --output 8 \
  --concurrency 1 \
  --session-requests 30 \
  --save configs/generated/smoke-workload.json
```

`--prompt-distribution` also accepts mixtures such as `512:0.25,1024:0.45,2048:0.30`. Every request is generated to the exact token count under the *target* tokenizer.

### 3 · Screen capabilities

*Capability, memory and duplicate pruning only — the output is candidates, never a recommendation.*

```bash
edgeflow tune screen \
  --workload configs/generated/smoke-workload.json \
  --parameter-count 360000000 \
  --save artifacts/planned_candidates.json
```

### 4 · Benchmark

*One isolated run, one artifact directory.*

```bash
edgeflow benchmark run \
  --model-ref HuggingFaceTB/SmolLM2-360M-Instruct \
  --workload configs/smoke/workload.json \
  --plan configs/smoke/pytorch-eager.json \
  --repetitions 30 \
  --warmup 5 \
  --experiment-id E04
```

```
artifacts/<run_id>/
├── run_manifest.json
├── hardware_fingerprint.json
├── workload.json
├── execution_plan.json
├── metrics.jsonl              ← per-request raw rows
├── stdout.log
├── stderr.log
├── validation_verdict.json
└── VALIDATION.md
```

Formal policy eligibility also demands correctness and quality artifacts. A bare latency smoke is expected to land on `CONDITIONAL_PASS` — and it is never dressed up as a finished conclusion.

### 5 · Validate and diagnose

*Profiled latency never overwrites unprofiled production timing.*

```bash
edgeflow validate artifacts/<run_id>
edgeflow profile --run <run_id> --level nsys
edgeflow diagnose --profile examples/sample_profiler_summary.json
python scripts/verify_results.py
```

### 6 · Validate the Triton path

*No cached `PASS` for this kernel / GPU / dtype / shape → the PyTorch reference runs, always.*

```bash
edgeflow kernel validate
edgeflow kernel validate --full
python scripts/benchmark_rmsnorm.py --quick
```

### 7 · Start the console

*Optionally build isolated, pinned runtimes first so they cannot contaminate each other.*

```bash
./scripts/bootstrap_llama_cpp.sh
./scripts/bootstrap_vllm.sh
edgeflow serve --host 127.0.0.1 --port 8787
```

---

## Architecture

<div align="center">
<img src="docs/assets/architecture.svg" alt="EdgeFlow architecture — control plane, data plane and evidence plane" width="100%">
</div>

Full dependency design for the control, data and presentation planes: [System Architecture](docs/01_SYSTEM_ARCHITECTURE.md).

---

## Validation gates

<div align="center">
<img src="docs/assets/validation-gates.svg" alt="EdgeFlow validation gates G0 to G8" width="100%">
</div>

| Gate | Enforced invariant |
| --- | --- |
| **G0** Schema | Required artifacts, schemas, IDs, canonical hashes, raw JSONL |
| **G1** Environment | GPU scope, and no unapproved background GPU process |
| **G2** Correctness | Reference parity / no NaN / kernel contract |
| **G3** Timing | Warmup split, monotonic timestamps, no profiler contamination |
| **G4** Stability | Robust CV, and first-vs-last-third drift ≤ 3% |
| **G5** Statistics | ≥ 30 engine requests, or ≥ 100 kernel iterations |
| **G6** Quality | Profile-specific hard gate; quantized plans cannot bypass it |
| **G7** Provenance | Pinned revision, exact command, git and source state |
| **G8** Eligibility | Only measured `PASS` runs may enter a policy |

Verdicts are `PASS`, `CONDITIONAL_PASS`, `FAIL`, `INVALID` and `SKIPPED`. `FAIL` and `INVALID` raw artifacts stay on disk for audit.

---

## Evidence and claim rules

- `demo`, `estimated` and profiled latency never appear in a headline.
- A plan that fails correctness or quality is never ranked.
- Cold start, compile, capture and steady state are stored separately.
- Engine-only and HTTP end-to-end results are never ranked against each other.
- Bottleneck diagnosis is a `HYPOTHESIS`; only a matched intervention plus a mediator promotes it.
- When the runtime, driver or model fingerprint changes, the policy reports `STALE` and falls back.
- RTX 4080 SUPER results are never extrapolated to multi-GPU or datacenter GPUs.

---

## What is built, and what is not

**Engineering, complete on this checkout**

- RTX 4080 SUPER hardware/software fingerprint and an environment doctor.
- Pydantic + JSON Schema contracts for workload, plan, manifest, metric, profile, evidence, verdict and policy.
- PyTorch eager / `torch.compile` data plane; llama.cpp and vLLM OpenAI-compatible adapters with capability-safe skip.
- Exact target-tokenizer synthetic prompts, isolated run artifacts, per-request JSONL, SQLite index.
- G0–G8 validation engine, robust statistics, 10,000-sample paired bootstrap, thermal and drift checks.
- Deterministic bottleneck diagnosis, controlled-intervention drafts, session-aware objectives, decision-list policy.
- Correctness-cached Triton fused residual + RMSNorm, with automatic fallback for any unvalidated shape.
- Typer CLI, localhost FastAPI, isolated background worker, allowlisted runtime service manager, Prometheus endpoint, a local-first web app with no fake data, tests and CI.

> [!NOTE]
> **Research conclusions are deliberately not claimed yet.** Cross-runtime verdicts for the primary 3B model, quality datasets, matched interventions, holdout replay and end-to-end kernel integration still require formal GPU experiments. See [Implementation Status](docs/IMPLEMENTATION_STATUS.md) for the phase-by-phase split between *engineering done* and *experiment outstanding*.

Research design lives in the [Executive Blueprint](docs/00_EXECUTIVE_BLUEPRINT.md) and the [Experiment Catalog](docs/03_EXPERIMENT_CATALOG.md).

---

## Repository map

```
src/edgeflow/
├── api/             localhost FastAPI control/read surface
├── cli/             Typer commands
├── core/            immutable contracts and canonical hashing
├── experiments/     isolated run orchestrator
├── hardware/        RTX/CUDA/software fingerprint + doctor
├── kernels/         correctness-gated Triton optimization
├── local/           typed single-GPU job + allowlisted runtime managers
├── metrics/         robust statistics and paired bootstrap
├── optimizer/       pruning, objectives, break-even
├── policy/          explainable scoped decision lists
├── profiler/        bounded diagnosis rules
├── runtimes/        PyTorch, compile, llama.cpp, vLLM adapters
├── storage/         SQLite migrations and evidence index
├── validation/      G0–G8 final authority
└── workloads/       exact-token controlled inputs
```

---

## CLI surface

```
edgeflow inspect [--json]
edgeflow doctor [--strict-optional]
edgeflow workload create ...
edgeflow tune screen --workload ...
edgeflow benchmark run --model-ref ... --workload ... --plan ...
edgeflow experiment plan E05
edgeflow profile --run <run_id> --level torch|nsys|ncu
edgeflow diagnose --profile <profiler_summary.json>
edgeflow validate <artifact_dir>
edgeflow policy build --results <eligible_rows.json> ...
edgeflow policy show <policy.json>
edgeflow kernel validate [--full]
edgeflow serve
```

---

## Documentation

| | |
| --- | --- |
| [00 · Executive Blueprint](docs/00_EXECUTIVE_BLUEPRINT.md) | Why the project exists and what counts as success |
| [01 · System Architecture](docs/01_SYSTEM_ARCHITECTURE.md) | Control, data and presentation planes |
| [02 · Experiment Master Plan](docs/02_EXPERIMENT_MASTER_PLAN.md) | How E00–E30 fit together |
| [03 · Experiment Catalog](docs/03_EXPERIMENT_CATALOG.md) | Every experiment, its hypothesis and its exit criteria |
| [05 · Autotuning & Causal Method](docs/05_AUTOTUNING_AND_CAUSAL_METHOD.md) | Interventions, mediators, policy synthesis |
| [06 · Profiling & Kernel Plan](docs/06_PROFILING_AND_KERNEL_PLAN.md) | Nsight workflow and the Triton path |
| [07 · Validation & Statistics](docs/07_VALIDATION_AND_STATISTICS.md) | Gates, robust statistics, paired bootstrap |
| [09 · UI/UX & Presentation](docs/09_UI_UX_GITHUB_PRESENTATION.md) | Design language and presentation rules |
| [11 · Reproducibility & Security](docs/11_REPRODUCIBILITY_RELEASE_SECURITY.md) | Release, provenance and data handling |
| [Implementation Status](docs/IMPLEMENTATION_STATUS.md) | Engineering done vs experiment outstanding |

---

## Development

```bash
ruff check src tests
pytest -q
python scripts/validate_package.py
python scripts/verify_results.py
```

Public CI needs no GPU: it validates schemas, tests, lint, and the secret / model-weight policy. Full GPU sweeps run on a self-hosted runner or locally. Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Security and data

Model weights, tokens, `.env` files, gated prompt text, Nsight binary traces and private artifacts are never committed. HTTP adapters accept loopback OpenAI-compatible endpoints only; formal runs never use `shell=True` and never install packages mid-benchmark. See [SECURITY.md](SECURITY.md) and [Reproducibility](docs/11_REPRODUCIBILITY_RELEASE_SECURITY.md).

---

## License and citation

EdgeFlow source is licensed under the [Apache License 2.0](LICENSE). Models, datasets and runtimes keep their own licenses; every formal run records the revision and the terms. Citation metadata: [CITATION.cff](CITATION.cff).

<div align="center">
<sub>Built and verified on a single NVIDIA GeForce RTX 4080 SUPER workstation.</sub>
</div>
