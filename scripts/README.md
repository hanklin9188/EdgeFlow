# Package Utility Scripts

- `validate_package.py` — parses specs, validates example instances, checks internal links, skill contracts, UI safety labels, and the package manifest.
- `build_package_manifest.py` — writes SHA-256 and byte size for every file in `PACKAGE_MANIFEST.json`.
- `render_ui_preview.py` — optionally regenerates the UI screenshot with Playwright.
- `run_foundation_experiments.py` — executes formal or quick E00–E03 measurement-integrity studies.
- `evaluate_hf_quality.py` — creates a pinned WikiText-2 + ARC-Challenge BF16 reference artifact.
- `evaluate_openai_runtime_quality.py` — evaluates a loopback OpenAI-compatible runtime with exact token IDs against the same pinned WikiText-2 + ARC-Challenge protocol.
- `run_pytorch_matrix.py` — runs/resumes the E04 or E05 registered matrix with an isolated worker, per-case evidence, atomic progress, and native-crash recovery.
  Use `--rerun-case CASE_ID` for a surgical retry without replacing other settled failures.
- `run_dynamic_shape_study.py` — runs E06 `dynamic=False/auto/True` in isolated workers over the registered mixed-shape sequence and emits recompilation/cache/spike evidence plus a bounded shape rule.
- `audit_runtime_fairness.py` — validates exact workload, hardware, quality, repetition, and provenance scope before any E09 cross-runtime ordering.
- `run_vllm_bucket.py` — runs the registered Llama 3.2 3B BF16 `p1024-o128` E08 bucket after an exact-scope runtime quality report exists; the formal launcher pins the V1 eager single-sequence scheduler profile.
- `start_vllm_server.sh` — starts an allowlisted loopback-only vLLM profile; set `EDGEFLOW_VLLM_PROFILE=llama32-3b-bf16` for the formal Llama profile.
- `audit_formal_readiness.py` — computes strict E10/E19 analyses and audits the E25–E28 data/grounding prerequisites without promoting incomplete evidence.
- `run_cold_warm_study.py` — runs E20 in fresh isolated Python processes and compares cached-host time-to-first-usable against a same-process warmed response; machine-reboot, dropped-filesystem-cache, persisted-compile-cache, and external-service scopes remain explicit exclusions.
- `benchmark_rmsnorm.py` — executes the E23 randomized correctness/performance sweep and dispatch calibration.
- `run_e24_integration.py` — performs paired ABBA kernel-off/on Llama inference on search and untouched holdout prompts, verifies greedy output parity, and records explicit rollback/fallback evidence.

Recommended sequence before creating a release ZIP:

```bash
python scripts/build_package_manifest.py
python scripts/validate_package.py
```
