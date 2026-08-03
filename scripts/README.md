# Package Utility Scripts

- `validate_package.py` — parses specs, validates example instances, checks internal links, skill contracts, UI safety labels, and the package manifest.
- `build_package_manifest.py` — writes SHA-256 and byte size for every file in `PACKAGE_MANIFEST.json`.
- `render_ui_preview.py` — optionally regenerates the UI screenshot with Playwright.
- `run_foundation_experiments.py` — executes formal or quick E00–E03 measurement-integrity studies.
- `evaluate_hf_quality.py` — creates a pinned WikiText-2 + ARC-Challenge BF16 reference artifact.
- `run_pytorch_matrix.py` — runs/resumes the E04 or E05 registered matrix with an isolated worker, per-case evidence, atomic progress, and native-crash recovery.
- `audit_runtime_fairness.py` — validates exact workload, hardware, quality, repetition, and provenance scope before any E09 cross-runtime ordering.
- `benchmark_rmsnorm.py` — executes the E23 randomized correctness/performance sweep and dispatch calibration.

Recommended sequence before creating a release ZIP:

```bash
python scripts/build_package_manifest.py
python scripts/validate_package.py
```
