# Contributing

Create a focused branch and include tests for behavior changes. Before opening a pull request, run:

```bash
ruff check src tests
pytest -q
python scripts/validate_package.py
python scripts/verify_results.py
```

Performance changes must include raw artifacts, exact hardware/software fingerprint, workload and plan hashes, correctness/quality status, and an unprofiled confirmatory result. Do not delete slow rows or relabel demo data as measured.
