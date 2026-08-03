# 16 · First 30 Days Execution Plan

This schedule is ordered by evidence dependency. A later task must not bypass an earlier gate merely to produce attractive results quickly.

---

## Week 1 · Trust the Environment and Clock

### Day 1 — Repository scaffold

Deliver:

- Python package and CLI skeleton;
- `pyproject.toml`, lock strategy, license, code style;
- schemas copied from this plan;
- CI parsing schemas/examples;
- issue and PR templates.

Acceptance:

```bash
edgeflow --help
python -m pytest
python scripts/validate_package.py
```

### Day 2 — Hardware fingerprint

Implement:

- `nvidia-smi` parser;
- PyTorch capability probe;
- OS/WSL/CPU/RAM/software capture;
- sanitized JSON + canonical SHA-256.

Test fixtures:

- normal RTX fingerprint;
- missing `nvidia-smi`;
- malformed output;
- CUDA unavailable;
- unsupported second GPU ignored or explicitly selected.

### Day 3 — Model/workload/plan contracts

Implement Pydantic/dataclass contracts corresponding to JSON Schemas. Add canonical JSON serialization and hash.

Acceptance:

- examples round-trip;
- unknown field rejected;
- invalid probability/range rejected;
- canonical hash stable.

### Day 4 — Timer calibration

Implement:

- wall clock boundaries;
- CUDA Event timer;
- synchronization policy;
- warmup state;
- raw iteration JSONL.

Run E01 negative controls. Write a short `TIMING_AUDIT.md`.

### Day 5 — Thermal and idle preflight

Implement GPU idle wait, process inventory, temperature/clock sampling, and drift rule. Run E02. Decide preliminary thresholds, labeled provisional until E03.

### Days 6–7 — Repetition study and cleanup

Run a 200-repetition reference on a smoke workload. Determine minimum repetition counts for screening and confirmatory runs. Review artifacts manually. Tag milestone:

```text
M0 Trust the Clock
```

Do not start runtime comparison if M0 fails.

---

## Week 2 · Build the First Fair Baseline

### Day 8 — Synthetic workload builder

Implement tokenizer-aware exact token buckets, fixed prompt IDs, output-length policy, arrival trace, and seed/hash capture.

Tests:

- exact length under each tokenizer;
- no accidental EOS;
- deterministic regeneration;
- Unicode and long-token edge cases.

### Days 9–10 — PyTorch eager adapter

Implement isolated worker lifecycle:

```text
probe → load → warmup → generate → metrics → shutdown
```

Capture model load, TTFT, token timestamps, TPOT, request latency, peak VRAM, and stdout/stderr.

### Day 11 — Correctness reference

Build logits/greedy fixtures and KV-cache parity checks. The eager path becomes the same-precision reference only after validation.

### Day 12 — Validation engine v0

Implement gates 1–6 from `edgeflow-validation/SKILL.md`:

- schema;
- environment;
- workload parity;
- correctness;
- timing;
- stability.

### Days 13–14 — E04 baseline and report

Run preregistered screening matrix on the primary 3B model. Produce the first artifact-backed HTML/Markdown report. No backend ranking yet.

Tag:

```text
M1a PyTorch Baseline
```

---

## Week 3 · Add Runtime Diversity Without Losing Fairness

### Days 15–16 — `torch.compile`

Implement prepare/cache/compile accounting, mode field, graph-break artifacts, and recompilation count. Run a small E05/E06 matrix.

Required result types:

- cold start;
- first request;
- steady state;
- dynamic sequence;
- break-even request count.

### Days 17–18 — llama.cpp

Implement process adapter around pinned `llama-cli`, `llama-server`, and `llama-bench` as appropriate. Normalize prompt/token/generation boundaries. Add GGUF provenance.

### Day 19 — Cross-runtime token parity

Compare prompt token IDs and generation settings. Document any unavoidable tokenizer/template difference. Do not compare incompatible pairs.

### Days 20–21 — vLLM capability and adapter

Implement server lifecycle, readiness probe, request trace, queue/TTFT/ITL metrics, and cleanup. Start with smoke model before primary model.

Tag only when at least one primary model has a valid three-runtime comparison:

```text
M1 Compare Fairly
```

---

## Week 4 · Establish EdgeFlow’s Differentiator

### Day 22 — Candidate and objective engine

Implement:

- capability pruning;
- analytical VRAM bound;
- objective profiles;
- session amortization;
- plan eligibility.

Unit tests must show that changing session request count can change the winner.

### Day 23 — Fixed plan and oracle analysis

Run E10 on the screening grid. Compute:

- best fixed plan;
- per-bucket oracle;
- expected regret;
- whether conditioned policy has meaningful headroom.

This is the motivation checkpoint. If one plan dominates, expand workloads or reconsider the policy claim rather than forcing it.

### Day 24 — Profiler summary adapter

Add `torch.profiler` and one Nsight Systems path. Produce normalized fields such as kernel gap ratio, launch count, active GPU ratio, and top-k kernel shares.

### Day 25 — Deterministic diagnosis

Implement the first three bottleneck rule groups:

- launch overhead;
- memory movement;
- compute-heavy prefill.

Each diagnostic output must name observed fields and a falsifiable intervention.

### Days 26–27 — First matched intervention

Run E13 or the profile-selected alternative:

1. preregister hypothesis;
2. randomize baseline/intervention order;
3. collect profile mediator separately;
4. collect unprofiled outcome;
5. validate correctness and statistics;
6. mark supported, rejected, or inconclusive.

### Day 28 — Evidence graph and policy v0

Persist Observation → Hypothesis → Intervention → Outcome → Scope. Generate a two-rule hand policy and an untouched holdout split.

### Day 29 — Dashboard wiring

Make the Local-first Web App the primary interface. Wire the localhost API, registered workload builder, typed single-GPU job queue, cancel/failure state, artifact drawer and source-type badges. Keep demo fixtures isolated under `ui-prototype/`; never inject them into production surfaces.

### Day 30 — Internal release review

Use the release checklist. Present:

- a trustworthy eager baseline;
- a fair runtime comparison;
- cold-vs-steady selection;
- one complete causal evidence chain;
- a preliminary workload-conditioned policy;
- all unresolved failures.

Tag:

```text
M2 Explain the Bottleneck · v0.0.1-internal
```

---

## After Day 30

Priority order:

1. full E10–E20 policy/causal evaluation;
2. real-distribution UltraChat replay;
3. quality confirmatory evaluation;
4. profile-selected Triton optimization;
5. holdout policy release;
6. local-first dashboard, optional read-only public export, and GitHub release;
7. optional cost model;
8. optional Copilot;
9. later FAD integration.

The 30-day outcome is not expected to contain every final experiment. It must establish trustworthy infrastructure and demonstrate one end-to-end example of EdgeFlow’s unique method.
