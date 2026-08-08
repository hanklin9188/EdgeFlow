# EdgeFlow Agent Operating Contract

EdgeFlow is both a software system and an experimental system. Treat those as two related but distinct work streams.

## Required context

Before changing code, experiments, validation, policy, claims, or UI evidence:

1. Read [`CONTEXT.md`](CONTEXT.md) for canonical domain language.
2. Read [`docs/agents/domain.md`](docs/agents/domain.md) for context-consumption rules.
3. Read the relevant ADRs under [`docs/adr/`](docs/adr/).
4. Read [`docs/17_ENGINEERING_AND_EXPERIMENT_GOVERNANCE.md`](docs/17_ENGINEERING_AND_EXPERIMENT_GOVERNANCE.md).
5. Locate the work item in [`docs/18_EXECUTION_BACKLOG.md`](docs/18_EXECUTION_BACKLOG.md) or its GitHub issue.

Use canonical terms exactly. Do not silently introduce synonyms for `formal run`, `diagnostic run`, `execution plan`, `observation`, `hypothesis`, `intervention`, `mediator`, `evidence record`, `eligible run`, `deployment policy`, or `public claim`.

## Two work streams

### Engineering stream

Use for schemas, APIs, storage, runtime adapters, UI, worker lifecycle, security, and deterministic validation code.

Required sequence:

```text
Design boundary → tracer-bullet ticket → red test → minimal implementation
→ full test suite → two-axis review → commit
```

- Design with [`skills/edgeflow-change-design/SKILL.md`](skills/edgeflow-change-design/SKILL.md).
- Split work with [`skills/edgeflow-ticketing/SKILL.md`](skills/edgeflow-ticketing/SKILL.md).
- Test behavior through public seams. Prefer one vertical slice at a time.
- Review with [`skills/edgeflow-code-review/SKILL.md`](skills/edgeflow-code-review/SKILL.md).

### Scientific stream

Use for E00–E30, performance comparisons, profiler claims, policy synthesis, kernel claims, and public result statements.

Required sequence:

```text
Preregister → capability screen → isolate → warm up → measure raw rows
→ validate G0–G8 → diagnose as hypothesis → matched intervention
→ mediator check → confirmatory holdout → claim audit
```

- Execute with `skills/experiment-runner/SKILL.md`.
- Certify with `skills/edgeflow-validation/SKILL.md`.
- Diagnose with `skills/profiler-diagnosis/SKILL.md`.
- Debug regressions with [`skills/edgeflow-performance-debugging/SKILL.md`](skills/edgeflow-performance-debugging/SKILL.md).

A passing unit test never substitutes for a formal experiment. A formal experiment never substitutes for regression tests around engineering behavior.

## Hard evidence rules

- Per-request raw records are the source of truth.
- Profiled latency is diagnostic and never ranked as production timing.
- Correctness and quality are hard policy gates.
- A profiler observation remains an `HYPOTHESIS` until a matched intervention changes the expected mediator and an unprofiled outcome.
- Search and confirmatory holdout data must be separated.
- Only a measured `PASS` run may enter policy synthesis.
- Public claims require hardware, model revision, workload scope, metric boundary, uncertainty, quality status, and supporting run IDs.
- Demo, estimated, stale, invalid, failed, or conditional artifacts never support headline claims.

The machine-readable authority for validation coverage is [`specs/validation_requirements.yaml`](specs/validation_requirements.yaml). Run:

```bash
python scripts/audit_validation_parity.py
```

before changing validation rules, evidence promotion, policy eligibility, or release claims.

## Performance debugging

When behavior is slow, flaky, or regresses, do not begin with a theory. First build one tight, deterministic, red-capable command that reproduces the exact symptom. Minimize it, rank multiple falsifiable hypotheses, instrument only discriminating signals, and preserve the final reproducer as a regression test or benchmark fixture.

## Architecture discipline

Favor deep modules with small public seams. Periodically inspect recently changed hot spots, especially:

- run orchestration;
- runtime lifecycle;
- validation certification;
- evidence promotion;
- policy synthesis.

Record an ADR only when the decision is hard to reverse, surprising without context, and reflects a real trade-off.

## Agent safety

- Never execute unrestricted shell supplied by model output or browser input.
- Never install packages during a formal run.
- Never change driver, clocks, power limits, BIOS, model revision, workload, or thresholds mid-run.
- Never edit raw artifacts in place. Derivations must create new artifacts with transformation provenance.
- Never promote evidence or approve a claim based on prose alone; use executable checks and artifacts.
