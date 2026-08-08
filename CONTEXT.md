# EdgeFlow Domain Language

This glossary defines canonical EdgeFlow terms. It intentionally excludes implementation details. Code, issues, experiments, UI, skills, and reports should use these terms consistently.

## System and workload

### Hardware fingerprint

An immutable description of the machine and relevant software state under which a run occurred, including GPU, driver, CUDA, runtime, model revision, repository state, and measurement controls.

### Workload

The complete request behavior evaluated by EdgeFlow: prompt source or token distribution, output length, batch, concurrency, arrival pattern, sampling, streaming, seed, and session horizon.

### Workload bucket

A bounded region of workload space used to compare plans or scope a policy rule.

### Workload distribution

A probability distribution or replay trace over multiple workload buckets.

### Execution plan

A canonical, hashed runtime configuration containing backend, precision or quantization, compilation, shape, batching, cache, scheduler, and custom-kernel choices.

### Candidate plan

An execution plan that passed capability and deterministic pruning but has not necessarily passed measurement or validation.

### Deployment policy

A scoped decision list mapping workload conditions to eligible execution plans, with a validated fallback and evidence references.

### Stale policy

A deployment policy whose hardware, software, model, or protocol fingerprint no longer matches the current environment.

## Runs and artifacts

### Run

One immutable execution of one execution plan against one workload under one hardware fingerprint.

### Smoke run

A small run proving that a code path or runtime lifecycle functions. It does not support performance or quality claims.

### Formal run

A measured run executed under a preregistered experiment protocol and eligible for the complete validation path.

### Diagnostic run

A run used to inspect behavior or collect profiler evidence. It is not ranked as production latency.

### Profiled run

A diagnostic run executed with profiler instrumentation. Its timings are considered perturbed unless a protocol explicitly studies profiler overhead.

### Confirmatory run

A formal run using frozen settings and data not used to select the candidate or tune thresholds.

### Holdout run

A confirmatory run evaluated on workload samples or buckets excluded from search and policy construction.

### Raw record

One request-level or iteration-level measurement stored without aggregation. Raw records are the primary measurement evidence.

### Derived summary

A reproducible aggregation computed from raw records. It is never a substitute for raw records.

### Capacity boundary

A validated OOM, timeout, or resource limit defining where a plan cannot operate. It is not a speed result.

## Validation and evidence

### Validation gate

One executable class of checks in G0–G8 that decides whether a run is structurally complete, correctly measured, functionally valid, statistically sufficient, quality-compliant, and traceable.

### Verdict

One of `PASS`, `CONDITIONAL_PASS`, `FAIL`, `INVALID`, or `SKIPPED`.

### Eligible run

A measured `PASS` run allowed to enter policy synthesis for its validated scope.

### Observation

A directly measured fact, such as kernel-gap ratio, queue-delay share, peak VRAM, or request latency.

### Hypothesis

A falsifiable explanation for one or more observations. A hypothesis is not evidence of causality.

### Intervention

A preregistered, controlled change intended to test one hypothesis.

### Mediator

The intermediate system quantity expected to change if the proposed causal path is correct.

### Negative control

A condition or intervention expected not to produce the proposed mediator and outcome changes.

### Evidence record

An immutable link between observations, hypothesis, intervention, mediator result, outcome result, scope, artifacts, and validation verdict.

### Evidence level

The maturity of evidence from idea through repeated observation, intervention support, holdout confirmation, and cross-environment confirmation.

### Claim

A human-readable statement supported by evidence records. A claim must include scope and uncertainty.

### Public claim

A claim cleared for publication after protocol, provenance, licensing, privacy, and evidence audits.

## Kernel terms

### Kernel region

The exact GPU, dtype, shape, layout, and kernel-version region for which correctness and performance have been validated.

### Reference fallback

The known-correct implementation used whenever the current kernel region lacks a valid correctness certificate.

## Canonical distinctions

- `Smoke run` is not `formal run`.
- `Diagnostic run` is not `production timing`.
- `Observation` is not `hypothesis`.
- `Hypothesis` is not `evidence`.
- `Candidate plan` is not `eligible run`.
- `Derived summary` is not `raw record`.
- `Microbenchmark speedup` is not `end-to-end speedup`.
- `Engineering complete` is not `experiment complete`.
