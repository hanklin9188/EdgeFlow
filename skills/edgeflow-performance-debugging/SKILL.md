---
name: edgeflow-performance-debugging
description: Diagnose EdgeFlow bugs, flaky runs, and performance regressions through a tight reproducible loop before proposing bottleneck hypotheses or fixes.
version: 1.0.0
---

# EdgeFlow Performance Debugging

## Purpose

Establish a trustworthy feedback loop for the exact failure before reading broadly, profiling indiscriminately, or choosing a fix. This skill resolves defects and regressions; formal causal evidence still requires the experiment and validation skills.

## Inputs

- exact user-observed symptom;
- affected commit/version and last known good state when available;
- redacted logs/artifacts;
- target hardware and workload;
- relevant ADRs and protocol.

## Workflow

### 1. Build a tight red-capable loop

Choose the narrowest real seam:

1. focused unit/integration test;
2. CLI fixture invocation;
3. localhost API request;
4. replay of captured run artifacts;
5. minimal benchmark harness;
6. old-vs-new differential run;
7. automated commit/config bisection;
8. repeated stress loop for a flaky failure.

The loop must:

- catch the exact symptom;
- produce a binary or quantitative threshold verdict;
- be deterministic, or raise reproduction probability enough for debugging;
- finish fast enough for iteration;
- run without manual interpretation.

Completion: one command has been run and can go red on the target symptom.

### 2. Reproduce and minimize

Confirm the exact failure, then remove one input, setting, process, or dependency at a time. Keep only load-bearing elements.

For performance, freeze:

- model/revision;
- prompt IDs;
- output protocol;
- backend and plan except the suspected factor;
- GPU precondition;
- timing boundary;
- repetitions.

Completion: removing any remaining element makes the loop green or changes the target symptom.

### 3. Rank multiple falsifiable hypotheses

Produce 3–5 hypotheses. Each states a prediction:

```text
If X is the cause, changing Y should move mediator Z and the symptom.
```

Include an alternative explanation and an irrelevant/negative-control intervention when practical.

Completion: every hypothesis has a discriminating prediction.

### 4. Instrument only discriminating signals

Prefer:

- debugger or direct state inspection;
- focused structured logs;
- internal timing/CUDA events;
- Nsight Systems for timeline;
- Nsight Compute only for a selected kernel;
- bisection for regressions.

Do not collect every available counter. Tag temporary instrumentation for cleanup.

Completion: each probe rules hypotheses in or out.

### 5. Fix or run a matched intervention

Engineering defect:

- turn the minimized case into a red regression test at the public seam;
- implement the smallest fix;
- rerun minimized and original loops.

Performance explanation:

- preregister the intervention;
- pair baseline/intervention rows;
- verify expected mediator;
- confirm outcome unprofiled;
- invoke EdgeFlow validation.

Completion: exact symptom is gone or hypothesis receives supported/rejected/inconclusive status.

### 6. Cleanup and postmortem

- remove temporary probes;
- preserve the regression fixture;
- state the winning explanation in commit/PR;
- record environment/version scope;
- identify missing seam or architecture debt;
- update the validation matrix if the defect exposed a certification gap.

## Never Do

- Never begin with “GPU utilization is low, therefore CPU bottleneck.”
- Never profile the entire stack before a red-capable loop exists.
- Never use profiled latency as the final performance result.
- Never change multiple plan factors and call the result causal.
- Never keep a fix that passes the minimized case but fails the original loop.
- Never expose credentials, auth headers, private prompts, or profiler binaries in reports.
