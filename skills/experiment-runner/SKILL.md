---
name: edgeflow-experiment-runner
description: Plan and execute reproducible EdgeFlow experiment blocks while preserving isolation, fairness, randomization, raw artifacts, and explicit failure states.
version: 1.0.0
---

# EdgeFlow Experiment Runner Skill

## Goal

Turn an experiment catalog entry into a complete, isolated, resumable run block. The runner executes; it does not reinterpret failed results or bypass validation.

## Inputs

- experiment ID and protocol version
- model registry entry
- workload matrix
- candidate plans
- hardware capability report
- random seed
- artifact root

## Preconditions

1. Hardware doctor passes.
2. Enough disk and VRAM safety estimate.
3. Model license/access accepted.
4. Plans are schema-valid and canonicalized.
5. No pending dirty artifact directory collision.
6. Validation configuration exists.

## Workflow

### 1. Expand Matrix

Expand only supported combinations. Save `planned_matrix.json` before executing.

Each row gets:

- immutable run ID;
- block ID;
- paired-group ID;
- randomized order;
- expected resources;
- status `PLANNED`.

### 2. Prune

Use only deterministic rules:

- unsupported backend/model;
- memory estimate beyond safe limit;
- duplicate plan;
- known failed kernel region;
- prohibited quality profile.

Record every pruned row and reason.

### 3. Randomize and Block

For matched A/B use ABBA or seed-randomized blocks. Keep model revision, prompt IDs, environment, and time window aligned.

### 4. Prepare Isolated Process

- set exact environment variables;
- write command preview;
- no `shell=True`;
- capture stdout/stderr;
- create temporary artifact directory;
- install no packages during a formal run.

### 5. Precheck

- GPU idle;
- temperature/clock policy;
- expected free VRAM;
- model files/hash;
- server ports free;
- no previous backend process;
- time synchronization plausible.

Precheck failure marks row `PRECHECK_FAILED`, not silently retried forever.

### 6. Warmup

Warmup until both minimum count and convergence criterion. Record all warmup observations separately.

### 7. Measure

- preserve each request/iteration;
- flush raw records incrementally;
- respect timeout;
- collect monitor samples;
- do not summarize-only;
- never reuse profiler-enabled process for production timing unless protocol explicitly requires and labels it diagnostic.

### 8. Shutdown

Graceful backend stop; then verify process exit and GPU memory release. Force-kill only after timeout and record.

### 9. Finalize Artifacts

Atomic move temporary directory to final run path. Generate checksums and manifest completion status.

### 10. Invoke Validation

Pass artifacts to `edgeflow-validation`. Do not add the run to policy/index until verdict is returned.

## Resume

Resume only unstarted rows. A partially measured row is not appended to; rerun with new run ID and link `supersedes`.

## Failure Handling

| Failure | Status | Action |
|---|---|---|
| OOM | FAILED_CAPACITY | preserve, clean process |
| CUDA illegal access | FAILED_RUNTIME | terminate process, reset worker |
| timeout | FAILED_TIMEOUT | preserve partial data, invalid timing |
| model load fail | FAILED_PREPARE | backend support issue |
| server readiness fail | FAILED_PREPARE | logs |
| validation fail | completed run, not eligible | no rerun unless cause fixed |

## Never Do

- do not change parameters mid-run;
- do not reduce output length after an OOM and keep same run ID;
- do not skip slow rows;
- do not overwrite raw artifacts;
- do not use a different prompt set for a plan because it is faster;
- do not launch Copilot on the GPU during benchmark.
