# Experiment Report · `EXPERIMENT_ID`

## Decision Summary

| Field | Value |
|---|---|
| Decision | `ACCEPT / REJECT / INCONCLUSIVE` |
| Evidence level | `E0–E5` |
| Primary claim | `ONE PRECISE SENTENCE` |
| Scope | `HARDWARE / MODEL / BACKEND / WORKLOAD` |
| Supporting run IDs | `RUN_IDS` |
| Validation verdicts | `LINKS` |

## 1. Preregistered Question

**Question:**
`What exact uncertainty does this experiment resolve?`

**Hypothesis:**
`Directional, falsifiable statement.`

**Null / alternative:**
`What result would reject or weaken the hypothesis?`

**Primary outcome:**
`One metric and aggregation rule.`

**Mediator:**
`Profiler metric expected to change if the mechanism is correct.`

**Acceptance criterion:**

```text
Correctness PASS
AND quality constraint PASS
AND paired effect exceeds minimum practical effect
AND confidence interval excludes the equivalence region
AND mediator changes in the predicted direction
```

## 2. Design

| Dimension | Baseline | Intervention | Controlled? |
|---|---|---|---|
| Hardware fingerprint |  |  | yes |
| Model and tokenizer |  |  | yes |
| Prompt IDs and seed |  |  | yes |
| Prompt/output tokens |  |  | yes |
| Backend |  |  |  |
| Changed variable |  |  | **only intended factor** |
| Cache state |  |  | yes |
| Run order | randomized block | randomized block | yes |

Include the negative control and any intentionally uncontrolled nuisance factor.

## 3. Environment and Provenance

- Hardware fingerprint: `LINK`
- Git commit / dirty state: `VALUE`
- Model files hash: `VALUE`
- Runtime commit/version: `VALUE`
- Command: `VALUE`
- Protocol version: `VALUE`
- Raw artifact directory: `LINK`

## 4. Validation Gates

| Gate | Baseline | Intervention | Notes |
|---|---|---|---|
| Schema |  |  |  |
| Workload parity |  |  |  |
| Correctness |  |  |  |
| Timing integrity |  |  |  |
| Stability / thermal |  |  |  |
| Quality |  |  |  |
| Provenance |  |  |  |

## 5. Results

### Primary outcome

Report median, paired delta, percent change, confidence interval, raw sample count, failure rate, and practical-effect threshold.

### Mediator

Show whether the proposed mechanism changed. A latency improvement without the expected mediator change cannot automatically support the causal explanation.

### Distribution

Include ECDF or violin/box-free distribution view, not only a mean bar.

## 6. Interpretation

Separate these paragraphs explicitly:

- **Measured facts:** direct artifact-backed observations.
- **Supported inference:** mechanism supported by intervention and mediator.
- **Remaining alternatives:** plausible unresolved explanations.
- **Generalization boundary:** where this conclusion must not be reused.

## 7. Failure Analysis

Record OOM, timeout, graph break, recompilation, fallback, thermal drift, and skipped samples. Never delete inconvenient runs.

## 8. Decision

`SUPPORTED / REJECTED / INCONCLUSIVE` and exact next action.

## 9. Reproduction

```bash
edgeflow experiment reproduce --experiment EXPERIMENT_ID --block BLOCK_ID
edgeflow validate runs/RUN_ID
```
