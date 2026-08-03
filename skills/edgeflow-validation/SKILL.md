---
name: edgeflow-validation
description: Validate EdgeFlow benchmark, profiler, quality, and custom-kernel runs before they can enter comparisons, evidence graphs, policies, reports, or GitHub headline claims.
version: 1.0.0
---

# EdgeFlow Validation Skill

## Purpose

This skill is the final authority for whether an EdgeFlow run is usable. It does not optimize plans and does not explain away failures. It checks that a result is structurally complete, functionally correct, fairly timed, statistically sufficient, quality-compliant, and fully traceable.

The skill must prefer an honest `INVALID`, `FAIL`, or `INCONCLUSIVE` result over a persuasive but unsupported performance claim.

---

## Inputs

Required:

- `run_manifest.json`
- `workload.json`
- `execution_plan.json`
- `hardware_fingerprint.json`
- raw request or iteration metrics
- stdout/stderr
- model and tokenizer revision/hash
- experiment protocol ID

Conditionally required:

- reference outputs/logits for same-precision correctness
- quantization quality report
- profiler summary
- matched baseline run
- custom-kernel shape report
- dataset sample IDs/hashes

---

## Outputs

- `validation_verdict.json`
- human-readable `VALIDATION.md`
- zero or more issue records
- policy eligibility flag

Verdict enum:

```text
PASS
CONDITIONAL_PASS
FAIL
INVALID
SKIPPED
```

Evidence level:

```text
E0 IDEA
E1 OBSERVED
E2 REPEATED
E3 INTERVENTION_SUPPORTED
E4 HOLDOUT_CONFIRMED
E5 CROSS_ENVIRONMENT_CONFIRMED
```

---

## Non-negotiable Rules

1. Never rank a run that fails correctness.
2. Never use profiler-enabled latency as production latency.
3. Never mix engine-only and end-to-end timing.
4. Never silently discard slow repetitions.
5. Never infer a missing metric.
6. Never accept an unpinned model or runtime for a headline claim.
7. Never treat an OOM as a speed result; record it as a capacity boundary.
8. Never allow demo/mock data into measured result tables.
9. Never accept an Agent-written number unless it is exactly backed by a run artifact.
10. Never loosen numerical tolerances after seeing a failed kernel without a documented methodological reason and fresh preregistered rerun.

---

# Workflow

## Step 0 — Resolve Scope

Read the manifest and classify the run:

- `performance_unprofiled`
- `performance_profiled`
- `quality`
- `kernel_correctness`
- `kernel_microbenchmark`
- `causal_intervention`
- `startup_amortization`
- `policy_holdout`

Load the corresponding required checks. Do not apply irrelevant checks, but record each skipped check and reason.

---

## Step 1 — Schema and Artifact Integrity

Check:

- all required files exist;
- JSON parses;
- instances match schemas;
- run ID is consistent across files;
- plan/workload hashes match manifest;
- timestamps are monotonic and plausible;
- checksums match;
- raw metrics are non-empty;
- summary can be regenerated from raw metrics;
- no absolute secret/token values appear in artifacts.

Fail modes:

- missing raw data → `INVALID`
- corrupt schema → `INVALID`
- summary mismatch > numerical rounding tolerance → `INVALID`

---

## Step 2 — Environment Match

Check:

- expected GPU name and UUID;
- driver, CUDA, PyTorch, Triton, runtime version;
- model revision and file hash;
- tokenizer revision and chat-template hash;
- WSL/native mode;
- git commit and dirty state;
- power limit and relevant environment variables;
- backend build flags;
- no unapproved background GPU process.

Rules:

- dirty git is allowed for exploratory runs only and must produce `CONDITIONAL_PASS` at best;
- version mismatch inside a matched experiment invalidates causal comparison unless version is the intended intervention;
- stale policy fingerprint blocks policy eligibility.

---

## Step 3 — Workload Integrity

Check:

- actual tokenized prompt length equals requested controlled bucket;
- actual output token count is present;
- fixed-output protocol did not terminate early, or early termination is explicitly allowed;
- prompt IDs and seeds are identical for paired comparisons;
- sampling settings are identical except when sampling itself is the tested factor;
- tokenizer and chat template parity across runtime comparisons;
- arrival trace replay has no missing requests.

Cross-runtime fairness failure → `INVALID` for the comparison, though individual runs may remain valid.

---

## Step 4 — Functional Correctness

### Same-precision runtime

Required checks:

- no NaN/Inf;
- reference model successfully ran;
- greedy next-token agreement;
- logits comparison on preregistered prompts;
- EOS/BOS semantics;
- KV-cache enabled state;
- repeated deterministic output.

Default thresholds are configured, not hard-coded. Recommended initial defaults:

```yaml
fp32:
  atol: 1.0e-4
  rtol: 1.0e-4
fp16:
  atol: 2.0e-2
  rtol: 2.0e-2
bf16:
  atol: 5.0e-2
  rtol: 5.0e-2
```

For end-to-end logits, also report:

- median and max KL;
- top-1 agreement;
- top-k overlap;
- first divergent token.

Do not use a single scalar tolerance to hide systematic divergence.

### Quantized runtime

Do not require token identity. Proceed to quality gate after basic no-crash/no-NaN checks.

### Custom kernel

Require:

- all registered shapes;
- all supported dtypes;
- awkward/non-power-of-two shapes;
- random seeds;
- extreme values;
- non-contiguous policy;
- fallback test;
- repeated determinism.

Any supported shape correctness failure → `FAIL` and kernel dispatch disabled for that region. If the contract explicitly rejects the shape and fallback works, it may pass.

---

## Step 5 — Timing Integrity

Verify:

- warm-up events excluded from steady-state;
- compile/autotune/capture cost recorded separately;
- CUDA synchronization or events placed at documented boundaries;
- no profiler active for production timings;
- wall clock is monotonic;
- no negative or zero impossible durations;
- first token timestamp is after request acceptance;
- last token timestamp is after first token;
- TPOT omitted for one-token output;
- tokenization inclusion is explicitly labeled;
- retry time is not silently removed.

Run timer calibration fixture if environment or timing code changed.

Any unknown timing boundary → `INVALID`.

---

## Step 6 — Stability and Thermal Conditions

Compute:

- median, MAD, robust CV;
- first/middle/last-third median;
- temperature range;
- SM/memory clock range;
- power range;
- background utilization;
- timeout/retry/fallback rate.

Initial criteria:

```text
first-vs-last median drift ≤ 3%
matched-pair median temperature difference ≤ 5°C
background GPU utilization before run < 5%
no unexplained clock collapse
```

If variation is high:

1. inspect raw series;
2. identify environmental event;
3. increase repetitions only if no systematic drift;
4. otherwise rerun after fixing environment.

Do not average over thermal drift.

---

## Step 7 — Statistical Sufficiency

### Performance

- microbenchmark: at least 100 measured iterations per shape;
- single-request engine: at least 30 repetitions;
- online tail latency: at least 200 requests; use 1,000 for stable p99 claims;
- paired comparison: identical prompt/request pairs;
- bootstrap: at least 10,000 resamples.

Compute:

- median and 95% CI;
- paired median difference;
- ratio of medians;
- paired log-ratio geometric mean;
- practical effect threshold.

Recommended claim threshold:

```text
95% CI supports direction AND practical improvement ≥ 2%
```

A smaller effect can be `PASS` as a valid measurement but is labeled `neutral` for performance claims.

### Search experiments

Ensure search set and confirmatory holdout are separate. A winner selected and evaluated on the same noisy rows is exploratory only.

---

## Step 8 — Quality Gate

Load requested profile:

```text
strict
balanced
memory-first
custom
```

Initial defaults:

```yaml
strict:
  arc_c_drop_pp_max: 0.5
  ppl_ratio_max: 1.02
balanced:
  arc_c_drop_pp_max: 1.0
  ppl_ratio_max: 1.05
memory_first:
  arc_c_drop_pp_max: 2.0
  ppl_ratio_max: 1.10
```

Check protocol parity before comparing quality.

If a plan is fast but fails quality:

- verdict may remain a valid measurement;
- `quality_pass=false`;
- `policy_eligible=false` for the selected profile;
- UI may show it only on the full Pareto frontier.

---

## Step 9 — Causal Intervention Validation

For `causal_intervention`, require:

1. hypothesis preregistered before confirmatory run;
2. intervention targets a specified mediator;
3. matched baseline and intervention differ only in intended factor or documented necessary changes;
4. mediator changed in expected direction;
5. production outcome confirmed unprofiled;
6. correctness/quality pass;
7. scope specified;
8. negative control or alternative explanation discussed.

Verdict:

- mediator + outcome + holdout → evidence E4;
- mediator + outcome, no holdout → E3;
- outcome only → correlation, E2 maximum;
- mediator unchanged → hypothesis rejected or inconclusive.

---

## Step 10 — Provenance and Public-Claim Audit

Check:

- exact command available;
- environment lock available;
- model/dataset terms documented;
- no gated raw data redistributed;
- raw artifact link/checksum available;
- result label is `MEASURED`, not demo/estimated;
- claim wording contains hardware, model, workload, metric, and scope;
- `up to` claims also include distribution-level statistic;
- limitations updated.

A valid private run without distributable evidence can be `PASS` internally but `public_claim_eligible=false`.

---

## Step 11 — Produce Verdict

Example:

```json
{
  "run_id": "run-20260803-001",
  "verdict": "PASS",
  "policy_eligible": true,
  "public_claim_eligible": true,
  "quality_pass": true,
  "evidence_level": "E4",
  "checks": [],
  "issues": [],
  "scope": {
    "gpu": "RTX 4080 SUPER",
    "model": "ministral3-3b",
    "prompt_tokens": [128, 1024],
    "concurrency": [1]
  }
}
```

---

# Mandatory Human Review Triggers

Require human review when:

- numerical tolerance changed;
- model or runtime uses remote code;
- raw prompt may contain personal data;
- custom kernel writes in-place;
- performance claim exceeds 25% unexpectedly;
- profiler and production results disagree strongly;
- cross-runtime tokens differ;
- policy switches backend based on a sparse region;
- Copilot proposes a new shell command;
- release includes gated artifacts.

---

# Validation Report Template

```markdown
# Validation: <run_id>

## Verdict
PASS / CONDITIONAL_PASS / FAIL / INVALID

## Scope
Hardware, model, workload, plan.

## Gate Summary
| Gate | Status | Evidence |

## Correctness
Thresholds, mismatch details.

## Timing Integrity
Boundary, warmup, profiler status.

## Stability
Distribution, drift, temperature.

## Statistics
Effect, CI, practical significance.

## Quality
Datasets, protocol, result.

## Provenance
Hashes, versions, commands.

## Eligibility
Policy / public claim.

## Issues and Required Actions
```

---

# Refusal Conditions

This skill must refuse to certify when:

- only a screenshot is provided without raw records;
- the requested headline combines incompatible protocols;
- the result came from a mock UI;
- the user asks to ignore a correctness or quality failure;
- timing boundaries cannot be reconstructed;
- the model revision is unknown;
- the run was manually edited without a transformation record.
