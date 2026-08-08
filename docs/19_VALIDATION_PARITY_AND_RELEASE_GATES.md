# 19 · Validation Specification Parity and Release Gates

## 1. Problem

EdgeFlow's validation skill describes a broader standard than the current executable validator enforces. This is expected during development, but it must be visible and managed.

A hard rule is complete only when four layers agree:

```text
Specification
→ Executable enforcement
→ Regression test
→ Run artifact evidence
```

Prose alone does not certify a run. An artifact field saying `pass=true` is not sufficient when the validator can independently recompute the result.

---

## 2. Machine-readable authority

`specs/validation_requirements.yaml` lists every hard or advisory requirement.

Each entry contains:

- stable requirement ID;
- validation gate;
- rule and risk;
- implementation status;
- enforcement path;
- test path;
- acceptance criterion;
- artifact/evidence source;
- release gates blocked by the requirement;
- backlog owner.

Run:

```bash
python scripts/audit_validation_parity.py
```

The default audit checks structure, unique IDs, valid states, and path existence for implemented rules.

Before an evidence release, run:

```bash
python scripts/audit_validation_parity.py --strict-release
```

Strict release mode fails when any hard requirement marked as an evidence-release gate is not `implemented`.

---

## 3. Status definitions

| Status | Meaning |
|---|---|
| `implemented` | Validator independently enforces the rule and regression tests cover pass/fail behavior |
| `partial` | Some behavior exists, but enforcement depends on attestation, lacks a negative fixture, or omits part of the rule |
| `planned` | Requirement is specified and assigned to a backlog ticket but not enforced |
| `deferred` | Deliberately postponed with an explicit trigger |

Do not use `implemented` because a field is present. Use it only when the validator can reject the invalid case through executable logic.

---

## 4. Hard-rule closure workflow

For each `partial` or `planned` hard rule:

### Step 1 — Define the public certification seam

Examples:

- `ValidationEngine.validate(run_dir)`;
- `compare_matched_runs(a, b)`;
- `promote_evidence(record)`;
- `build_policy(eligible_rows)`.

The seam must expose the behavior that can reject the invalid case.

### Step 2 — Create a red fixture

Construct the smallest artifact directory that violates exactly the target rule.

Examples:

- one cross-runtime prompt ID differs;
- quantized plan lacks quality rows;
- search and holdout IDs overlap;
- mediator does not move;
- claim omits workload scope.

The test must fail for the intended rule, not for an earlier unrelated schema failure.

### Step 3 — Implement independent enforcement

Prefer recomputation from primary artifacts:

- token parity from stored IDs;
- summary from raw rows;
- quality from task-level rows;
- overlap from sample IDs/hashes;
- mediator from profiler summaries;
- evidence promotion from linked verdicts.

Attestation-only fields can be inputs but cannot be sole authority for hard release gates.

### Step 4 — Add the green and negative cases

At minimum:

- valid fixture passes;
- invalid fixture is rejected;
- missing fixture cannot become eligible;
- existing unrelated valid runs retain their verdict.

### Step 5 — Update requirement status

Change `status` to `implemented` only in the same PR that adds enforcement and tests.

### Step 6 — Two-axis review

- Standards: correctness, security, test seam, module depth, no duplicated rule logic.
- Specification: exact requirement behavior, missing branches, prohibited eligibility.

---

## 5. Priority closure set

These gaps block the first public performance claim.

### VR-005 Workload token parity

**Required behavior:** Cross-runtime comparison is invalid unless actual prompt token IDs, template hash, and output protocol match.

**Acceptance:** A single mismatched prompt in a paired group makes the comparison ineligible while preserving individually valid runs.

### VR-006 Same-precision correctness

**Required behavior:** The validator computes no-NaN, top-1 agreement, KL summary, top-k overlap, and first divergent token from reference/candidate artifacts.

**Acceptance:** Systematic divergence cannot be hidden by one permissive scalar tolerance.

### VR-010 Quality recomputation

**Required behavior:** Quantized quality eligibility is derived from task-level records and the selected profile, not only `quality.pass`.

**Acceptance:** Missing or protocol-mismatched quality artifacts block policy eligibility.

### VR-014 Search/holdout separation

**Required behavior:** Sample IDs/hashes and workload buckets used for selection cannot overlap confirmatory holdout.

**Acceptance:** Leakage creates an invalid confirmatory claim even when performance is strong.

### VR-016 Causal promotion

**Required behavior:** Evidence cannot exceed repeated observation unless mediator and unprofiled outcome pass a matched intervention.

**Acceptance:** Outcome-only improvements remain correlation; unchanged mediator rejects or makes the hypothesis inconclusive.

### VR-018 Public-claim scope

**Required behavior:** Public eligibility requires hardware, model revision, workload, timing boundary, uncertainty, quality status, and supporting run IDs.

**Acceptance:** Missing scope blocks public claim without necessarily invalidating the internal measurement.

---

## 6. Gate-specific standards

### G0 — Schema and artifact integrity

Pass only when:

- required artifacts parse and match schemas;
- run, plan, workload, and hashes agree;
- raw rows are non-empty;
- summary is recomputable;
- immutable checksums match;
- no secret-bearing values appear.

### G1 — Environment

Pass only when:

- expected GPU and execution mode match protocol;
- runtime/model revisions are pinned;
- no unapproved GPU process exists;
- intended runtime is distinguished from unrelated background compute;
- power/clock policy is recorded.

### G2 — Correctness

Pass only when applicable correctness is independently checked. Missing correctness may allow a diagnostic measurement but never policy eligibility.

### G3 — Timing

Pass only when:

- warmup and production rows are distinct;
- compile/capture/autotune are separated;
- timing boundary is known;
- profiler is off for ranked timing;
- timestamps are monotonic and plausible;
- retries are not silently removed.

### G4 — Stability

Pass only when empirically selected E02/E03 thresholds are satisfied. Do not average through drift.

### G5 — Statistics

Pass only when:

- repetition minimum matches run type;
- paired designs preserve pairing;
- uncertainty is computed;
- search and confirmation are separated;
- practical effect is distinguished from statistical direction.

### G6 — Quality

Pass only when protocol parity and selected profile thresholds hold. A quality failure may remain on an exploratory Pareto frontier but cannot enter the failed profile's policy.

### G7 — Provenance

Pass only when exact command, revisions, git state, dataset handling, and transformation provenance are recoverable.

### G8 — Eligibility

Pass only when all gates required for the run's intended use pass. Eligibility is purpose-specific:

- measurement-valid;
- policy-eligible;
- evidence-promotion-eligible;
- public-claim-eligible.

Avoid collapsing those into one ambiguous Boolean over time; migrate to explicit eligibility facets.

---

## 7. Evidence promotion standard

| Level | Required evidence |
|---|---|
| E0 | design idea only |
| E1 | one valid observation |
| E2 | repeated stable observation |
| E3 | matched intervention changed mediator and unprofiled outcome |
| E4 | frozen result confirmed on holdout |
| E5 | result confirmed across a second environment or meaningful hardware/software boundary |

Promotion is monotonic and artifact-backed. An agent cannot promote evidence by wording.

---

## 8. Release gates

### Engineering alpha

Must pass:

- package CI;
- governance audit;
- security controls;
- smoke lifecycle;
- no demo leakage;
- implementation status accurately separates engineering and evidence.

### First evidence release

Additionally requires:

- E01–E03 complete;
- priority validation closure set implemented;
- primary E04–E09 complete;
- quality and fairness pass;
- no unsupported cross-runtime headline.

### Policy claim release

Additionally requires:

- E10–E12;
- strongest simple baseline;
- holdout replay;
- scoped fallback and stale behavior;
- claim audit.

### Causal diagnosis release

Additionally requires:

- at least one E3/E4 supported hypothesis;
- at least one rejected/inconclusive negative-control case;
- mediator and outcome artifacts;
- diagnostic accuracy limitations.

### Kernel performance release

Additionally requires:

- profiler-backed target selection;
- full correctness region;
- complete heatmap;
- end-to-end matched result;
- fallback reliability.

### v1.0 product release

Additionally requires:

- E29 usability;
- E30 clean reproduction;
- stable release channel and rollback;
- public documentation and sanitized evidence bundle.
