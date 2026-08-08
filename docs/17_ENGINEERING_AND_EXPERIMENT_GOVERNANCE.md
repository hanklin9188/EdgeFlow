# 17 · Engineering and Experiment Governance

## 1. Purpose

EdgeFlow has reached a stage where adding more features is less important than converting existing architecture into reliable evidence. This governance model integrates small, composable engineering practices with EdgeFlow's scientific validation system.

The target operating loop is:

```text
Shared language
→ design decision
→ tracer-bullet issue
→ test-first engineering
→ formal experiment
→ executable validation
→ evidence record
→ scoped claim
→ deployment policy
```

The central distinction is:

> **Engineering completion proves that a capability behaves as specified. Experimental completion proves that a systems claim is supported under a defined scope.**

Neither substitutes for the other.

---

## 2. Current program diagnosis

EdgeFlow already has substantial platform engineering:

- hardware and software fingerprinting;
- runtime adapters;
- raw request records and SQLite indexing;
- validation gates;
- profiling and deterministic diagnosis;
- session-aware objectives and policy construction;
- correctness-gated Triton fallback;
- a local-first control console.

The remaining work is primarily **evidence debt**:

1. measurement calibration is not complete;
2. primary-model cross-runtime comparisons are not complete;
3. quality and fairness artifacts are incomplete;
4. policy claims lack eligible holdout rows;
5. causal hypotheses lack matched intervention evidence;
6. the Triton path lacks a profiler-selected and end-to-end-confirmed claim;
7. the prose validation specification is broader than current executable enforcement.

This document orders work to retire those debts without losing engineering quality.

---

## 3. Sources of authority

Use one source of truth for each concern.

| Concern | Authority |
|---|---|
| Canonical terminology | `CONTEXT.md` |
| Hard architectural rationale | `docs/adr/` |
| Experiment hypothesis and protocol | `docs/03_EXPERIMENT_CATALOG.md` and formal experiment issue |
| Work order and blockers | `docs/18_EXECUTION_BACKLOG.md` and GitHub Issues |
| Validation requirements | `specs/validation_requirements.yaml` |
| Run truth | immutable raw artifacts |
| Eligibility | executable validation verdict |
| Claim support | evidence records and holdout results |
| Agent behavior | `AGENTS.md` and focused skills |

Do not duplicate full rules across documents. Link to the authority and keep branch-specific detail behind concise pointers.

---

## 4. Completion states

Every work item declares one of five completion states.

| State | Meaning | Minimum proof |
|---|---|---|
| `S0 PLANNED` | Scope exists but implementation has not started | issue/spec with blockers and acceptance criteria |
| `S1 ENGINEERED` | Capability behaves through its public seam | red→green tests, clean review, CI |
| `S2 MEASURED` | Formal raw records exist and pass applicable measurement gates | immutable artifacts and validation verdict |
| `S3 CONFIRMED` | Search-independent confirmatory or matched intervention evidence exists | holdout or mediator+outcome evidence |
| `S4 RELEASED` | Claim and product surface are publishable | claim audit, licensing/privacy, reproducibility bundle |

A feature can be `S1` while its research claim remains `S0`. This is the expected state for much of the current repository.

---

## 5. Work-stream separation

### 5.1 Engineering stream

Use when implementing deterministic software behavior.

#### Entry requirements

- canonical terms are clear;
- public seam is named;
- issue has acceptance criteria and blockers;
- relevant ADRs are read;
- change is one vertical slice.

#### Execution

1. **Design**
   - Run the `edgeflow-change-design` workflow.
   - Resolve user behavior, module seam, failure states, data ownership, and validation impact.
   - Update `CONTEXT.md` if language changed.
   - Add an ADR only for a load-bearing decision.
2. **Ticket**
   - Create one tracer-bullet issue.
   - Specify what a user or caller can observe when complete.
   - Name the test seam and validation command.
3. **Red**
   - Add a failing test through the public seam.
   - Confirm it fails for the intended reason.
4. **Green**
   - Implement the smallest behavior that passes.
   - Avoid speculative abstractions and adjacent features.
5. **Integration**
   - Run focused test, full test suite, lint, schema checks, package validation, and governance audit.
6. **Review**
   - Standards axis: code quality, security, locality, module depth, repository conventions.
   - Specification axis: missing requirements, scope creep, incorrect behavior.
7. **Close**
   - Merge only when acceptance criteria and review findings are resolved.

#### Engineering definition of done

```text
Public seam works
+ regression test exists
+ failure behavior is explicit
+ no undocumented domain term
+ no unresolved spec finding
+ CI green
```

### 5.2 Scientific stream

Use when producing measurements, comparisons, profiler explanations, policy evidence, or public claims.

#### Entry requirements

- experiment issue exists;
- hypothesis is falsifiable;
- model/runtime revisions are pinned;
- changed variable and controlled variables are explicit;
- outcome, mediator, quality gate, and negative control are specified;
- measurement calibration prerequisites are satisfied;
- search and holdout roles are defined.

#### Execution

1. **Preregister**
   - Freeze experiment ID, protocol version, model, workload, plan matrix, repetition policy, metrics, failure handling, and acceptance criteria.
2. **Capability screen**
   - Prune only unsupported, duplicate, known-invalid, or estimated-OOM candidates.
   - Record every prune reason.
3. **Isolate**
   - Run one formal configuration in an isolated process.
   - Verify GPU idle, thermal precondition, model hash, ports, disk, and no unrelated runtime.
4. **Warm up**
   - Record warmup separately.
   - Do not delete cold cost; store it under the correct phase.
5. **Measure**
   - Flush raw rows incrementally.
   - Preserve timeouts, retries, OOMs, and fallback events.
   - Keep profiler off for production timing.
6. **Validate**
   - Apply applicable G0–G8 checks.
   - A failed gate is a result, not an invitation to change thresholds after inspection.
7. **Diagnose**
   - Convert literal profiler observations into ranked hypotheses.
   - Do not claim causality.
8. **Intervene**
   - Change one intended factor.
   - Use the same prompts, time block, environment, and paired order.
9. **Verify mediator and outcome**
   - Expected mediator must change.
   - Unprofiled outcome must improve beyond the practical threshold.
   - Correctness and quality must still pass.
10. **Confirm**
    - Re-run frozen settings on holdout prompts or buckets.
11. **Audit the claim**
    - Include scope, uncertainty, run IDs, protocol, quality, limitations, and stale triggers.

#### Scientific definition of done

```text
Preregistered protocol
+ raw artifacts
+ applicable gates PASS
+ search/holdout separation
+ scoped evidence record
+ reproducible analysis
+ claim wording matches evidence level
```

---

## 6. Skill operating sequence

### Before an ambiguous change

```text
edgeflow-change-design
→ resolve glossary terms
→ record ADR when justified
→ create spec or issue
```

### Before implementation

```text
edgeflow-ticketing
→ one vertical slice
→ blockers explicit
→ public seam explicit
→ acceptance criteria executable
```

### During engineering

```text
TDD discipline
→ red through public seam
→ minimal green
→ no horizontal bulk implementation
```

### When a bug or regression appears

```text
edgeflow-performance-debugging
→ tight red-capable loop
→ minimize
→ rank 3–5 falsifiable hypotheses
→ instrument discriminating signals
→ fix/intervene
→ regression fixture
```

### Before merge

```text
edgeflow-code-review
→ Standards review
→ Specification review
→ both resolved independently
```

### During formal experiments

```text
experiment-runner
→ edgeflow-validation
→ profiler-diagnosis where needed
→ claim audit
```

---

## 7. Experiment-order policy

The following dependency order is mandatory.

### Gate A — Governance foundation

Deliver:

- `AGENTS.md`;
- `CONTEXT.md`;
- ADRs;
- machine-readable validation matrix;
- issue templates;
- governance CI.

Exit:

```bash
python scripts/audit_validation_parity.py
pytest -q tests/test_governance_contract.py
```

### Gate B — Trust the clock: E01–E03

Do not run a large performance matrix before:

- timer boundary is calibrated;
- thermal/background policy is empirically chosen;
- repetition count is justified by stability and winner-reversal analysis.

Exit artifacts:

- `timer_calibration.json`;
- `measurement_policy.yaml`;
- `repetition_policy.json`;
- corresponding validation reports.

### Gate C — Executable validation parity

Close all hard requirements that block formal baseline claims:

- workload token parity;
- same-precision correctness;
- quantized quality enforcement;
- search/holdout separation;
- causal evidence promotion;
- public-claim scope audit.

A rule is not complete until specification, code, regression test, and artifact fixture agree.

### Gate D — Primary 3B baseline: E04

Use one primary 3B model and one pinned tokenizer/revision. Establish the PyTorch eager reference across the preregistered matrix.

Exit:

- correctness baseline;
- BF16 quality baseline;
- full raw measurements;
- validated `PASS` rows for supported buckets.

### Gate E — Compile behavior: E05–E06

Measure compile modes, first compile, recompile, graph breaks, dynamic shapes, and shape buckets. Produce both steady-state and session-horizon costs.

Exit:

- at least one frozen comparison showing whether the steady winner differs from short-session winner, or a documented null result;
- a validated shape-bucket policy or evidence that bucketing is unnecessary.

### Gate F — Runtime breadth: E07–E09

1. Run llama.cpp quantization/quality sweep on the same primary model family or a documented comparable checkpoint.
2. Run vLLM scheduling on the primary 3B model.
3. Complete token, template, sampling, output, and timing-boundary fairness audit.

No cross-runtime ranking before E09 passes.

### Gate G — First headline claim: E10–E12

Test the central EdgeFlow motivation:

> A workload-conditioned policy reduces expected regret relative to the strongest fixed plan.

Required baselines:

- global fixed winner;
- simple hand rule;
- decision tree or nearest-neighbor rule;
- EdgeFlow evidence-constrained decision list;
- oracle by bucket for regret calculation.

Required holdout:

- workload bucket holdout;
- real-distribution replay.

### Gate H — Causal diagnosis: E13–E18

Run at least one supported and one rejected/inconclusive hypothesis before claiming diagnosis capability.

Recommended sequence:

1. launch-overhead intervention;
2. memory-bandwidth intervention;
3. compute-bound prefill intervention;
4. KV-capacity boundary;
5. scheduler-bound mixed workload;
6. negative-control diagnosis.

### Gate I — Amortization: E19–E20

Separate process cold, model-cache warm, compile-cache warm, capture, and steady-state costs. Compute break-even over preregistered session horizons.

### Gate J — Hot-path optimization: E21–E24

1. Select kernel from formal profiler evidence.
2. Validate full kernel region.
3. Publish the complete performance heatmap, including slower regions.
4. Integrate into the model and measure unprofiled end-to-end effect.

A prototype kernel remains an engineering demonstration until this gate closes.

### Gate K — Optional learned components: E25–E27

Begin only after the run database reaches the preregistered number of independent plan-workload points and supports grouped, model-family, workload-range, and time-based splits.

The learned model may prune candidates; it never certifies the final winner.

### Gate L — Product and release validation: E28–E30

- E28: zero unsupported numeric claims in Copilot grounding evaluation.
- E29: at least five relevant users complete evidence-tracing tasks.
- E30: clean environment or second-machine smoke reproduces the documented path.

---

## 8. Acceptance thresholds

### Measurement

Initial policy, subject to E01–E03 calibration:

- engine performance: at least 30 measured requests;
- kernel microbenchmark: at least 100 iterations per region;
- online p95: at least 200 requests;
- p99 claim: target at least 1,000 requests;
- paired bootstrap: 10,000 resamples;
- practical performance threshold: at least 2% unless a product-specific threshold is preregistered;
- first-to-last-third drift: at most 3%;
- robust CV: at most 10%;
- matched temperature difference: at most 5°C;
- unexpected background GPU compute: none.

### Correctness

- no NaN or Inf;
- deterministic greedy behavior where required;
- exact prompt-token and template parity for cross-runtime comparisons;
- same-precision logits and token agreement under configured tolerances;
- custom kernel correctness over every advertised region;
- reference fallback for every unvalidated region.

### Quality

Quantized plans require quality artifacts. A plan may remain a valid measurement while being policy-ineligible for a chosen quality profile.

### Causality

Evidence reaches intervention support only when:

- hypothesis was preregistered;
- matched runs differ only in the intended factor or documented necessities;
- expected mediator changed;
- unprofiled production outcome changed in the expected direction;
- correctness and quality passed;
- scope and alternative explanations are recorded.

### Policy

- only measured `PASS` rows;
- no search/holdout leakage;
- fallback exists;
- sparse regions do not silently extrapolate;
- fingerprint changes produce `STALE`;
- policy is not worse than the strongest simple baseline on holdout within the predefined tolerance.

---

## 9. Pull-request protocol

Every PR must identify its lane.

### Engineering PR body

- originating issue/spec;
- public seam;
- user-visible behavior;
- red test and final test commands;
- architecture/ADR impact;
- validation-requirement impact;
- Standards review findings;
- Specification review findings.

### Formal experiment PR body

- experiment ID and protocol version;
- preregistered hypothesis;
- model/runtime revisions;
- workload and randomization;
- raw artifact locations and hashes;
- validation verdicts;
- deviations;
- search/holdout role;
- evidence level;
- exact claim allowed and claims still prohibited.

---

## 10. Change control

After a formal run begins, changing any of the following requires a new protocol version and run ID:

- model or tokenizer revision;
- prompt set or workload distribution;
- output stopping behavior;
- timing boundary;
- warmup or repetition policy;
- correctness or quality threshold;
- changed variable or control definition;
- profiler level;
- summary transformation;
- acceptance criterion.

Never retroactively relax a threshold after observing failure. Document the failure, justify a new protocol, and rerun.

---

## 11. Architecture review cadence

Run a focused architecture review after every two to three substantial PRs or whenever one concept requires touching many modules.

Prioritize recently changed hot spots. Evaluate:

- Is the public interface smaller than the behavior behind it?
- Can tests exercise the real behavior through one seam?
- Is knowledge localized or duplicated?
- Does one logical change require shotgun edits?
- Are validation, evidence, and policy concepts leaking across modules?
- Would deleting a proposed abstraction concentrate complexity or merely move it?

Do not perform speculative wide refactors during formal experiment execution. Finish the current evidence block, then refactor with preserved fixtures.

---

## 12. Release readiness

A release may present engineering capability before research claims, but must label them separately.

### Engineering release

Requires:

- CI and security checks;
- package and governance validation;
- smoke lifecycle;
- clean documentation;
- no fake result data;
- limitations and current implementation status.

### Evidence release

Additionally requires:

- primary formal runs;
- fairness and quality;
- holdout;
- public-claim audit;
- sanitized reproducibility bundle;
- no stale policy or unresolved evidence blocker.
