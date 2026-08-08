# 20 · Skill Operating Model

## 1. Goal

Use small, composable skills to improve agent consistency without allowing a generic process to override EdgeFlow's domain and validation authority.

The skill set is divided by responsibility.

## 2. User-invoked orchestration

| Skill | Use |
|---|---|
| `edgeflow-change-design` | Resolve an ambiguous change and record domain/ADR consequences |
| `edgeflow-ticketing` | Convert an approved design into vertical issues with blocking edges |
| Existing `experiment-runner` | Execute a preregistered formal experiment block |
| Existing `performance-copilot` | Query validated evidence after E28 grounding passes |

Orchestration skills coordinate. They do not certify results.

## 3. Model-invoked disciplines

| Skill | Use |
|---|---|
| `edgeflow-performance-debugging` | Build a tight reproducer for bugs or performance regressions |
| `edgeflow-code-review` | Review Standards and Specification as independent axes |
| Existing `edgeflow-validation` | Decide run eligibility |
| Existing `profiler-diagnosis` | Produce bounded hypotheses from traces |

## 4. Progressive disclosure

Keep root pointers short and load detailed material only when the branch requires it.

Examples:

- A UI wording change reads `CONTEXT.md`, not every experiment protocol.
- A causal run reads causal-validation requirements, not kernel rules.
- A kernel correctness change reads kernel contracts, not Copilot grounding rules.
- A release claim reads public-claim audit and provenance requirements.

The root `AGENTS.md` names triggers and authorities; the detailed process stays in focused documents and skills.

## 5. Skill quality criteria

Every EdgeFlow skill must have:

- a precise trigger in frontmatter;
- named inputs and outputs;
- ordered steps;
- completion criteria for each major phase;
- refusal or escalation conditions;
- links to authoritative specs instead of duplicated rule text;
- an explicit statement of what the skill cannot certify.

## 6. Change-design flow

```text
Problem statement
→ user-visible behavior
→ canonical terms
→ public seam
→ data ownership
→ failure modes
→ alternatives
→ validation impact
→ ADR decision
→ approved spec
```

Do not begin implementation while the seam, acceptance behavior, or evidence impact remains ambiguous.

## 7. Ticketing flow

A ticket is ready when:

- it fits one fresh implementation context;
- it cuts through all required layers;
- it is demonstrable by one behavior;
- blockers are genuine;
- acceptance criteria are executable;
- the final command is named;
- formal experiments name artifacts and gates.

Wide mechanical changes use expand→migrate→contract instead of forced vertical slices.

## 8. TDD flow

TDD applies to deterministic software behavior:

```text
Agree public seam
→ write one failing test
→ confirm intended failure
→ implement minimum behavior
→ rerun focused test
→ repeat next slice
```

Avoid tests coupled to private implementation, tautological expected values, and bulk horizontal test creation.

## 9. Performance-debugging flow

```text
Tight loop
→ reproduce
→ minimize
→ rank hypotheses
→ instrument predictions
→ fix or intervene
→ verify original symptom
→ preserve regression fixture
```

A tight loop is:

- red-capable for the exact symptom;
- deterministic or high-reproduction;
- fast enough for iteration;
- runnable without manual interpretation.

No tight loop means no confident hypothesis work.

## 10. Two-axis review

### Standards axis

Checks:

- repository and security standards;
- canonical terms;
- module depth and locality;
- public test seam;
- duplication and speculative generality;
- validator parity;
- no prohibited data or shell path.

### Specification axis

Checks:

- missing or partial acceptance criteria;
- behavior not requested;
- wrong experiment protocol;
- incorrect evidence promotion;
- missing negative branches;
- unsupported claim.

Do not let a pass on one axis hide a failure on the other.

## 11. Primary-source research

Use `docs/research/` for version-sensitive questions such as:

- torch.compile graph/capture behavior;
- vLLM scheduling semantics;
- llama.cpp timing boundaries or quantization;
- Nsight metric interpretation;
- Triton backend support.

Research establishes what the upstream system claims. EdgeFlow experiments establish what happens on the target hardware.

## 12. Architecture review

Review recent hot spots, not the whole repository indiscriminately. Seek deepening opportunities:

- one small interface hiding significant behavior;
- evidence and policy rules localized behind one seam;
- runtime lifecycle represented once;
- validation checks independently testable;
- fewer modules needed to understand one domain action.

Architecture work must not change a frozen formal experiment mid-block.
