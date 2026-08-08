# Skills Integration Strategy

EdgeFlow selectively adapts the small, composable engineering practices published in `mattpocock/skills`; it does not delegate ownership of the project process to a monolithic framework.

## Adapted practices

| Upstream practice | EdgeFlow adaptation |
|---|---|
| Grilling before implementation | `edgeflow-change-design` resolves scope, seams, terminology, alternatives, and acceptance criteria |
| Domain modeling | `CONTEXT.md` and focused ADRs create a shared language |
| Tracer-bullet ticketing | `edgeflow-ticketing` creates vertical issues with blocking edges |
| TDD | Engineering behavior is added through public seams using red → green slices |
| Tight diagnosis loop | `edgeflow-performance-debugging` requires a deterministic reproducer before hypotheses |
| Primary-source research | Version-sensitive runtime facts are captured under `docs/research/` |
| Architecture deepening | Periodic hotspot reviews examine locality, leverage, and test seams |
| Two-axis review | `edgeflow-code-review` separates code standards from specification fidelity |
| Progressive disclosure | Root pointers stay concise; branch-specific rules live in focused documents and skills |

## What is deliberately not copied

- No upstream skill is treated as an unmodified universal process.
- No agent is allowed to bypass EdgeFlow G0–G8 validation.
- No generic TDD result is treated as scientific evidence.
- No autonomous shell or environment mutation is introduced.
- No upstream text is copied as a substitute for EdgeFlow-specific requirements.

## Attribution

The upstream repository is MIT licensed. EdgeFlow's documents and skills are project-specific adaptations informed by those practices. Review the upstream license before vendoring any substantial upstream text or code.

## Skill invocation map

```text
Ambiguous change
  → edgeflow-change-design
  → CONTEXT / ADR update if needed
  → edgeflow-ticketing

Engineering implementation
  → public seam agreed
  → red test
  → minimal implementation
  → edgeflow-code-review

Performance regression
  → edgeflow-performance-debugging
  → tight loop
  → minimized reproducer
  → matched intervention

Formal experiment
  → experiment-runner
  → edgeflow-validation
  → profiler-diagnosis when diagnostic evidence is required
```
