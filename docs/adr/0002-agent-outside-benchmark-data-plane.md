# ADR-0002: Keep agents outside the benchmark data plane

- Status: Accepted
- Date: 2026-08-09

## Context

An LLM agent can consume GPU memory, CPU time, network bandwidth, file descriptors, and scheduler capacity. Allowing it to run inside the measured process would contaminate timing and make results difficult to reproduce.

## Decision

Agents may translate intent, retrieve validated evidence, draft experiments, and explain results. They may not execute inside the measured data plane or approve validation.

## Consequences

- Formal measurements remain deterministic and attributable.
- Copilot work pauses while the benchmark GPU is active unless it runs on a separate resource.
- Agent output is advisory until deterministic tools and validators confirm it.
- Run-specific numbers must come from artifacts and include run IDs.

## Rejected alternatives

- Fully autonomous performance agent in the measurement process: rejected because it makes the workload and resource state non-stationary.
