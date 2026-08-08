# ADR-0004: Make executable validation the certification authority

- Status: Accepted
- Date: 2026-08-09

## Context

A detailed validation skill can describe rules that the current Python validator does not yet enforce. Prose-only certification creates a dangerous gap: an agent may believe a gate was applied when the executable system merely trusted an artifact flag.

## Decision

Every hard validation rule must be represented in `specs/validation_requirements.yaml` and have:

1. a canonical requirement ID;
2. an executable enforcement path or an explicit planned status;
3. regression tests for implemented rules;
4. a defined artifact and acceptance criterion;
5. a release gate describing what the rule blocks.

The validator, not an agent narrative, decides eligibility.

## Consequences

- Validation specification and implementation parity becomes measurable.
- Planned rules cannot silently appear implemented.
- Release checks can fail when required hard rules remain partial.
- Skills remain orchestration and reference documents, not certification engines.
