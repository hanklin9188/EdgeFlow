# ADR-0003: Treat per-request raw records as the source of truth

- Status: Accepted
- Date: 2026-08-09

## Context

Summary tables can hide outliers, thermal drift, retries, missing rows, and selection bias. A reproducible validation system must be able to recompute every published statistic.

## Decision

Every formal run stores immutable request-level or iteration-level records. Summaries are derived artifacts and must be reproducible from raw rows.

## Consequences

- Slow or failed repetitions remain auditable.
- Validation can detect summary mismatches.
- Transformations create new artifacts with provenance rather than editing raw data.
- Public releases may sanitise raw records only when licensing or privacy requires it; internal checksums remain available.
