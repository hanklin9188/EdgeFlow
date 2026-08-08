# ADR-0006: Defer learned components until the run database is sufficient

- Status: Accepted
- Date: 2026-08-09

## Context

A cost model or bottleneck classifier trained on a small, correlated run database can appear sophisticated while increasing recommendation error and hiding uncertainty.

## Decision

Keep screening, diagnosis, and policy construction deterministic until the database satisfies the preregistered sample, split, calibration, and regret criteria. Repetitions of one configuration do not count as independent training points.

## Consequences

- The MVP does not train a foundation model or add ML merely for product positioning.
- Learned components are optional accelerators, never final authorities.
- Final candidates still require measurement and full validation.
