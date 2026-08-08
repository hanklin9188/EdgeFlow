# ADR-0005: Build deployment policies from measured PASS runs only

- Status: Accepted
- Date: 2026-08-09

## Context

Policy synthesis can amplify a weak measurement into repeated deployment decisions. Conditional, diagnostic, stale, or quality-failing runs must not become recommendations.

## Decision

Only measured `PASS` runs with matching fingerprints and quality status may enter policy synthesis. Each policy rule records supporting run IDs, workload scope, uncertainty, and a validated fallback.

## Consequences

- Sparse workload regions fall back rather than extrapolate silently.
- Fingerprint drift marks policies `STALE`.
- Faster but quality-failing plans may appear on an exploratory Pareto plot but cannot be selected under the failed profile.
