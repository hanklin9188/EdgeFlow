# ADR-0001: Keep the control plane local-first and loopback-only

- Status: Accepted
- Date: 2026-08-09

## Context

EdgeFlow starts and stops GPU runtimes, schedules local benchmarks, reads private artifacts, and may access licensed model files. Exposing this control surface over a public network would enlarge the attack surface and complicate authentication, privacy, and reproducibility.

## Decision

The production control plane binds only to loopback. Public GitHub Pages or exported reports are read-only and contain sanitised, validated artifacts only.

## Consequences

- Browser control remains a local workstation feature.
- Remote use requires an explicitly designed trusted tunnel or future authenticated deployment mode.
- Public presentation cannot start experiments or read local artifacts.
- Host, origin, request-size, token, path, process, and command boundaries remain release gates.

## Rejected alternatives

- Public multi-tenant control service: rejected because it is outside the current product scope and security budget.
- Browser-supplied commands or paths: rejected because typed, allowlisted actions provide a narrower seam.
