# Domain Documentation

EdgeFlow uses a single bounded context.

- Canonical glossary: `CONTEXT.md`
- Architectural decisions: `docs/adr/`
- System and experiment specifications: `docs/`

## Consumption rules

1. Read `CONTEXT.md` before changing domain-facing names, schemas, API labels, issue language, validation statuses, evidence states, or UI copy.
2. Read relevant ADRs before revisiting a hard-to-reverse decision.
3. When a term is ambiguous, resolve it before implementation and update `CONTEXT.md` in the same change.
4. `CONTEXT.md` contains definitions only. Implementation, procedures, and rationale belong in specifications or ADRs.
5. Create an ADR only for a hard-to-reverse, surprising decision produced by a real trade-off.
