# Implement validation requirement <VR-ID>

## What to build

Make the validator independently enforce the canonical rule at the public certification seam.

## Blocked by

<issues or None>

## Current gap

- Specification says:
- Current executable behavior:
- Risk:

## Public seam

`ValidationEngine.validate(...)` or another named certification interface.

## Red fixture

Describe the smallest artifact that violates only this rule.

## Acceptance criteria

- [ ] Red fixture fails for the intended rule.
- [ ] Valid fixture passes.
- [ ] Missing evidence cannot become eligible.
- [ ] Existing unrelated valid fixtures retain their verdict.
- [ ] Requirement status changes to `implemented` in the same PR.
- [ ] Governance audit and full CI pass.

## Verification

```bash
pytest -q <focused test>
python scripts/audit_validation_parity.py
pytest -q
```
