## Experiment identity

- Experiment / protocol:
- Originating issue:
- Search, diagnostic, or confirmatory role:

## Preregistered hypothesis

## Scope

- Hardware fingerprint:
- Model/tokenizer revision:
- Runtime revision:
- Workload and timing boundary:

## Changed and controlled variables

## Raw evidence

- Run IDs:
- Artifact paths/checksums:
- Validation verdicts:
- Quality/correctness:

## Results

- Primary outcome and uncertainty:
- Mediator result when causal:
- Negative control / alternative explanation:
- Capacity or failure rows:

## Deviations

State every deviation from the preregistration. Write `None` only after checking.

## Claim audit

**Permitted claim:**

**Claims still prohibited:**

**Evidence level:**

## Verification

```bash
python scripts/verify_results.py
python scripts/audit_validation_parity.py
pytest -q
```
