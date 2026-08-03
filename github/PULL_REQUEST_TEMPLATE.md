## Scope

Describe one bounded change. Link the experiment, issue, or design decision.

## Change type

- [ ] Runtime adapter
- [ ] Benchmark protocol
- [ ] Validation or statistics
- [ ] Profiler / diagnosis
- [ ] Triton kernel
- [ ] Policy synthesis
- [ ] Dashboard / documentation
- [ ] Build / CI

## Evidence

- Run IDs:
- Hardware fingerprint:
- Validation verdicts:
- Raw artifacts:
- Before/after protocol parity:

## Correctness and safety

- [ ] Unit/contract tests pass.
- [ ] Same-precision or quantization quality gate passes where applicable.
- [ ] No secret, private prompt, gated data, or model weight is committed.
- [ ] Demo/estimated numbers are visibly labeled.
- [ ] Fallback path was exercised.

## Performance claim checklist

- [ ] Production timing was collected without profiler.
- [ ] Warm-up and cold-start costs are reported separately.
- [ ] Run order and repetitions follow the protocol.
- [ ] Confidence interval and sample count are included.
- [ ] Scope and slower/failure regions are documented.

## Reviewer decision

`APPROVE / REQUEST_CHANGES / INCONCLUSIVE_EVIDENCE`
