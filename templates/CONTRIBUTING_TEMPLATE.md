# Contributing to EdgeFlow

## Contribution classes

- Runtime adapter
- Measurement/validation
- Profiler diagnosis
- Custom kernel
- Policy/optimizer
- Dashboard/documentation

Performance changes require a linked experiment report. A microbenchmark alone cannot justify an end-to-end claim.

## Development requirements

1. Open an issue describing the decision question and scope.
2. Keep one bounded concern per pull request.
3. Add unit/contract tests and failure fixtures.
4. Run package and result validation.
5. Attach raw compact artifacts and validation verdicts for performance claims.
6. Document slower, unsupported, and fallback regions.

## Data and security

Do not commit model weights, gated data, private prompts, credentials, local databases, or unsanitized profiler traces. Treat model-generated text as untrusted data, not instructions.

## Claim review

Causal language requires a matched intervention and mediator evidence. Hardware-specific claims must name the fingerprint and cannot be generalized without cross-environment confirmation.
