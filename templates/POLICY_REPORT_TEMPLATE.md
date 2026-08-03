# Deployment Policy Report · `POLICY_ID`

## Policy Contract

| Item | Value |
|---|---|
| Hardware fingerprint |  |
| Model revision |  |
| Objective |  |
| Quality constraint |  |
| Workload distribution |  |
| Holdout split |  |
| Expiration condition | Any environment/model/runtime fingerprint change |

## Piecewise Rules

| Priority | Predicate | Selected plan | Expected cost | Evidence | Holdout regret |
|---:|---|---|---:|---|---:|
|  |  |  |  |  |  |

## Comparison against fixed plans

Report expected objective over the complete holdout distribution. Include best fixed plan, default runtime settings, oracle-per-bucket lower bound, and EdgeFlow policy.

## Amortization view

| Requests/session | Selected plan | Startup cost | Request cost | Total session cost |
|---:|---|---:|---:|---:|
| 1 |  |  |  |  |
| 5 |  |  |  |  |
| 20 |  |  |  |  |
| 100 |  |  |  |  |
| 1000 |  |  |  |  |

## Safety fallback

Document the correctness-proven fallback plan and every dispatch condition that triggers it.

## Drift monitoring

- model/runtime/hash mismatch;
- prompt-length distribution PSI/KS;
- observed latency regret;
- OOM/fallback rate;
- policy expiry and retune trigger.
