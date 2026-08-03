# EdgeFlow Release Checklist

## Code

- [ ] Release commit is clean, tagged, and signed when possible.
- [ ] CI passes on Linux/WSL-compatible smoke path.
- [ ] Optional backends fail gracefully when unavailable.
- [ ] No secrets, model weights, gated data, profiler dumps with private paths, or local usernames are included.
- [ ] License, NOTICE, third-party attribution, and model/data terms are reviewed.

## Results

- [ ] Every public number is `source_type=measured`.
- [ ] Every headline row has a validation verdict and raw compact artifact.
- [ ] Demo UI is visibly labeled and cannot be mistaken for measured output.
- [ ] Cross-runtime table passed protocol parity audit.
- [ ] Cold start, steady state, engine-only, and end-to-end are separate.
- [ ] Failed/OOM configurations are represented, not silently omitted.
- [ ] Confidence intervals, sample counts, and practical-effect thresholds are published.

## Claims

- [ ] Claims include GPU, model, workload, backend version, metric definition, and scope.
- [ ] Causal language is used only at evidence level E3+.
- [ ] No RTX 4080 SUPER result is generalized to datacenter GPUs.
- [ ] No single-shape kernel result is advertised as universal.
- [ ] Quality and correctness constraints accompany speed claims.

## Documentation and UX

- [ ] README answers motivation, difference, method, results, reproduction, and limitations in 90 seconds.
- [ ] Local Web Console rejects non-loopback bind, cross-origin writes, missing control token, oversized bodies, arbitrary commands, and path traversal.
- [ ] Background GPU jobs are single-worker, cancellable, and failure-isolated.
- [ ] Dashboard chart points link to run artifacts.
- [ ] Color is not the only status signal.
- [ ] Keyboard navigation, focus states, contrast, reduced motion, and mobile layouts are tested.
- [ ] Screenshots use measured or clearly labeled demo data.

## Reproduction

- [ ] Fresh-machine setup documented.
- [ ] Hardware capture and backend probe commands tested.
- [ ] Screening matrix can be reproduced independently.
- [ ] Package validation script passes.
- [ ] Artifact manifest and SHA-256 checksums included.
