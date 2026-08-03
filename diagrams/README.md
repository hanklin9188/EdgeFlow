# Mermaid Diagram Sources

These `.mmd` files are source-of-truth diagrams for README, documentation, and the dashboard.

- `system_architecture.mmd` — control/data/presentation planes.
- `causal_loop.mmd` — diagnosis and intervention state machine.
- `validation_pipeline.mmd` — gates before a run can enter ranking.
- `policy_synthesis.mmd` — train/calibration/holdout policy flow.
- `repository_map.mmd` — intended implementation repository layout.

For GitHub, paste the content into fenced `mermaid` blocks. For publication-quality SVG, render with a pinned Mermaid CLI version and commit both source and generated SVG. Keep colors synchronized with `specs/ui_tokens.json`.
