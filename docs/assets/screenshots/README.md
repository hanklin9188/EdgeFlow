# Dashboard screenshots

The README and the landing page currently use `docs/assets/console-preview.svg`, a vector
illustration of the Local Control Console. It is layout-accurate and always renders, but it
is not a capture of a live session.

When you want real captures, drop them here with these exact filenames and the README /
landing page can swap to them with a one-line change each.

| Filename | What to capture | Notes |
| --- | --- | --- |
| `overview.png` | `01 Overview` — hero band, system overview cards, capability probes | Widest value; use this as the hero |
| `research-map.png` | `02 Research map` — E00–E30 tracks and formal-gate cards | Shows scope honestly |
| `tune-workspace.png` | `03 Tune workspace` — workload builder plus candidate preview | Shows the controlled-input idea |
| `run-explorer.png` | `05 Run explorer` — filtered run table with verdicts | Shows traceability |
| `evidence-policy.png` | `06 Evidence & policy` — evidence chain and policy rule detail | Shows the payoff |
| `demo.gif` | 15–30 s: create workload → screen → open evidence | Optional, per `docs/09` §9.10 |

## Capture recipe

1. `edgeflow serve --host 127.0.0.1 --port 8787`
2. Browser window at **1440 × 900**, page zoom 100%, dark theme.
3. Capture the full viewport (or full page for the Run explorer).
4. Save as PNG, then downscale the long edge to **1600 px** to keep the repo light.

## Rules before committing a screenshot

Per `docs/09_UI_UX_GITHUB_PRESENTATION.md` §9.9:

- No `demo` or `estimated` value may be presented as a result.
- Nothing that looks like a headline performance number unless the underlying run passed G0–G8.
- No raw prompts, private paths, tokens, or absolute filesystem paths in frame.
- Caption every screenshot with what it is, and when it was captured.
