# EdgeFlow UI Prototype

A dependency-free static prototype for the **Calm Technical Observatory** visual system.

```bash
python -m http.server 8788 --directory ui-prototype
```

Open `http://127.0.0.1:8788`. This port is reserved for the clearly labelled demo prototype; the production Local Control Console uses `8787`.

## Included interactions

- Light/dark theme with persisted preference.
- Workload-conditioned plan cards.
- Evidence-chain inspector.
- Run search and validation filter.
- Tune request dialog.
- Responsive desktop, tablet, and narrow layouts.

## Production conversion requirements

1. Replace all hard-coded values with processed JSON generated from validated artifacts.
2. Preserve `source_type` and show `DEMO`, `MEASURED`, or `ESTIMATED` beside every metric.
3. Make every chart point and policy rule open a provenance drawer.
4. Render distributions and confidence intervals rather than only summary values.
5. Never expose local absolute paths, prompts, credentials, or gated data.
6. Add accessibility testing for keyboard flow, contrast, reduced motion, table semantics, and screen-reader labels.
7. Add screenshot regression tests at desktop/tablet/mobile widths.

The prototype intentionally uses system fonts and no CDN, external analytics, or network request.
