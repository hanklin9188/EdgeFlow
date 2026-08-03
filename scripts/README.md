# Package Utility Scripts

- `validate_package.py` — parses specs, validates example instances, checks internal links, skill contracts, UI safety labels, and the package manifest.
- `build_package_manifest.py` — writes SHA-256 and byte size for every file in `PACKAGE_MANIFEST.json`.
- `render_ui_preview.py` — optionally regenerates the UI screenshot with Playwright.

Recommended sequence before creating a release ZIP:

```bash
python scripts/build_package_manifest.py
python scripts/validate_package.py
```
