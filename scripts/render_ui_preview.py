#!/usr/bin/env python3
"""Render the dependency-free UI prototype to PNG with Playwright.

Playwright is optional. This script embeds CSS/JS into the document so it can run
in restricted environments that block file:// or localhost navigation.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "ui-prototype"
HTML = ROOT / "index.html"
CSS = ROOT / "styles.css"
JS = ROOT / "app.js"
OUTPUT = ROOT / "preview.png"

try:
    from playwright.sync_api import sync_playwright
except ImportError as exc:
    raise SystemExit("Install Playwright and a Chromium browser to render the preview") from exc

html = HTML.read_text(encoding="utf-8")
html = html.replace('<link rel="stylesheet" href="styles.css" />', f"<style>{CSS.read_text(encoding='utf-8')}</style>")
html = html.replace('<script src="app.js"></script>', f"<script>{JS.read_text(encoding='utf-8')}</script>")

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
    page = browser.new_page(viewport={"width": 1600, "height": 1200}, device_scale_factor=1)
    page.set_content(html, wait_until="load")
    page.screenshot(path=str(OUTPUT), full_page=True)
    browser.close()
print(OUTPUT)
