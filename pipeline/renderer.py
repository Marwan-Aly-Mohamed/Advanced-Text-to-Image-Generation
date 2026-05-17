"""
renderer.py — HTML/SVG → PNG via Playwright.
"""

import os
import time
import tempfile
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

VIEWPORT_W = 960
VIEWPORT_H = 640


def _render_html(html: str, out_path: Path):
    with tempfile.NamedTemporaryFile(
        suffix=".html", delete=False, mode="w", encoding="utf-8"
    ) as f:
        f.write(html)
        tmp = f.name
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": VIEWPORT_W, "height": VIEWPORT_H})
            page.goto(f"file://{os.path.abspath(tmp)}")
            page.wait_for_load_state("networkidle")
            page.screenshot(path=str(out_path), full_page=False)
            browser.close()
    finally:
        os.unlink(tmp)


class HTMLRenderer:

    def __init__(self):
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch()
                browser.close()
            print("[renderer] Playwright / Chromium ready.")
        except Exception as e:
            raise RuntimeError(f"Playwright not found. Run: playwright install chromium\n{e}")

    def render(self, html: str) -> Path:
        # Save for debugging
        (OUTPUTS_DIR / "last_debug.html").write_text(html, encoding="utf-8")

        out_path = OUTPUTS_DIR / f"answer_{int(time.time())}.png"
        _render_html(html, out_path)

        img = Image.open(out_path)
        img.verify()
        return out_path
