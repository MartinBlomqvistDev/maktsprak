"""Screenshot a study heatmap from the live site as a shareable PNG.

The image attached to posts about the LLM study has to match what the site is
serving. Doing that by hand is how it goes stale: after the study was regenerated,
the old PNG still showed the pre-fix Gemini row and contradicted the page it was
posted alongside. This captures it from the deployed page instead.

Light theme is forced explicitly, because headless Chrome will otherwise inherit
the OS preference and render the dark palette.

Requires Playwright and a local Chrome:
    pip install playwright

Usage:
    python scripts/capture_heatmap.py
    python scripts/capture_heatmap.py --which neutral --out neutral.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

URL = "https://maktsprak.se/llm"
PADDING = 64


def capture(url: str, index: int, out: Path, padding: int = PADDING) -> None:
    """Screenshot the nth <figure> on the page and pad it with the page background.

    Args:
        url: Page to load.
        index: Which figure to grab; 0 is the speech heatmap, 1 the neutral control.
        out: Destination PNG path.
        padding: Border in device pixels added around the figure.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page(
            color_scheme="light",
            device_scale_factor=2,
            viewport={"width": 1180, "height": 1000},
        )
        page.goto(url, wait_until="networkidle")
        page.wait_for_timeout(1500)
        figures = page.query_selector_all("figure")
        if index >= len(figures):
            raise SystemExit(f"figure {index} not found; page has {len(figures)}")
        figures[index].screenshot(path=str(out))
        browser.close()

    shot = Image.open(out).convert("RGB")
    background = shot.getpixel((2, 2))
    padded = Image.new("RGB", (shot.width + padding * 2, shot.height + padding * 2), background)
    padded.paste(shot, (padding, padding))
    padded.save(out)
    print(f"Wrote {out} ({padded.width}x{padded.height})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=URL)
    parser.add_argument("--which", choices=["speech", "neutral"], default="speech")
    parser.add_argument("--out", type=Path, default=Path("maktsprak_post_heatmap.png"))
    args = parser.parse_args()
    capture(args.url, 0 if args.which == "speech" else 1, args.out)


if __name__ == "__main__":
    main()
