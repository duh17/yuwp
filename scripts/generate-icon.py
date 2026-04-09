#!/usr/bin/env -S uv run --python 3.14 --script
# /// script
# requires-python = ">=3.14"
# dependencies = ["pillow"]
# ///
"""Generate Yuwp app icon as pixel art.

Concept: A microphone with radiating sound arcs on a dark background.
Clean, centered, minimal. 16x16 grid of rounded squares.

Outputs 1024x1024 PNG and .icns via iconutil.
"""

from pathlib import Path
from PIL import Image, ImageDraw
import subprocess
import shutil

GRID = 16
CANVAS = 1024
CELL = CANVAS // GRID  # 64px per cell
RADIUS = CELL // 5     # rounded corner radius
GAP = 3               # gap between cells

# Colors
BG = (17, 17, 24)
TEAL = "#4ecdc4"
TEAL_75 = (78, 205, 196, 190)
TEAL_50 = (78, 205, 196, 128)
TEAL_25 = (78, 205, 196, 64)
WHITE = (240, 240, 240, 255)
WHITE_90 = (235, 235, 235, 230)
WHITE_70 = (220, 220, 220, 180)
WHITE_50 = (200, 200, 200, 128)
WHITE_30 = (200, 200, 200, 77)
WHITE_15 = (200, 200, 200, 38)

def draw_cell(draw, row, col, color):
    x = col * CELL + GAP
    y = row * CELL + GAP
    w = CELL - GAP * 2
    h = CELL - GAP * 2
    draw.rounded_rectangle([x, y, x + w, y + h], radius=RADIUS, fill=color)

def generate_icon():
    img = Image.new("RGBA", (CANVAS, CANVAS), BG + (255,))
    draw = ImageDraw.Draw(img)

    # ── Microphone (centered, cols 6-9, rows 2-12) ──

    # Mic head (capsule shape — top rounded)
    #     6  7  8  9
    # 2      ##
    # 3   ## ## ##
    # 4   ## ## ##
    # 5   ## ## ##
    # 6   ## ## ##
    # 7      ##         (neck)
    # 8      ##         (stand)
    # 9   ## ## ##   (cradle arc bottom)
    # 10     ##         (pole)
    # 11     ##         (pole)
    # 12  ## ## ## ##  (base)

    # Mic capsule top
    draw_cell(draw, 2, 7, WHITE_90)
    draw_cell(draw, 2, 8, WHITE_90)

    # Mic body
    for r in range(3, 7):
        draw_cell(draw, r, 6, WHITE_70)
        draw_cell(draw, r, 7, WHITE)
        draw_cell(draw, r, 8, WHITE)
        draw_cell(draw, r, 9, WHITE_70)

    # Grille accents (subtle darker lines on the mic body)
    draw_cell(draw, 3, 7, WHITE_90)
    draw_cell(draw, 3, 8, WHITE_90)
    draw_cell(draw, 5, 7, WHITE_90)
    draw_cell(draw, 5, 8, WHITE_90)

    # Cradle arc (U-shape)
    draw_cell(draw, 3, 5, WHITE_30)
    draw_cell(draw, 4, 5, WHITE_30)
    draw_cell(draw, 5, 5, WHITE_30)
    draw_cell(draw, 6, 5, WHITE_30)
    draw_cell(draw, 7, 5, WHITE_30)
    draw_cell(draw, 7, 6, WHITE_30)
    draw_cell(draw, 7, 7, WHITE_50)
    draw_cell(draw, 7, 8, WHITE_50)
    draw_cell(draw, 7, 9, WHITE_30)
    draw_cell(draw, 7, 10, WHITE_30)
    draw_cell(draw, 3, 10, WHITE_30)
    draw_cell(draw, 4, 10, WHITE_30)
    draw_cell(draw, 5, 10, WHITE_30)
    draw_cell(draw, 6, 10, WHITE_30)

    # Stand
    draw_cell(draw, 8, 7, WHITE_50)
    draw_cell(draw, 8, 8, WHITE_50)
    draw_cell(draw, 9, 7, WHITE_50)
    draw_cell(draw, 9, 8, WHITE_50)

    # Base
    draw_cell(draw, 10, 6, WHITE_30)
    draw_cell(draw, 10, 7, WHITE_50)
    draw_cell(draw, 10, 8, WHITE_50)
    draw_cell(draw, 10, 9, WHITE_30)

    # ── Sound arcs (teal, radiating from mic) ──

    # Arc 1 (close, bright) — 2 cells from mic body
    draw_cell(draw, 3, 12, TEAL_75)
    draw_cell(draw, 4, 12, TEAL_75)
    draw_cell(draw, 5, 12, TEAL_75)

    # Arc 2 (mid, dimmer)
    draw_cell(draw, 2, 14, TEAL_50)
    draw_cell(draw, 3, 14, TEAL_50)
    draw_cell(draw, 4, 14, TEAL_50)
    draw_cell(draw, 5, 14, TEAL_50)
    draw_cell(draw, 6, 14, TEAL_50)

    # Left-side arcs (mirror, subtler)
    draw_cell(draw, 3, 3, TEAL_50)
    draw_cell(draw, 4, 3, TEAL_50)
    draw_cell(draw, 5, 3, TEAL_50)

    draw_cell(draw, 2, 1, TEAL_25)
    draw_cell(draw, 3, 1, TEAL_25)
    draw_cell(draw, 4, 1, TEAL_25)
    draw_cell(draw, 5, 1, TEAL_25)
    draw_cell(draw, 6, 1, TEAL_25)

    # ── Text cursor (bottom-right, small) ──
    # A blinking cursor suggests "text output" —
    # just a single bright teal line
    draw_cell(draw, 12, 11, TEAL_75)
    draw_cell(draw, 13, 11, TEAL_75)
    draw_cell(draw, 12, 12, WHITE_15)
    draw_cell(draw, 12, 13, WHITE_15)
    draw_cell(draw, 13, 12, WHITE_15)

    # ── Ambient scatter (depth) ──
    for r, c, a in [
        (0, 4, 18), (1, 13, 22), (14, 3, 15),
        (14, 14, 20), (12, 0, 12), (0, 10, 14),
    ]:
        draw_cell(draw, r, c, (200, 200, 200, a))

    return img


def create_icns(img: Image.Image, output_dir: Path):
    iconset = output_dir / "Yuwp.iconset"
    iconset.mkdir(exist_ok=True)

    sizes = [
        ("icon_16x16.png", 16),
        ("icon_16x16@2x.png", 32),
        ("icon_32x32.png", 32),
        ("icon_32x32@2x.png", 64),
        ("icon_128x128.png", 128),
        ("icon_128x128@2x.png", 256),
        ("icon_256x256.png", 256),
        ("icon_256x256@2x.png", 512),
        ("icon_512x512.png", 512),
        ("icon_512x512@2x.png", 1024),
    ]

    for name, size in sizes:
        resized = img.resize((size, size), Image.LANCZOS)
        resized.save(iconset / name)

    icns_path = output_dir / "Yuwp.icns"
    subprocess.run(
        ["iconutil", "-c", "icns", str(iconset), "-o", str(icns_path)],
        check=True,
    )
    shutil.rmtree(iconset)
    return icns_path


def main():
    repo = Path(__file__).resolve().parent.parent
    output_dir = repo / "icon-layers"

    img = generate_icon()

    preview = output_dir / "icon-preview.png"
    img.save(preview)
    print(f"Preview: {preview}")

    icns = create_icns(img, output_dir)
    print(f"Icon:    {icns}")

    resources = repo / "Sources" / "Resources"
    resources.mkdir(exist_ok=True)
    shutil.copy2(icns, resources / "Yuwp.icns")
    print(f"Copied:  {resources / 'Yuwp.icns'}")


if __name__ == "__main__":
    main()
