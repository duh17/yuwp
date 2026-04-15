#!/usr/bin/env -S uv run --python 3.14 --script
# /// script
# requires-python = ">=3.14"
# dependencies = ["pillow"]
# ///
"""Generate Yuwp app icon.

Design target:
- Pixel-mic primary mark (clear at small sizes)
- Dark, neutral "liquid glass" style backplate
- Layered output for future Icon Composer migration

Outputs:
- icon-layers/icon-preview.png
- icon-layers/Yuwp.icns
- internal/icon-exploration/liquid-glass-layers/*.png (debug/reference layers)
"""

from pathlib import Path
import random
import shutil
import subprocess

from PIL import Image, ImageDraw, ImageFilter

CANVAS = 1024
RADIUS = 224

# Refined palette (less blue, more neutral graphite + aqua accents)
PLATE_TOP = (18, 22, 30)
PLATE_BOTTOM = (6, 9, 14)
GLOW_TEAL = (86, 223, 210)
GLOW_BLUE = (58, 120, 218)

PIXEL_WHITE = (245, 247, 244, 255)
PIXEL_SOFT = (219, 232, 239, 255)
SLOT_DARK = (9, 20, 46, 230)

SPARK_AQUA = (118, 236, 226, 255)
SPARK_MINT = (211, 247, 236, 255)
SPARK_GOLD = (240, 229, 145, 255)


def lerp(a: int, b: int, t: float) -> int:
    return int(a + (b - a) * t)


def rounded_alpha_mask() -> Image.Image:
    mask = Image.new("L", (CANVAS, CANVAS), 0)
    d = ImageDraw.Draw(mask)
    d.rounded_rectangle([0, 0, CANVAS - 1, CANVAS - 1], radius=RADIUS, fill=255)
    return mask


def build_backplate_layer() -> Image.Image:
    base = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))

    # Vertical neutral gradient.
    grad = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 255))
    gd = ImageDraw.Draw(grad)
    for y in range(CANVAS):
        t = y / (CANVAS - 1)
        color = (
            lerp(PLATE_TOP[0], PLATE_BOTTOM[0], t),
            lerp(PLATE_TOP[1], PLATE_BOTTOM[1], t),
            lerp(PLATE_TOP[2], PLATE_BOTTOM[2], t),
            255,
        )
        gd.line([(0, y), (CANVAS, y)], fill=color)

    # Optical glow blobs.
    glow = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    gl = ImageDraw.Draw(glow)
    gl.ellipse([180, 110, 760, 730], fill=GLOW_TEAL + (58,))
    gl.ellipse([380, 200, 940, 880], fill=GLOW_BLUE + (54,))
    glow = glow.filter(ImageFilter.GaussianBlur(72))
    grad.alpha_composite(glow)

    # Film grain (very subtle).
    noise = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    nd = ImageDraw.Draw(noise)
    random.seed(42)
    for _ in range(15000):
        x = random.randint(0, CANVAS - 1)
        y = random.randint(0, CANVAS - 1)
        v = random.randint(170, 238)
        a = random.randint(6, 16)
        nd.point((x, y), fill=(v, v, v, a))
    noise = noise.filter(ImageFilter.GaussianBlur(0.5))
    grad.alpha_composite(noise)

    mask = rounded_alpha_mask()
    base.paste(grad, (0, 0), mask)

    # Liquid-glass edge polish.
    edge = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    ed = ImageDraw.Draw(edge)
    ed.rounded_rectangle([4, 4, CANVAS - 5, CANVAS - 5], radius=RADIUS, outline=(179, 206, 255, 58), width=4)
    ed.rounded_rectangle([14, 14, CANVAS - 15, CANVAS - 15], radius=RADIUS - 8, outline=(154, 185, 240, 26), width=2)

    edge = edge.filter(ImageFilter.GaussianBlur(1.1))
    base.alpha_composite(edge)

    return base


def draw_cells(draw: ImageDraw.ImageDraw, cells: list[tuple[int, int]], origin: tuple[int, int], step: int, color: tuple[int, int, int, int]) -> None:
    ox, oy = origin
    for gx, gy in cells:
        x = ox + gx * step
        y = oy + gy * step
        draw.rectangle([x, y, x + step - 1, y + step - 1], fill=color)


def plus_shape(cx: int, cy: int) -> list[tuple[int, int]]:
    return [(cx, cy), (cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)]


def corner_shape(cx: int, cy: int, dir_: str) -> list[tuple[int, int]]:
    if dir_ == "ne":
        return [(cx, cy), (cx + 1, cy), (cx, cy - 1)]
    if dir_ == "nw":
        return [(cx, cy), (cx - 1, cy), (cx, cy - 1)]
    if dir_ == "se":
        return [(cx, cy), (cx + 1, cy), (cx, cy + 1)]
    return [(cx, cy), (cx - 1, cy), (cx, cy + 1)]


def build_sparks_layer(step: int, origin: tuple[int, int]) -> Image.Image:
    layer = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    groups: list[tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int, int, int]]] = []

    # Small pixel pi glyph (Π-like) as requested.
    pi_shape = [(0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (0, 2), (2, 2)]

    # Upper constellation, intentionally sparse.
    groups += [
        (plus_shape(0, 0), (244, 142), SPARK_AQUA),
        (plus_shape(0, 0), (507, 142), SPARK_AQUA),
        (plus_shape(0, 0), (686, 142), SPARK_GOLD),
        ([(0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (0, 2), (1, 2), (2, 2)], (444, 174), SPARK_AQUA),
        (corner_shape(0, 0, "se") + corner_shape(2, 2, "nw"), (220, 270), SPARK_AQUA),
        (corner_shape(0, 0, "sw") + corner_shape(2, 0, "se"), (760, 258), SPARK_AQUA),
        ([(0, 0), (1, 1), (2, 2), (1, 3), (0, 4)], (640, 258), SPARK_GOLD),
        ([(0, 0), (4, 0), (1, 1), (3, 1), (2, 2)], (470, 330), SPARK_GOLD),
        (pi_shape, (822, 140), SPARK_MINT),
        (plus_shape(0, 0), (336, 414), SPARK_MINT),
        ([(0, 0)], (428, 408), SPARK_GOLD),
        ([(0, 0)], (607, 408), SPARK_GOLD),
    ]

    for cells, local_origin, color in groups:
        draw_cells(d, cells, local_origin, step, color)

    # Glow pass.
    glow = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    for cells, local_origin, color in groups:
        c = (color[0], color[1], color[2], 82)
        draw_cells(gd, cells, local_origin, step, c)
    glow = glow.filter(ImageFilter.GaussianBlur(7))
    layer.alpha_composite(glow)

    return layer


def mic_body_cells() -> list[tuple[int, int]]:
    c: list[tuple[int, int]] = []

    # Gemini-inspired pixel microphone silhouette.
    # Slightly wider head, clearer body, stronger cradle.
    c += [(7, 0), (8, 0)]
    c += [(6, 1), (7, 1), (8, 1), (9, 1)]
    c += [(5, 2), (6, 2), (7, 2), (8, 2), (9, 2), (10, 2)]
    for y in [3, 4, 5, 6, 7, 8, 9]:
        c += [(4, y), (5, y), (6, y), (7, y), (8, y), (9, y), (10, y), (11, y)]

    # Taper toward the stem.
    c += [(5, 10), (6, 10), (7, 10), (8, 10), (9, 10), (10, 10)]
    c += [(6, 11), (7, 11), (8, 11), (9, 11)]
    c += [(7, 12), (8, 12)]

    # U cradle (thicker side rails).
    for y in range(8, 15):
        c += [(2, y), (3, y), (12, y), (13, y)]
    c += [(3, 15), (4, 15), (5, 15), (6, 15), (9, 15), (10, 15), (11, 15), (12, 15)]

    # Stem and base.
    c += [(7, 16), (8, 16), (7, 17), (8, 17)]
    c += [(6, 18), (7, 18), (8, 18), (9, 18)]

    return sorted(set(c))


def build_mic_layer(step: int, origin: tuple[int, int]) -> Image.Image:
    layer = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    cells = mic_body_cells()

    # Shadow under mic mark.
    shadow = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    draw_cells(sd, cells, (origin[0] + 5, origin[1] + 8), step, (0, 0, 0, 130))
    shadow = shadow.filter(ImageFilter.GaussianBlur(8))
    layer.alpha_composite(shadow)

    # Main white body.
    draw_cells(draw, cells, origin, step, PIXEL_WHITE)

    # Subtle cool highlights on body.
    hi = [(7, 4), (8, 4), (7, 6), (8, 6), (7, 8), (8, 8), (7, 10), (8, 10)]
    draw_cells(draw, hi, origin, step, PIXEL_SOFT)

    # Grille slots as dark bars (overlay, not transparent holes).
    slots: list[tuple[int, int]] = []
    for y in [4, 6, 8, 10]:
        slots += [(6, y), (7, y), (8, y), (9, y)]
    draw_cells(draw, slots, origin, step, SLOT_DARK)

    return layer


def compose_icon() -> tuple[Image.Image, dict[str, Image.Image]]:
    step = 22
    mic_origin = (365, 430)

    backplate = build_backplate_layer()
    sparks = build_sparks_layer(step, mic_origin)
    mic = build_mic_layer(step, mic_origin)

    out = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    out.alpha_composite(backplate)
    out.alpha_composite(sparks)
    out.alpha_composite(mic)

    # enforce rounded alpha
    out.putalpha(rounded_alpha_mask())

    layers = {
        "1_backplate": backplate,
        "2_sparks": sparks,
        "3_mic": mic,
        "4_composite": out,
    }
    return out, layers


def create_icns(master: Image.Image, output_dir: Path) -> Path:
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
        # Nearest-neighbor preserves intentional pixel edges.
        resized = master.resize((size, size), Image.Resampling.NEAREST)
        resized.save(iconset / name)

    icns_path = output_dir / "Yuwp.icns"
    subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(icns_path)], check=True)
    shutil.rmtree(iconset)
    return icns_path


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    output_dir = repo / "icon-layers"
    layers_dir = repo / "internal" / "icon-exploration" / "liquid-glass-layers"
    layers_dir.mkdir(parents=True, exist_ok=True)

    icon, layers = compose_icon()

    preview = output_dir / "icon-preview.png"
    icon.save(preview)
    print(f"Preview: {preview}")

    for name, image in layers.items():
        layer_path = layers_dir / f"{name}.png"
        image.save(layer_path)
    print(f"Debug layers: {layers_dir}")

    icns = create_icns(icon, output_dir)
    print(f"Icon:         {icns}")


if __name__ == "__main__":
    main()
