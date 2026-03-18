#!/usr/bin/env python3
"""Generate a high-resolution Minecraft-style sand block texture (1024x1024, seamless tiling)."""

import random
import math
from PIL import Image

random.seed(42)

W, H = 1024, 1024

# Sand color palettes
SAND_BASE = [
    (219, 199, 151),
    (212, 192, 144),
    (225, 205, 158),
    (207, 186, 138),
    (230, 210, 162),
    (215, 195, 147),
    (222, 202, 154),
    (210, 189, 141),
]

SAND_LIGHT = [
    (235, 218, 172),
    (240, 222, 178),
    (232, 215, 168),
    (238, 220, 175),
]

SAND_DARK = [
    (190, 170, 125),
    (185, 164, 118),
    (195, 175, 130),
    (180, 160, 114),
    (175, 155, 108),
]

SAND_GRAIN = [
    (200, 180, 132),
    (195, 174, 126),
    (205, 185, 138),
    (188, 168, 120),
]


def seamless_noise(x, y, w, h, seed=0, octaves=4):
    """Generate seamless tileable noise with multiple octaves."""
    val = 0.0
    amp = 1.0
    freq = 1.0
    total_amp = 0.0
    for _ in range(octaves):
        wx = (x * freq) % w
        wy = (y * freq) % h
        ix = int(wx) % w
        iy = int(wy) % h
        fx = wx - int(wx)
        fy = wy - int(wy)
        # Smoothstep
        fx = fx * fx * (3 - 2 * fx)
        fy = fy * fy * (3 - 2 * fy)

        def hsh(a, b):
            random.seed(seed + a * 7919 + b * 6271)
            return random.random()

        ix2 = (ix + 1) % w
        iy2 = (iy + 1) % h
        c00 = hsh(ix, iy)
        c10 = hsh(ix2, iy)
        c01 = hsh(ix, iy2)
        c11 = hsh(ix2, iy2)
        top = c00 + (c10 - c00) * fx
        bot = c01 + (c11 - c01) * fx
        val += (top + (bot - top) * fy) * amp
        total_amp += amp
        amp *= 0.5
        freq *= 2
        seed += 1000
    return val / total_amp


def generate_sand_block():
    img = Image.new('RGBA', (W, H))
    pixels = img.load()

    # Generate multiple noise maps at different scales
    noise_large = [[seamless_noise(x, y, W, H, seed=100, octaves=3) for x in range(W)] for y in range(H)]
    noise_medium = [[seamless_noise(x, y, W, H, seed=200, octaves=4) for x in range(W)] for y in range(H)]
    noise_fine = [[seamless_noise(x, y, W, H, seed=300, octaves=5) for x in range(W)] for y in range(H)]
    noise_grain = [[seamless_noise(x, y, W, H, seed=400, octaves=6) for x in range(W)] for y in range(H)]
    noise_speckle = [[seamless_noise(x, y, W, H, seed=500, octaves=2) for x in range(W)] for y in range(H)]

    for y in range(H):
        for x in range(W):
            n_large = noise_large[y][x]
            n_med = noise_medium[y][x]
            n_fine = noise_fine[y][x]
            n_grain = noise_grain[y][x]
            n_speckle = noise_speckle[y][x]

            # Base sand color from large-scale noise
            idx = int(n_large * len(SAND_BASE)) % len(SAND_BASE)
            r, g, b = SAND_BASE[idx]

            # Blend with light/dark based on medium noise
            if n_med > 0.65:
                lidx = int(n_fine * len(SAND_LIGHT)) % len(SAND_LIGHT)
                lr, lg, lb = SAND_LIGHT[lidx]
                blend = (n_med - 0.65) / 0.35
                r = int(r * (1 - blend) + lr * blend)
                g = int(g * (1 - blend) + lg * blend)
                b = int(b * (1 - blend) + lb * blend)
            elif n_med < 0.35:
                didx = int(n_fine * len(SAND_DARK)) % len(SAND_DARK)
                dr, dg, db = SAND_DARK[didx]
                blend = (0.35 - n_med) / 0.35
                r = int(r * (1 - blend) + dr * blend)
                g = int(g * (1 - blend) + dg * blend)
                b = int(b * (1 - blend) + db * blend)

            # Fine grain detail
            grain_var = int((n_fine - 0.5) * 20)
            r += grain_var
            g += grain_var
            b += int(grain_var * 0.7)

            # Individual sand grain texture (very fine noise)
            grain_detail = int((n_grain - 0.5) * 12)
            r += grain_detail
            g += grain_detail
            b += int(grain_detail * 0.6)

            # Occasional darker speckles (small pebbles/shells)
            if n_speckle < 0.06:
                gidx = int(n_fine * len(SAND_GRAIN)) % len(SAND_GRAIN)
                gr, gg, gb = SAND_GRAIN[gidx]
                r, g, b = gr - 15, gg - 15, gb - 10

            # Occasional light sparkle (quartz grains)
            if n_speckle > 0.97:
                sparkle = int((n_speckle - 0.97) / 0.03 * 30)
                r = min(255, r + sparkle + 10)
                g = min(255, g + sparkle + 8)
                b = min(255, b + sparkle + 5)

            # Subtle shadow/depth pattern
            shadow = math.sin(n_large * math.pi * 2) * 0.03
            r = int(r * (1.0 + shadow))
            g = int(g * (1.0 + shadow))
            b = int(b * (1.0 + shadow))

            pixels[x, y] = (
                max(0, min(255, r)),
                max(0, min(255, g)),
                max(0, min(255, b)),
                255
            )

    return img


def generate_tile_check(img, filename, tiles_x=4, tiles_y=4):
    """Generate a tiled check image."""
    w, h = img.size
    check = Image.new('RGBA', (w * tiles_x, h * tiles_y))
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            check.paste(img, (tx * w, ty * h))
    check.save(filename)


if __name__ == '__main__':
    out_dir = 'textures'

    img = generate_sand_block()
    img.save(f'{out_dir}/sand_block_{W}x{H}.png')
    print(f'Saved {out_dir}/sand_block_{W}x{H}.png ({W}x{H})')

    generate_tile_check(img, f'{out_dir}/sand_block_{W}x{H}_tile_check.png')
    print(f'Saved {out_dir}/sand_block_{W}x{H}_tile_check.png')
