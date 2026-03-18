#!/usr/bin/env python3
"""Generate a Minecraft-style grass block texture (64x64, seamless tiling)."""

import random
import math
from PIL import Image

random.seed(42)

W, H = 64, 64
GRASS_HEIGHT = 14  # rows of grass from top

# Color palettes
DIRT_COLORS = [
    (134, 96, 67),
    (121, 85, 58),
    (145, 104, 72),
    (110, 78, 52),
    (128, 90, 62),
    (139, 99, 68),
    (118, 82, 55),
    (150, 108, 76),
    (105, 74, 48),
    (142, 101, 70),
]

GRASS_TOP_COLORS = [
    (89, 157, 50),
    (78, 142, 42),
    (95, 165, 55),
    (72, 135, 38),
    (100, 170, 60),
    (85, 150, 46),
    (68, 128, 35),
    (92, 160, 52),
    (80, 145, 44),
    (105, 175, 65),
]

GRASS_SIDE_COLORS = [
    (75, 135, 40),
    (65, 120, 33),
    (82, 145, 45),
    (58, 110, 28),
    (70, 128, 36),
    (88, 150, 48),
]

# Darker dirt colors for depth
DIRT_DARK_COLORS = [
    (95, 68, 45),
    (88, 62, 40),
    (102, 72, 48),
    (82, 58, 37),
]

# Pebble/stone colors embedded in dirt
PEBBLE_COLORS = [
    (160, 150, 135),
    (140, 132, 118),
    (150, 140, 125),
    (130, 122, 108),
]


def seamless_noise(x, y, w, h, seed=0):
    """Generate a seamless tileable noise value using wrapping."""
    # Use multiple octaves of value noise with wrapping
    val = 0.0
    amp = 1.0
    freq = 1.0
    for _ in range(3):
        # Wrap coordinates
        wx = (x * freq) % w
        wy = (y * freq) % h
        ix = int(wx) % w
        iy = int(wy) % h
        fx = wx - int(wx)
        fy = wy - int(wy)
        # Smoothstep
        fx = fx * fx * (3 - 2 * fx)
        fy = fy * fy * (3 - 2 * fy)
        # Corner values (seeded hash)
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
        amp *= 0.5
        freq *= 2
        seed += 1000
    return val / 1.75


def generate_grass_block():
    img = Image.new('RGBA', (W, H))
    pixels = img.load()

    # Generate seamless noise maps
    dirt_noise = [[seamless_noise(x, y, W, H, seed=100) for x in range(W)] for y in range(H)]
    detail_noise = [[seamless_noise(x, y, W, H, seed=200) for x in range(W)] for y in range(H)]
    grass_edge_noise = [seamless_noise(x, 0, W, 1, seed=300) for x in range(W)]

    for y in range(H):
        for x in range(W):
            # Determine grass edge for this column (irregular boundary)
            edge = GRASS_HEIGHT + int(grass_edge_noise[x] * 5 - 2)
            edge = max(8, min(20, edge))

            n1 = dirt_noise[y][x]
            n2 = detail_noise[y][x]

            if y < edge - 2:
                # Pure grass zone
                idx = int(n1 * len(GRASS_TOP_COLORS)) % len(GRASS_TOP_COLORS)
                r, g, b = GRASS_TOP_COLORS[idx]
                # Add subtle variation
                variation = int((n2 - 0.5) * 16)
                r = max(0, min(255, r + variation))
                g = max(0, min(255, g + variation + 3))
                b = max(0, min(255, b + variation - 2))
            elif y < edge:
                # Grass-dirt transition (side grass colors, darker)
                idx = int(n1 * len(GRASS_SIDE_COLORS)) % len(GRASS_SIDE_COLORS)
                r, g, b = GRASS_SIDE_COLORS[idx]
                # Darken slightly toward dirt
                darken = (y - (edge - 2)) / 2.0
                r = int(r * (1.0 - darken * 0.2))
                g = int(g * (1.0 - darken * 0.15))
                b = int(b * (1.0 - darken * 0.2))
            else:
                # Dirt zone
                idx = int(n1 * len(DIRT_COLORS)) % len(DIRT_COLORS)
                r, g, b = DIRT_COLORS[idx]

                # Add dark patches
                if n2 > 0.7:
                    didx = int(n1 * len(DIRT_DARK_COLORS)) % len(DIRT_DARK_COLORS)
                    dr, dg, db = DIRT_DARK_COLORS[didx]
                    blend = (n2 - 0.7) / 0.3
                    r = int(r * (1 - blend) + dr * blend)
                    g = int(g * (1 - blend) + dg * blend)
                    b = int(b * (1 - blend) + db * blend)

                # Occasional pebbles
                if n2 < 0.12:
                    pidx = int(n1 * len(PEBBLE_COLORS)) % len(PEBBLE_COLORS)
                    r, g, b = PEBBLE_COLORS[pidx]

                # Subtle vertical depth gradient (slightly darker at bottom)
                depth_factor = 1.0 - (y - edge) / (H - edge) * 0.08
                r = int(r * depth_factor)
                g = int(g * depth_factor)
                b = int(b * depth_factor)

            pixels[x, y] = (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)), 255)

    return img


def generate_tile_check(img, filename, tiles_x=10, tiles_y=10):
    """Generate a tiled check image."""
    w, h = img.size
    check = Image.new('RGBA', (w * tiles_x, h * tiles_y))
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            check.paste(img, (tx * w, ty * h))
    check.save(filename)


if __name__ == '__main__':
    out_dir = 'textures'

    img = generate_grass_block()
    img.save(f'{out_dir}/grass_block.png')
    print(f'Saved {out_dir}/grass_block.png')

    generate_tile_check(img, f'{out_dir}/grass_block_tile_check.png')
    print(f'Saved {out_dir}/grass_block_tile_check.png')
