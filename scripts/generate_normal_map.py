#!/usr/bin/env python3
"""Generate a normal map from an albedo texture using Sobel-based height estimation."""

import sys
import math
from PIL import Image

def generate_normal_map(input_path, output_path, strength=2.0):
    img = Image.open(input_path).convert('RGB')
    w, h = img.size
    pixels = img.load()

    # Convert to grayscale heightmap
    height = [[0.0] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            r, g, b = pixels[x, y]
            height[y][x] = (r * 0.299 + g * 0.587 + b * 0.114) / 255.0

    # Compute normals using Sobel filter (seamless wrapping)
    out = Image.new('RGBA', (w, h))
    out_pixels = out.load()

    for y in range(h):
        for x in range(w):
            # Sample heights with wrapping for seamless tiling
            tl = height[(y - 1) % h][(x - 1) % w]
            t  = height[(y - 1) % h][x]
            tr = height[(y - 1) % h][(x + 1) % w]
            l  = height[y][(x - 1) % w]
            r  = height[y][(x + 1) % w]
            bl = height[(y + 1) % h][(x - 1) % w]
            b  = height[(y + 1) % h][x]
            br = height[(y + 1) % h][(x + 1) % w]

            # Sobel operator
            dx = (tr + 2.0 * r + br) - (tl + 2.0 * l + bl)
            dy = (bl + 2.0 * b + br) - (tl + 2.0 * t + tr)

            dx *= strength
            dy *= strength

            # Normal vector (tangent space: X right, Y up, Z out)
            nx = -dx
            ny = -dy
            nz = 1.0

            # Normalize
            length = math.sqrt(nx * nx + ny * ny + nz * nz)
            nx /= length
            ny /= length
            nz /= length

            # Map [-1,1] to [0,255]
            out_pixels[x, y] = (
                int((nx * 0.5 + 0.5) * 255),
                int((ny * 0.5 + 0.5) * 255),
                int((nz * 0.5 + 0.5) * 255),
                255
            )

    out.save(output_path)
    print(f'Saved {output_path}')


if __name__ == '__main__':
    generate_normal_map('textures/sand_block_256x256.png', 'textures/sand_block_256x256_normal.png', strength=5.0)
