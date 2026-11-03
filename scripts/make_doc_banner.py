"""Draw the contour banner behind the documentation title.

The banner is a contour map of a smooth random surface, drawn with numpy and
matplotlib from a fixed seed, so every run produces the same image and no
third-party artwork is involved. The surface is a sum of Gaussian bumps at three
scales, which contours like terrain.

Usage:
    MPLBACKEND=Agg python scripts/make_doc_banner.py [--out static/doc_banner.jpg]
"""

from __future__ import annotations

import argparse
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 20260904
WIDTH, HEIGHT = 1800, 420  # pixels
BACKGROUND = "#122e24"  # the masthead green
LINE = "#5d7a68"  # a lichen green, legible on the background without competing
LEVELS = 38


def surface(nx: int = 900, ny: int = 210) -> np.ndarray:
    """A smooth surface on an nx by ny grid: Gaussian bumps at three scales."""
    rng = np.random.default_rng(SEED)
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, HEIGHT / WIDTH, ny)
    grid_x, grid_y = np.meshgrid(x, y)
    z = np.zeros_like(grid_x)
    for count, width, height in ((10, 0.15, 1.0), (55, 0.05, 0.35), (500, 0.012, 0.12)):
        cx = rng.uniform(-0.1, 1.1, count)
        cy = rng.uniform(-0.1, HEIGHT / WIDTH + 0.1, count)
        sign = rng.choice([-1.0, 1.0], count)
        for i in range(count):
            z += sign[i] * height * np.exp(-((grid_x - cx[i]) ** 2 + (grid_y - cy[i]) ** 2) / (2 * width**2))
    # Shifted positive so every contour draws as a solid line; matplotlib dashes
    # the contours of negative levels.
    return z - z.min()


def draw(out: str) -> str:
    z = surface()
    fig = plt.figure(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_facecolor(BACKGROUND)
    fig.patch.set_facecolor(BACKGROUND)
    ax.contour(z, levels=np.linspace(z.min(), z.max(), LEVELS), colors=LINE, linewidths=0.7)
    ax.set_axis_off()
    fig.savefig(out, dpi=100, facecolor=BACKGROUND)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "static", "doc_banner.jpg"))
    args = ap.parse_args()
    print(f"wrote {draw(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
