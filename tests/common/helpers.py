from __future__ import annotations

__all__ = ("bilinear_sample",)

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _reflected(coords: NDArray[np.float64], size: int) -> NDArray[np.float64]:
    """Fold coordinates back over the edge pixels, which is what reflection is."""
    last = size - 1
    if last == 0:
        return np.zeros_like(coords)

    folded = np.abs(coords) % (2 * last)

    return np.where(folded > last, 2 * last - folded, folded)


def bilinear_sample(
    image: NDArray[np.float64],
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    mode: Literal["border", "zeros", "reflection"] = "border",
) -> NDArray[np.float64]:
    """Sample `image` at `(xs, ys)` pixel coordinates, written out by hand.

    What the warp is checked against, so the check does not run through
    `grid_sample` a second time: four neighbours weighted by the fractional part,
    with each out-of-bounds policy spelled out rather than named.

    Args:
        image: The `(H, W)` field to sample.
        xs: The horizontal coordinates to sample at, in pixels.
        ys: The vertical coordinates, in pixels.
        mode: What a coordinate outside the image reads. `"border"` clamps it,
            `"zeros"` reads zero for the neighbours that fall outside, and
            `"reflection"` folds it back over the edge. Defaults to `"border"`.

    Returns:
        The sampled values, shaped like `xs`.
    """
    image = np.asarray(image, dtype=np.float64)
    height, width = image.shape

    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)

    if mode == "border":
        xs = np.clip(xs, 0, width - 1)
        ys = np.clip(ys, 0, height - 1)
    elif mode == "reflection":
        xs = _reflected(xs, width)
        ys = _reflected(ys, height)

    left = np.floor(xs).astype(int)
    top = np.floor(ys).astype(int)
    across = xs - left
    down = ys - top

    def at(x: NDArray[np.int_], y: NDArray[np.int_]) -> NDArray[np.float64]:
        inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        taken = image[np.clip(y, 0, height - 1), np.clip(x, 0, width - 1)]

        return np.where(inside, taken, 0.0)

    upper = at(left, top) * (1 - across) + at(left + 1, top) * across
    lower = at(left, top + 1) * (1 - across) + at(left + 1, top + 1) * across

    return upper * (1 - down) + lower * down
