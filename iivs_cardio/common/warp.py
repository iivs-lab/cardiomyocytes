from __future__ import annotations

__all__ = ("BackwardWarp", "backward_warp")

from typing import Literal

import torch
from beartype import beartype
from jaxtyping import Float32, Real, jaxtyped
from torch import Tensor, nn
from torch.nn.functional import grid_sample

ImageType = Real[Tensor, "*dim H W"]
OffsetType = Float32[Tensor, "*dim 2 H W"]
PaddingMode = Literal["border", "zeros", "reflection"]


def _norm_scale(image: Tensor) -> Tensor:
    """The `(2,)` float32 `(x, y)` pixel-to-normalized scale, on `image`'s device.

    `(2/(W-1), 2/(H-1))` -- one pixel step spans this much of grid_sample's `[-1, 1]`
    range under `align_corners=True`. An axis of extent 1 has no pixel step to
    measure, and scales by `0`.
    """
    *_, height, width = image.shape
    norm_x = 2.0 / (width - 1) if width > 1 else 0.0
    norm_y = 2.0 / (height - 1) if height > 1 else 0.0
    return torch.tensor((norm_x, norm_y), dtype=torch.float32, device=image.device)


def _identity_grid(image: Tensor, scale: Tensor) -> Tensor:
    """The identity sampling grid `(H, W, 2)` for `image`, last dim `(x, y)`.

    The grid a zero offset samples. Like `scale`, it depends only on `image`'s `(H, W)`
    and device, so the two are built together.

    Args:
        image: the field whose `(H, W)` and device the grid is built for.
        scale: `_norm_scale(image)`, ordered `(x, y)` to match the stacked axes.
    """
    *_, height, width = image.shape
    grid_x, grid_y = torch.meshgrid(  # `xy` gives each axis as (H, W), x first
        torch.arange(width, dtype=torch.float32, device=image.device),
        torch.arange(height, dtype=torch.float32, device=image.device),
        indexing="xy",
    )
    return torch.stack((grid_x, grid_y), dim=-1) * scale - 1.0


def _grid_is_stale(grid: Tensor, image: Tensor) -> bool:
    """Test whether `grid` was built for a different size or device than `image`."""
    return grid.shape[:2] != image.shape[-2:] or grid.device != image.device


def _warp_with_grid(
    image: Tensor,
    offset: Tensor,
    grid: Tensor,
    scale: Tensor,
    padding_mode: PaddingMode,
) -> Tensor:
    *batch, height, width = image.shape
    images = image.reshape(-1, 1, height, width)  # the (N, C, H, W) grid_sample wants
    offsets = offset.reshape(-1, 2, height, width)

    # Rebind rather than shift in place: `grid` may be a caller's cached tensor.
    grid = grid + offsets.permute(0, 2, 3, 1) * scale  # (N, H, W, 2)

    sampled = grid_sample(
        images.float(),
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )[:, 0]  # remove redundant channel dim, back to (N, H, W)

    if image.dtype.is_floating_point:
        warped = sampled.to(image.dtype)
    else:  # round + clamp to the integer type's range
        info = torch.iinfo(image.dtype)
        warped = sampled.round().clamp(info.min, info.max).to(image.dtype)
    return warped.reshape(*batch, height, width)


@jaxtyped(typechecker=beartype)
def backward_warp(
    image: ImageType,
    offset: OffsetType,
    *,
    padding_mode: PaddingMode = "border",
) -> ImageType:
    """Sample `image` at ``grid + offset`` (bilinear pull sampling), batched.

    The output pixel at `x` takes `image[x + offset(x)]`, so `offset` says where
    to *read from*, not where to move content to. **Note the sign**: content ends up
    displaced by `-offset`, so to move an image *by* a displacement, negate it.

    An offset already defined *on the output grid* -- one that says where in `image`
    each output position reads from -- is used exactly as given, with no inversion.
    Deriving the opposite direction by negating only approximates that inverse,
    with an error growing as `|offset| * |grad offset|`.

    The coordinate grid is rebuilt on every call; reach for `BackwardWarp` to reuse it
    across same-size warps.

    Args:
        image: `(*dim, H, W)` field(s) to sample, any real (integer or float) dtype
            -- a frame, a mask, or one channel of a multi-channel field.
        offset: `(*dim, 2, H, W)` float32 sampling offset (channel 0 = dx, 1 = dy),
            sharing `image`'s leading dims. Those dims warp together.
        padding_mode: out-of-bounds policy (`border`, `zeros`, or `reflection`).

    Returns:
        The sampled field, shaped and dtyped like `image`. Sampling runs in float32:
        a float dtype keeps its fractional values, an integer dtype is rounded
        and clamped back to its range.

    Raises:
        TypeError: If a shape, dtype, or `padding_mode` breaks the contract above --
            a `jaxtyping.TypeCheckError`, raised at the call boundary.
    """
    scale = _norm_scale(image)
    grid = _identity_grid(image, scale)
    return _warp_with_grid(image, offset, grid, scale, padding_mode)


class BackwardWarp(nn.Module):
    """`backward_warp` with the coordinate grid cached across same-size calls.

    The grid and its scale depend only on `(H, W)` and device, so both are built once
    and reused, rebuilt lazily when either changes.

    See `backward_warp` for the sign convention and why an output-grid offset is used
    unchanged. Unlike it, `forward` skips the runtime typecheck, but holds callers
    to the same contract.

    Args:
        padding_mode: out-of-bounds policy (`border`, `zeros`, or `reflection`).
    """

    def __init__(self, *, padding_mode: PaddingMode = "border") -> None:
        super().__init__()
        self.padding_mode = padding_mode
        self._cache: tuple[Tensor, Tensor] | None = None  # (grid, scale), one size

    def _coords(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """The cached grid and scale for `image`, rebuilt on a size/device change."""
        cache = self._cache

        if cache is None or _grid_is_stale(cache[0], image):
            scale = _norm_scale(image)
            grid = _identity_grid(image, scale)
            cache = grid, scale
            self._cache = cache

        return cache

    def forward(self, image: Tensor, offset: Tensor) -> Tensor:
        """Return `image` sampled at ``grid + offset``, reusing the cached grid.

        Args:
            image: `(*dim, H, W)` field(s) to sample, any real integer or float dtype.
            offset: `(*dim, 2, H, W)` float32 offset (channel 0 = dx, 1 = dy),
                sharing `image`'s leading dims.

        Returns:
            The sampled field, shaped and dtyped like `image` -- integers rounded
            and clamped to their range, floats kept fractional.
        """
        grid, scale = self._coords(image)
        return _warp_with_grid(image, offset, grid, scale, self.padding_mode)
