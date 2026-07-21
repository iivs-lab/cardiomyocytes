from __future__ import annotations

__all__ = ("BackwardWarp", "backward_warp")

from typing import Literal

import torch
from beartype import beartype
from jaxtyping import Float32, Real, jaxtyped
from torch import Tensor, nn
from torch.nn.functional import grid_sample

ImageType = Real[Tensor, "*dim H W"]
TransformType = Float32[Tensor, "*dim 2 H W"]
PaddingMode = Literal["border", "zeros", "reflection"]


def _norm_scale(shape: tuple[int, int]) -> tuple[float, float]:
    """The `(x, y)` pixel-to-normalized scale for `shape`: `(2/(W-1), 2/(H-1))`.

    One pixel step spans this much of grid_sample's `[-1, 1]` range
    (`align_corners=True`), for the x and y axes respectively.
    """
    height, width = shape
    return 2.0 / (width - 1), 2.0 / (height - 1)


def _identity_grid(image: Tensor) -> Tensor:
    """The identity sampling grid `(H, W, 2)` for `image`, last dim `(x, y)`.

    The grid a zero transform samples; `_warp_with_grid` broadcasts it over the
    batch and offsets it, so the per-call path needs no `meshgrid`/stack.
    Cacheable -- depends only on `image`'s size and device.
    """
    *_, height, width = image.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=image.device),
        torch.arange(width, dtype=torch.float32, device=image.device),
        indexing="ij",
    )
    scale_x, scale_y = _norm_scale((height, width))
    norm_x = grid_x * scale_x - 1.0  # pixel coords -> [-1, 1]
    norm_y = grid_y * scale_y - 1.0
    return torch.stack((norm_x, norm_y), dim=-1)


def _warp_with_grid(
    image: Tensor, transform: Tensor, base: Tensor, padding_mode: PaddingMode
) -> Tensor:
    # `base` is the identity grid `(H, W, 2)` (fresh or cached); per call we only
    # offset it by the transform -- no meshgrid, no stack -- then sample and
    # restore `image`'s dtype. Batch dims are flattened to one N.
    *batch, height, width = image.shape
    images = image.reshape(-1, height, width)
    transforms = transform.reshape(-1, 2, height, width)

    # sample = grid - transform in normalized coords; `_norm_scale` is the
    # per-axis pixel->[-1, 1] factor. View `transforms` as (N, H, W, 2) to line up
    # with `base`; `scale` is explicitly float32 so the offset is float.
    scale = transforms.new_tensor(_norm_scale((height, width)), dtype=torch.float32)
    grid = base - transforms.permute(0, 2, 3, 1) * scale  # (N, H, W, 2)

    sampled = grid_sample(
        images.float()[:, None],
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )[:, 0]

    if image.dtype.is_floating_point:
        warped = sampled.to(image.dtype)
    else:  # round + clamp to the integer type's range
        info = torch.iinfo(image.dtype)
        warped = sampled.round().clamp(info.min, info.max).to(image.dtype)
    return warped.reshape(*batch, height, width)


@jaxtyped(typechecker=beartype)
def backward_warp(
    image: ImageType,
    transform: TransformType,
    *,
    padding_mode: PaddingMode = "border",
) -> ImageType:
    """Backward-warp `image` by `transform` (bilinear pull sampling), batched.

    Samples `image` at ``grid - transform``, a per-pixel displacement (channel
    0 = dx, 1 = dy). Sampling runs in float32 and the result is cast back to
    `image`'s dtype -- integers rounded and clamped to their range, floats kept
    fractional. Shared leading dims are warped together.

    Args:
        image: `(*dim, H, W)` image(s), any real (integer or float) dtype.
        transform: `(*dim, 2, H, W)` float32 displacement (channel 0 = dx, 1 = dy).
        padding_mode: out-of-bounds policy (`border`, `zeros`, or `reflection`).
    """
    base = _identity_grid(image)
    return _warp_with_grid(image, transform, base, padding_mode)


class BackwardWarp(nn.Module):
    """Backward-warp images by a transform, caching the coordinate grid.

    `forward(image, transform)` takes a `(*dim, H, W)` image of any real dtype and
    a `(*dim, 2, H, W)` float32 displacement (channel 0 = dx, 1 = dy), sampling
    `image` at ``grid - transform`` in float32 and casting back to `image`'s dtype
    (integers rounded and clamped, floats kept fractional). The `(H, W)` grid
    depends only on image size and device, so it is built once and reused across
    same-size calls, rebuilt lazily on a size/device change.

    Args:
        padding_mode: out-of-bounds policy (`border`, `zeros`, or `reflection`).
    """

    def __init__(self, *, padding_mode: PaddingMode = "border") -> None:
        super().__init__()
        self.padding_mode = padding_mode
        self._grid: Tensor | None = None  # cached (H, W, 2) identity grid

    def _base_grid(self, image: Tensor) -> Tensor:
        """The cached `(H, W, 2)` grid for `image`, rebuilt on a size/device change."""
        shape = image.shape[-2:]
        grid = self._grid
        if grid is None or grid.shape[:2] != shape or grid.device != image.device:
            grid = _identity_grid(image)
            self._grid = grid
        return grid

    def forward(self, image: Tensor, transform: Tensor) -> Tensor:
        """Return `image` backward-warped by `transform`, reusing the cached grid."""
        base = self._base_grid(image)
        return _warp_with_grid(image, transform, base, self.padding_mode)
