from __future__ import annotations

__all__ = ("FrameNormalizer", "NormalizationMode")

from typing import Literal

import torch
from beartype import beartype
from jaxtyping import Real, jaxtyped
from kaparoo.utils.optional import unwrap_or_default
from torch import Tensor

NormalizationMode = Literal["perframe", "pairwise", "injected"]
FrameType = Real[Tensor, "*dim H W"]

# The source `(min, max)` a frame is scaled *from*: scalars when injected,
# `(*dim, 1, 1)` tensors when measured, so either broadcasts over the frame. The
# target range a frame is scaled *to* is always a plain `(float, float)`.
_Range = tuple[Tensor | float, Tensor | float]


def _ranges(frames: Tensor) -> tuple[Tensor, Tensor]:
    flat = frames.flatten(start_dim=-2)
    return flat.min(dim=-1).values, flat.max(dim=-1).values


def _validate_source_ranges(minimum: Tensor, maximum: Tensor) -> None:
    """Raise if any measured range is empty, which a uniform frame produces."""
    empty = maximum <= minimum
    if bool(empty.any()):
        count = int(empty.sum())
        msg = f"{count} frame(s) are uniform (max == min); drop or repair them"
        raise ValueError(msg)


def _validate_target(target: tuple[float, float], dtype: torch.dtype) -> None:
    """Raise if `target` is empty, or an integer `dtype` cannot hold it.

    Stands apart from `FrameNormalizer._resolve_target` so the `target_range`
    setter can check a value before storing it.
    """
    minimum, maximum = target
    if maximum <= minimum:
        span = f"[{minimum}, {maximum}]"
        msg = f"empty target_range {span}: maximum must exceed minimum"
        raise ValueError(msg)

    if dtype.is_floating_point:  # a float dtype bounds nothing
        return

    info = torch.iinfo(dtype)
    if minimum < info.min or maximum > info.max:
        span = f"[{minimum}, {maximum}]"
        msg = f"target_range {span} overflows {dtype} [{info.min}, {info.max}]"
        raise ValueError(msg)


class FrameNormalizer:
    """Min-max scale a frame pair onto a target range and dtype.

    `apply(frame1, frame2)` returns both frames scaled, each keeping its
    `(*dim, H, W)` shape. The mode decides only *which* range they scale from:

    - `perframe` — each frame by its own range, which breaks the brightness
      constancy every optical-flow estimator assumes.
    - `pairwise` — both frames by their joint range, so a pair stays comparable.
    - `injected` — both by `source_range`, spanning more frames than one call
      sees, which keeps the result order-independent under random access.
      Sequence or dataset scope is the caller's choice; scaling is identical.

    Args:
        mode: which range each frame is scaled from.
        dtype: output dtype, or `None` to keep each input's own.
        source_range: range to scale from, for `injected` mode; also the
            baseline `reset` returns to.
        target_range: the span the output covers, or `None` for the dtype's own
            -- `[0, 1]` for floats, the full `iinfo` span for integers.

    Raises:
        ValueError: If `source_range` is set on a mode that measures its own,
            either range is empty, or `dtype` cannot hold `target_range`.
    """

    def __init__(
        self,
        mode: NormalizationMode,
        *,
        dtype: torch.dtype | None = None,
        source_range: tuple[float, float] | None = None,
        target_range: tuple[float, float] | None = None,
    ) -> None:
        self.mode = mode
        self.dtype = dtype

        self._target_range: tuple[float, float] | None = None
        if target_range is not None:
            self.target_range = target_range

        self._source_range: tuple[float, float] | None = None
        if source_range is not None:
            self.source_range = source_range

        self._initial_source_range = self._source_range

    @property
    def source_range(self) -> tuple[float, float] | None:
        """The range `injected` mode scales from, until `reset` or the next set."""
        return self._source_range

    @source_range.setter
    def source_range(self, value: tuple[float, float]) -> None:
        if self.mode != "injected":
            msg = f"source_range is for the 'injected' mode, not {self.mode!r}"
            raise ValueError(msg)
        minimum, maximum = value
        if maximum <= minimum:
            span = f"[{minimum}, {maximum}]"
            msg = f"empty source_range {span}: maximum must exceed minimum"
            raise ValueError(msg)
        self._source_range = value

    @property
    def target_range(self) -> tuple[float, float] | None:
        """The span the output covers, or `None` to take `dtype`'s own."""
        return self._target_range

    @target_range.setter
    def target_range(self, value: tuple[float, float]) -> None:
        if self.dtype is not None:  # otherwise checked per input in `apply`
            _validate_target(value, self.dtype)
        self._target_range = value

    def reset(self) -> None:
        """Restore the `source_range` given at construction, dropping any set since.

        Back-to-construction rather than back-to-empty, so ending a sequence does
        not discard configuration.
        """
        self._source_range = self._initial_source_range

    @jaxtyped(typechecker=beartype)
    def apply(
        self, frame1: FrameType, frame2: FrameType
    ) -> tuple[FrameType, FrameType]:
        """Min-max scale both frames per the mode, onto the target range and dtype.

        Args:
            frame1: `(*dim, H, W)` frame(s) of any real dtype.
            frame2: `(*dim, H, W)` frame(s) sharing `frame1`'s leading dims.

        Returns:
            Both frames scaled, each keeping its shape and dtype rule. Values
            outside an injected range are clamped -- lossy, but expected when the
            range was measured on another split.

        Raises:
            ValueError: If the `injected` mode has no `source_range`, a measured
                range is empty (which a uniform frame produces), or an input's
                own dtype cannot hold `target_range`.
        """
        dtype1 = unwrap_or_default(self.dtype, frame1.dtype)
        dtype2 = unwrap_or_default(self.dtype, frame2.dtype)

        source1, source2 = self._resolve_source_ranges(frame1, frame2)

        frame1 = self._scale(frame1, source1, dtype1)
        frame2 = self._scale(frame2, source2, dtype2)
        return frame1, frame2

    def _resolve_source_ranges(
        self, frame1: Tensor, frame2: Tensor
    ) -> tuple[_Range, _Range]:
        """The `(min, max)` each frame scales from, ready to broadcast.

        An injected range comes back as the stored scalars; a measured one as
        `(*dim, 1, 1)` tensors, unsqueezed here so `_scale` need not know which.
        """
        if self.mode == "injected":
            if self._source_range is None:
                msg = f"mode {self.mode!r} has no source_range; set one first"
                raise ValueError(msg)
            return self._source_range, self._source_range

        min1, max1 = _ranges(frame1)
        min2, max2 = _ranges(frame2)

        if self.mode == "pairwise":
            min1 = min2 = torch.minimum(min1, min2)
            max1 = max2 = torch.maximum(max1, max2)

        _validate_source_ranges(min1, max1)
        _validate_source_ranges(min2, max2)

        range1 = (min1[..., None, None], max1[..., None, None])
        range2 = (min2[..., None, None], max2[..., None, None])

        return range1, range2

    def _resolve_target(self, dtype: torch.dtype) -> tuple[float, float]:
        """The `(min, max)` an output of `dtype` scales to.

        `target_range` when set, else the dtype's own span. Re-validates the set
        range because a `dtype` of `None` leaves the output dtype unknown until a
        frame arrives, so assignment could not check it.
        """
        if self._target_range is None:
            if dtype.is_floating_point:
                return 0.0, 1.0
            info = torch.iinfo(dtype)
            return float(info.min), float(info.max)

        _validate_target(self._target_range, dtype)

        return self._target_range

    def _scale(self, frame: Tensor, source: _Range, dtype: torch.dtype) -> Tensor:
        # One affine map for every dtype: the source range folds to `[0, 1]`,
        # which then spans the target range. Integers add only the rounding step.
        source_min, source_max = source
        target_min, target_max = self._resolve_target(dtype)

        normalized = (frame.float() - source_min) / (source_max - source_min)
        normalized = normalized.clamp(0.0, 1.0)
        scaled = normalized * (target_max - target_min) + target_min

        if dtype.is_floating_point:
            return scaled.to(dtype)
        return scaled.round().clamp(target_min, target_max).to(dtype)
