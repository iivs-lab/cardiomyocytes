from __future__ import annotations

__all__ = ("RANGE_LEVELS", "FrameNormalizer", "NormalizerConfig", "RangeLevel")

from dataclasses import dataclass
from typing import Final, Literal

import torch
from beartype import beartype
from jaxtyping import Real, jaxtyped
from kaparoo.utils import ensure_one_of, literal_values
from torch import Tensor

# Which range a frame is scaled from. Not how far the constants are measured
# over so much as how far two frames stay comparable: a level names the widest
# thing whose frames a single pair of constants covers.
type RangeLevel = Literal["given", "sequence", "dataset"]

RANGE_LEVELS: Final[tuple[RangeLevel, ...]] = literal_values(RangeLevel)

type FrameType = Real[Tensor, "*dim H W"]


def _ensure_span(span: tuple[float, float], key: str) -> tuple[float, float]:
    """Return `span`, refusing one whose maximum does not exceed its minimum."""
    minimum, maximum = span
    if maximum <= minimum:
        msg = f"empty {key} [{minimum}, {maximum}]: maximum must exceed minimum"
        raise ValueError(msg)

    return span


def _dtype_span(dtype: torch.dtype) -> tuple[float, float]:
    """Return the span an output of `dtype` covers when nothing else says.

    `[0, 1]` for a float, whose own span bounds nothing worth scaling onto, and
    the whole of `iinfo` for an integer, where every step spent is a step the
    estimator downstream can read a brightness change in.
    """
    if dtype.is_floating_point:
        return 0.0, 1.0

    info = torch.iinfo(dtype)

    return float(info.min), float(info.max)


@dataclass(frozen=True, slots=True)
class FrameNormalizer:
    """Min-max scale a frame from one fixed range onto another, and onto a dtype.

    The range is fixed rather than measured per call, which is what makes this a
    pure function of the frame. Every frame a normalizer touches scales by the
    same two constants however it was read, so the brightness constancy a dense
    estimator assumes survives the scaling, and nothing has to see two frames at
    once to scale either of them.

    Values outside the source range are clamped. That is lossy and expected: a
    range measured across a sequence or a dataset is not a bound on any one
    frame of it.

    Attributes:
        source: The `(min, max)` a frame is scaled from.
        target: The `(min, max)` the output covers.
        dtype: The dtype the output is cast to, rounding on the way where it is
            an integer one.

    Raises:
        ValueError: If either span is empty, or an integer `dtype` cannot hold
            `target`.
    """

    source: tuple[float, float]
    target: tuple[float, float]
    dtype: torch.dtype

    def __post_init__(self) -> None:
        """Refuse a scaling that could not be carried out."""
        _ensure_span(self.source, "source")
        _ensure_span(self.target, "target")

        if self.dtype.is_floating_point:  # a float dtype bounds nothing
            return

        info = torch.iinfo(self.dtype)
        minimum, maximum = self.target
        if minimum < info.min or maximum > info.max:
            span = f"[{minimum}, {maximum}]"
            msg = f"target {span} overflows {self.dtype} [{info.min}, {info.max}]"
            raise ValueError(msg)

    @jaxtyped(typechecker=beartype)
    def apply(self, frame: FrameType) -> FrameType:
        """Scale `frame` onto the target range and dtype, keeping its shape.

        Args:
            frame: The `(*dim, H, W)` frame or frames to scale, of any real
                dtype. Leading dimensions are along for the ride: one pair of
                constants covers them all, so a batch scales as one frame does.
        """
        minimum, maximum = self.source
        low, high = self.target

        normalized = (frame.float() - minimum) / (maximum - minimum)
        scaled = normalized.clamp(0.0, 1.0) * (high - low) + low

        if self.dtype.is_floating_point:
            return scaled.to(self.dtype)

        return scaled.round().clamp(low, high).to(self.dtype)


@dataclass(frozen=True, slots=True)
class NormalizerConfig:
    """Where a normalizer's source range comes from, as one value.

    The level is the whole of the choice. `"sequence"` and `"dataset"` name a
    layer of the range document an earlier run wrote, which the caller reads and
    hands to `build`. `"given"` carries the span itself, which is what makes two
    runs under different filters comparable: a measured range is a property of
    the filter that shaped it, so filter sweeps scale by ranges that are not the
    same range.

    Attributes:
        level: Which range a frame is scaled from.
        source: The span `"given"` scales from. Defaults to `None`, which is
            what a measured level requires.
        target: The span the output covers. Defaults to `None`, which takes the
            output dtype's own.

    Raises:
        ValueError: If `level` is not one this offers, if `"given"` carries no
            `source`, if a measured level carries one, or if either span given
            is empty.
    """

    level: RangeLevel = "dataset"
    source: tuple[float, float] | None = None
    target: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        """Refuse a level nobody offers, and a source its level cannot use."""
        ensure_one_of(self.level, RANGE_LEVELS, name="level")

        if self.level == "given":
            if self.source is None:
                msg = "level 'given' scales from `source`, which is not set"
                raise ValueError(msg)
        elif self.source is not None:
            msg = f"level {self.level!r} is measured: drop `source`, or use 'given'"
            raise ValueError(msg)

        if self.source is not None:
            _ensure_span(self.source, "source")

        if self.target is not None:
            _ensure_span(self.target, "target")

    def build(
        self, dtype: torch.dtype, measured: tuple[float, float] | None = None
    ) -> FrameNormalizer:
        """Construct the normalizer this describes, for frames of `dtype`.

        Args:
            dtype: The dtype the output is cast to, which is the one the
                estimator downstream takes: `EstimatorConfig.FRAME_DTYPE`.
            measured: The range the document holds at this level. Every level
                but `"given"` needs it, and `"given"` refuses it rather than
                scaling by one of two ranges without saying which.

        Raises:
            ValueError: If a measured level was handed no range, if `"given"`
                was handed one, or if the normalizer this describes could not
                scale.
        """
        if self.source is not None:
            if measured is not None:
                msg = f"level {self.level!r} brings its own range: drop `measured`"
                raise ValueError(msg)
            source = self.source
        elif measured is None:
            msg = f"level {self.level!r} scales from the range the document holds"
            raise ValueError(msg)
        else:
            source = measured

        target = _dtype_span(dtype) if self.target is None else self.target

        return FrameNormalizer(source, target, dtype)
