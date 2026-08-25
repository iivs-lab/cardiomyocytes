from __future__ import annotations

__all__ = (
    "METRICS",
    "DatasetEvaluation",
    "FrameEvaluation",
    "Measured",
    "SequenceEvaluation",
    "Spread",
)

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import isfinite
from typing import TYPE_CHECKING, Any, Final, Self

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

# The scores one pair is judged on, in the order a document lists them. `gain`
# is not among them: it is `ssim` above `ssim_floor`, and folding a difference
# of two means is the same as the difference of their folds only where both
# were scored on the same pairs, which a non-finite one breaks.
METRICS: Final[tuple[str, ...]] = (
    "ssim",
    "ssim_floor",
    "psnr",
    "mse",
    "mae",
    "magnitude",
    "fb_error",
)


def _entry[T](
    document: Mapping[str, Any], key: str, kind: type[T] | tuple[type[T], ...]
) -> T:
    """Read `key` from a document, refusing it by name when it cannot be read."""
    value = document.get(key)
    if not isinstance(value, kind):
        msg = f"malformed evaluation document: {key!r} is {value!r}"
        raise ValueError(msg)  # noqa: TRY004

    return value


def _number(document: Mapping[str, Any], key: str) -> float:
    """Read `key` as a score, refusing what only looks like one.

    `bool` is an `int` to `isinstance`, so `true` would otherwise read as 1.0.
    A non-finite score is refused because what is written was already folded
    over the finite ones: one here means the document was not written by this.

    Raises:
        ValueError: If the value is absent, not a number, or not finite.
    """
    value = _entry(document, key, (int, float))
    if isinstance(value, bool) or not isfinite(value):
        msg = f"malformed evaluation document: {key!r} is {value!r}"
        raise ValueError(msg)

    return float(value)


# ========================== #
#           Scores           #
# ========================== #


@dataclass(frozen=True, slots=True)
class FrameEvaluation:
    """What one pair of frames scored, and the flow between them.

    Every score is kept as it came, non-finite ones included, so that folding is
    the one place that decides what to do about them. A duplicated frame reaches
    `mse` zero exactly, and `psnr` is then infinite: a fact about the dataset
    rather than a defect in the metric, and one no fold can recover once it has
    been dropped here.

    Attributes:
        source: The frame this pair starts from, which is what a flow is
            labelled by and so what names the score.
        ssim: Structural similarity of the reconstruction against `frame1`.
        ssim_floor: What a zero flow would have scored, which `ssim` is read
            above rather than on its own.
        psnr: Peak signal-to-noise ratio of the same reconstruction, in dB.
        mse: Mean squared error of it.
        mae: Mean absolute error of it.
        magnitude: Mean `|flow|` in pixels, which is how much motion was found.
        fb_error: Mean forward-backward inconsistency in pixels, or `None`
            where no estimator was there to compute the reverse flow.
    """

    source: str
    ssim: float
    ssim_floor: float
    psnr: float
    mse: float
    mae: float
    magnitude: float
    fb_error: float | None = None

    @property
    def gain(self) -> float:
        """How far `ssim` rose above what doing nothing would have scored."""
        return self.ssim - self.ssim_floor

    def score(self, metric: str) -> float | None:
        """The score `metric` holds, or `None` where it was not measured.

        Raises:
            ValueError: If `metric` is not one this is scored on.
        """
        if metric not in METRICS:
            listed = ", ".join(METRICS)
            msg = f"unsupported metric {metric!r}: expected one of {listed}"
            raise ValueError(msg)

        return getattr(self, metric)

    def to_dict(self) -> dict[str, Any]:
        """Return the scores as plain data, ready to be written as JSON."""
        return asdict(self)

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> Self:
        """Rebuild one pair's scores from what `to_dict` produced.

        A non-finite score is not read back, having been written by something
        other than this: the fold that writes a part leaves them out.

        Raises:
            ValueError: If a key it needs is absent or unreadable.
        """
        fb_error = document.get("fb_error")

        return cls(
            source=_entry(document, "source", str),
            ssim=_number(document, "ssim"),
            ssim_floor=_number(document, "ssim_floor"),
            psnr=_number(document, "psnr"),
            mse=_number(document, "mse"),
            mae=_number(document, "mae"),
            magnitude=_number(document, "magnitude"),
            fb_error=None if fb_error is None else _number(document, "fb_error"),
        )


# ========================== #
#            Folds           #
# ========================== #


@dataclass(frozen=True, slots=True)
class Measured:
    """One metric folded over what was actually scored on it.

    `scored` is not the same as how many pairs there were, and the difference is
    the point of keeping both: a metric that was never measured scores none, and
    one a duplicated frame sent to infinity scores one fewer than its
    neighbours. Weighting a fold by anything else counts what was left out as a
    zero.

    Attributes:
        scored: How many pairs this metric was measured on, finitely.
        mean: The mean over those, or `0` where there were none.
    """

    scored: int
    mean: float

    def __post_init__(self) -> None:
        """Refuse a count that cannot have been reached.

        Raises:
            ValueError: If `scored` is negative, or the mean of nothing is not
                the zero that stands for it.
        """
        if self.scored < 0:
            msg = f"negative score count {self.scored}: expected 0 or more"
            raise ValueError(msg)

        if not self.scored and self.mean:
            msg = f"mean {self.mean} over nothing scored: expected 0"
            raise ValueError(msg)

    @classmethod
    def over(cls, values: Iterable[float | None]) -> Self:
        """Fold the finite scores of one metric, leaving the rest out."""
        finite = [value for value in values if value is not None and isfinite(value)]

        return cls(len(finite), sum(finite) / len(finite) if finite else 0.0)

    def to_dict(self) -> dict[str, Any]:
        """Return the fold as plain data, ready to be written as JSON."""
        return asdict(self)

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> Self:
        """Rebuild a fold from what `to_dict` produced.

        Raises:
            ValueError: If a key it needs is absent or unreadable.
        """
        return cls(_entry(document, "scored", int), _number(document, "mean"))


@dataclass(frozen=True, slots=True)
class Spread(Measured):
    """One metric folded across sequences, with the ends and who reached them.

    A mean alone cannot show the shape this search is most likely to produce: a
    setting that lifts most sequences and collapses a few. Naming the ends is
    what settles the next move, since a worst that differs per setting means the
    setting breaks something and one that stays the same means the sequence
    does.

    Both ends rather than the worse of them, so nothing here has to know which
    end is bad for each metric: low is bad for `ssim`, high for `mse` and
    `fb_error`, and the reader knows that where this cannot.

    Attributes:
        scored: As `Measured`, summed over the sequences.
        mean: As `Measured`, weighted by each sequence's own `scored`.
        minimum: The lowest sequence mean, or `0` where none was scored.
        maximum: The highest, on the same terms.
        min_source: The sequence holding `minimum`, empty where none was.
        max_source: The sequence holding `maximum`, on the same terms.
    """

    minimum: float
    maximum: float
    min_source: str
    max_source: str

    @classmethod
    def across(cls, folded: Mapping[str, Measured]) -> Self:
        """Fold one metric across sequences, weighting each by what it scored.

        The weight is the sequence's own `scored` for this metric rather than
        the pairs it held, which makes the two-level fold exactly the mean over
        every finite score: weighting by pairs would count what was left out as
        a zero.

        Sequences that scored none are left out of the ends as well as the mean,
        so a metric nobody measured reads as absent rather than as zero
        everywhere.

        Args:
            folded: What each sequence scored on this metric, by sequence name.
        """
        taken = {name: one for name, one in folded.items() if one.scored}
        scored = sum(one.scored for one in taken.values())
        if not taken:
            return cls(0, 0.0, 0.0, 0.0, "", "")

        total = sum(one.scored * one.mean for one in taken.values())
        low = min(taken, key=lambda name: taken[name].mean)
        high = max(taken, key=lambda name: taken[name].mean)

        return cls(
            scored,
            total / scored,
            taken[low].mean,
            taken[high].mean,
            low,
            high,
        )

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> Self:
        """Rebuild a spread from what `to_dict` produced.

        Raises:
            ValueError: If a key it needs is absent or unreadable.
        """
        return cls(
            _entry(document, "scored", int),
            _number(document, "mean"),
            _number(document, "minimum"),
            _number(document, "maximum"),
            _entry(document, "min_source", str),
            _entry(document, "max_source", str),
        )


# ========================== #
#         Evaluations        #
# ========================== #


@dataclass(frozen=True, slots=True)
class SequenceEvaluation:
    """What one sequence scored, folded from the pairs it was measured over.

    Attributes:
        source: The name the sequence has in its dataset.
        pairs: How many flows it answered, which is one fewer than the frames it
            holds and is what every metric's `scored` is read against.
        metrics: What each metric scored, by name.
    """

    source: str
    pairs: int
    metrics: Mapping[str, Measured]

    @classmethod
    def over(cls, source: str, frames: Sequence[FrameEvaluation]) -> Self:
        """Fold every pair of one sequence, one metric at a time."""
        metrics = {
            metric: Measured.over(frame.score(metric) for frame in frames)
            for metric in METRICS
        }

        return cls(source, len(frames), metrics)

    def dropped(self, metric: str) -> int:
        """How many of the pairs this metric did not come back finite for."""
        return self.pairs - self.metrics[metric].scored

    def to_dict(self) -> dict[str, Any]:
        """Return the fold as plain data, ready to be written as JSON."""
        return {
            "source": self.source,
            "pairs": self.pairs,
            "metrics": {
                name: one.to_dict() for name, one in sorted(self.metrics.items())
            },
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> Self:
        """Rebuild one sequence's fold from what `to_dict` produced.

        Raises:
            ValueError: If a key it needs is absent or unreadable.
        """
        metrics = _entry(document, "metrics", dict)

        return cls(
            source=_entry(document, "source", str),
            pairs=_entry(document, "pairs", int),
            metrics={
                name: Measured.from_dict(_entry(metrics, name, dict))
                for name in METRICS
            },
        )


@dataclass(frozen=True, slots=True)
class DatasetEvaluation:
    """What a dataset scored, folded from the sequences it was measured over.

    Attributes:
        source: The name the dataset is filed under.
        pairs: The flows every sequence answered together.
        metrics: What each metric scored across them, with the ends and who
            reached them.
    """

    source: str
    pairs: int
    metrics: Mapping[str, Spread]

    @classmethod
    def over(cls, source: str, sequences: Sequence[SequenceEvaluation]) -> Self:
        """Fold every sequence of one dataset, one metric at a time.

        Folding the parts in one pass rather than merging folded documents is
        what keeps this exact: every sequence is in view at once, so the ends
        are the real ends and the weights are the real weights however the run
        was split into chunks.

        Raises:
            ValueError: If two sequences are filed under one name, which would
                leave one of them out of every fold without saying so.
        """
        names = [one.source for one in sequences]
        if len(set(names)) != len(names):
            msg = f"a sequence appears twice in {source!r}: {sorted(names)}"
            raise ValueError(msg)

        metrics = {
            metric: Spread.across(
                {one.source: one.metrics[metric] for one in sequences}
            )
            for metric in METRICS
        }

        return cls(source, sum(one.pairs for one in sequences), metrics)

    def dropped(self, metric: str) -> int:
        """How many pairs this metric did not come back finite for.

        Summed over the dataset, this is how many duplicated frames and empty
        fields it holds: the only way a reconstruction is exact.
        """
        return self.pairs - self.metrics[metric].scored

    def to_dict(self) -> dict[str, Any]:
        """Return the fold as plain data, ready to be written as JSON."""
        return {
            "source": self.source,
            "pairs": self.pairs,
            "metrics": {
                name: one.to_dict() for name, one in sorted(self.metrics.items())
            },
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> Self:
        """Rebuild a dataset's fold from what `to_dict` produced.

        Raises:
            ValueError: If a key it needs is absent or unreadable.
        """
        metrics = _entry(document, "metrics", dict)

        return cls(
            source=_entry(document, "source", str),
            pairs=_entry(document, "pairs", int),
            metrics={
                name: Spread.from_dict(_entry(metrics, name, dict)) for name in METRICS
            },
        )
