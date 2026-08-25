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
from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import TYPE_CHECKING, Any, Final, Self

if TYPE_CHECKING:
    from collections.abc import Iterable

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


def _score(document: Mapping[str, Any], key: str) -> float | None:
    """Read `key` as a score that may be absent, refusing a non-finite one."""
    return None if document.get(key) is None else _number(document, key)


# ========================== #
#           Scores           #
# ========================== #


@dataclass(frozen=True, slots=True)
class FrameEvaluation:
    """What one pair of frames scored, and the flow between them.

    A score is a finite number or it is absent, and nothing in between: JSON has
    no infinity to write and the fold has nothing to do with one, so a
    non-finite score is taken as absent here rather than carried to be dropped
    later. What is lost is only why it is absent, and what a metric that is
    always computed cannot say is that it was not.

    A duplicated frame is how that happens: the reconstruction is exact, `mse`
    is zero, and `psnr` has nowhere to go. The count survives as the difference
    between a fold's `pairs` and its `scored`, and which pair it was survives as
    the absence here.

    Attributes:
        source: The frame this pair starts from, which is what a flow is
            labelled by and so what names the score.
        ssim: Structural similarity of the reconstruction against `frame1`.
        ssim_floor: What a zero flow would have scored, which `ssim` is read
            above rather than on its own.
        psnr: Peak signal-to-noise ratio of the same reconstruction, in dB,
            which an exact reconstruction leaves absent.
        mse: Mean squared error of it.
        mae: Mean absolute error of it.
        magnitude: Mean `|flow|` in pixels, which is how much motion was found.
        fb_error: Mean forward-backward inconsistency in pixels, absent where
            no estimator was there to compute the reverse flow.
    """

    source: str
    ssim: float | None
    ssim_floor: float | None
    psnr: float | None
    mse: float | None
    mae: float | None
    magnitude: float | None
    fb_error: float | None = None

    def __post_init__(self) -> None:
        """Take a non-finite score as absent, there being nowhere to put one."""
        for metric in METRICS:
            value = getattr(self, metric)
            if value is not None and not isfinite(value):
                object.__setattr__(self, metric, None)

    @property
    def gain(self) -> float | None:
        """How far `ssim` rose above what doing nothing would have scored."""
        if self.ssim is None or self.ssim_floor is None:
            return None

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

        A non-finite score is refused rather than read as absent: what wrote
        it was not this, and taking it for an absence would be a guess.

        Raises:
            ValueError: If a key it needs is absent or unreadable.
        """
        scores = {metric: _score(document, metric) for metric in METRICS}

        return cls(source=_entry(document, "source", str), **scores)


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
    """What one sequence scored, over the pairs it was measured on.

    The pairs are kept, not only the fold of them, so a document carries what
    it was folded from. That is what lets a run split into chunks be folded
    again from its parts, and what a reader goes to when a mean is not the
    whole story.

    Attributes:
        source: The name the sequence has in its dataset.
        frames: What each pair scored, in the order they were measured.
        pairs: How many flows the sequence answered, which is one fewer than
            the frames it holds and is what every `scored` is read against.
        metrics: What each metric scored, by name, over the finite ones.

    Raises:
        ValueError: If there are no pairs, since a sequence that answered
            nothing has nothing to say and a part standing for it would count
            as covered.
    """

    source: str
    frames: tuple[FrameEvaluation, ...]
    pairs: int = field(init=False)
    metrics: Mapping[str, Measured] = field(init=False)

    def __post_init__(self) -> None:
        """Fold the pairs, one metric at a time."""
        if not self.frames:
            msg = f"evaluation is undefined: {self.source!r} answered no pair"
            raise ValueError(msg)

        folded = {
            metric: Measured.over(frame.score(metric) for frame in self.frames)
            for metric in METRICS
        }

        object.__setattr__(self, "pairs", len(self.frames))
        object.__setattr__(self, "metrics", folded)

    def __len__(self) -> int:
        """The number of pairs folded here."""
        return len(self.frames)

    def dropped(self, metric: str) -> int:
        """How many of the pairs this metric did not come back finite for."""
        return self.pairs - self.metrics[metric].scored

    def to_dict(self) -> dict[str, Any]:
        """Return the evaluation as plain data, ready to be written as JSON."""
        return asdict(self)

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> Self:
        """Rebuild one sequence's evaluation from its `source` and its `frames`.

        The fold is taken again rather than read back, so a document whose
        numbers were edited by hand cannot disagree with the pairs under them.

        Raises:
            ValueError: If either key is absent, or a pair cannot be read.
        """
        frames = _entry(document, "frames", (list, tuple))

        return cls(
            _entry(document, "source", str),
            tuple(FrameEvaluation.from_dict(frame) for frame in frames),
        )


@dataclass(frozen=True, slots=True)
class DatasetEvaluation:
    """What a dataset scored, over the sequences it covers.

    Folding the parts in one pass rather than merging folded documents is what
    keeps this exact: every sequence is in view at once, so the ends are the
    real ends and the weights the real weights however the run was split.

    Attributes:
        source: The dataset root the run read, which is what tells two
            documents apart when someone comes to merge them.
        sequences: What each sequence scored, in the order they were folded.
        pairs: The flows every sequence answered together.
        metrics: What each metric scored across them, with the ends and who
            reached them.

    Raises:
        ValueError: If there are no sequences, or if two are filed under one
            name, which would leave one out of every fold without saying so.
    """

    source: str
    sequences: tuple[SequenceEvaluation, ...]
    pairs: int = field(init=False)
    metrics: Mapping[str, Spread] = field(init=False)

    def __post_init__(self) -> None:
        """Fold the sequences, one metric at a time."""
        if not self.sequences:
            msg = f"evaluation is undefined: {self.source!r} holds no sequence"
            raise ValueError(msg)

        names = [one.source for one in self.sequences]
        if len(set(names)) != len(names):
            msg = f"a sequence appears twice in {self.source!r}: {sorted(names)}"
            raise ValueError(msg)

        folded = {
            metric: Spread.across(
                {one.source: one.metrics[metric] for one in self.sequences}
            )
            for metric in METRICS
        }

        object.__setattr__(self, "pairs", sum(one.pairs for one in self.sequences))
        object.__setattr__(self, "metrics", folded)

    def __len__(self) -> int:
        """The number of sequences folded here."""
        return len(self.sequences)

    def __str__(self) -> str:
        """The one axis the document exists for, shortened for reading.

        Reconstruction alone says little without what the pair scored against
        each other, so the gain over that floor is given beside it.
        """
        ssim = self.metrics["ssim"].mean
        gain = ssim - self.metrics["ssim_floor"].mean

        return f"SSIM {ssim:.4f} ({gain:+.4f})"

    def dropped(self, metric: str) -> int:
        """How many pairs this metric did not come back finite for.

        Summed over the dataset, this is how many duplicated frames and empty
        fields it holds: an exact reconstruction is the only way to reach one.
        """
        return self.pairs - self.metrics[metric].scored

    def to_dict(self) -> dict[str, Any]:
        """Return the evaluation as plain data, ready to be written as JSON."""
        return asdict(self)

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> Self:
        """Rebuild a dataset's evaluation from its `source` and its `sequences`.

        Raises:
            ValueError: If either key is absent, or a sequence cannot be read.
        """
        sequences = _entry(document, "sequences", (list, tuple))

        return cls(
            _entry(document, "source", str),
            tuple(SequenceEvaluation.from_dict(one) for one in sequences),
        )
