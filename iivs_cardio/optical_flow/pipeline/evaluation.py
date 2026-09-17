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
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Self, override

from iivs_cardio.common.pipeline.document import (
    DatasetResult,
    SequenceResult,
    StepResult,
    read_entry,
    read_number,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

# The scores one pair is judged on, in the order a document lists them.
METRICS: Final[tuple[str, ...]] = (
    "ssim",
    "ssim_floor",
    "psnr",
    "mse",
    "mae",
    "magnitude",
    "fb_error",
)


def _score(document: Mapping[str, Any], key: str) -> float | None:
    """Read `key` as a score that may be absent, refusing a non-finite one."""
    return None if document.get(key) is None else read_number(document, key)


# ========================== #
#           Scores           #
# ========================== #


@dataclass(frozen=True, slots=True)
class FrameEvaluation(StepResult):
    """What one pair of frames scored, and the flow between them.

    A score is a finite number or it is absent, and nothing in between: JSON has no
    infinity to write and the summary has nothing to do with one, so a non-finite score
    is taken as absent here rather than carried to be dropped later. What is lost is
    only why it is absent, and what a metric that is always computed cannot say is that
    it was not.

    A duplicated frame is how that happens: the reconstruction is exact, `mse` is zero,
    and `psnr` has nowhere to go. The count survives as the difference between a
    summary's `pairs` and its `scored`, and which pair it was survives as the absence
    here.

    Attributes:
        source: The frame this pair starts from, which is what a flow is labelled by and
            so what names the score.
        scores: What the pair scored on each metric, by name, `None` where it was not
            measured. Every metric of `METRICS` is a key, and nothing else is.

    Metrics:
        ssim: Structural similarity of the reconstruction against the first frame. A
            forward flow is defined on that frame's grid, so sampling the second one
            along it reconstructs the first.
        ssim_floor: What a zero flow would have scored, which `ssim` is read above
            rather than on its own.
        psnr: Peak signal-to-noise ratio of the same reconstruction, in dB, which an
            exact reconstruction leaves absent.
        mse: Mean squared error of it.
        mae: Mean absolute error of it.
        magnitude: Mean `|flow|` in pixels, which is how much motion was found.
        fb_error: Mean forward-backward inconsistency in pixels, absent where no
            estimator was there to compute the reverse flow.

    Raises:
        ValueError: If a metric nothing is scored on is named.
    """

    scores: Mapping[str, float | None]

    def __post_init__(self) -> None:
        """Take a non-finite score as absent, refusing a metric nobody scores on."""
        if unknown := sorted(set(self.scores) - set(METRICS)):
            listed = ", ".join(METRICS)
            msg = f"unsupported metric {unknown[0]!r}: expected one of {listed}"
            raise ValueError(msg)

        kept = {
            metric: found
            if (found := self.scores.get(metric)) is not None and isfinite(found)
            else None
            for metric in METRICS
        }

        object.__setattr__(self, "scores", MappingProxyType(kept))

    def score(self, metric: str) -> float | None:
        """The score `metric` holds, or `None` where it was not measured.

        Raises:
            ValueError: If `metric` is not one this is scored on.
        """
        if metric not in METRICS:
            listed = ", ".join(METRICS)
            msg = f"unsupported metric {metric!r}: expected one of {listed}"
            raise ValueError(msg)

        return self.scores[metric]

    @override
    def to_dict(self) -> dict[str, Any]:
        """Return the scores as plain data, ready to be written as JSON."""
        return {"source": self.source, "scores": dict(self.scores)}

    @override
    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> Self:
        """Rebuild one pair's scores from its `source` and its `scores`.

        A non-finite score is refused rather than read as absent: what wrote it was not
        this, and taking it for an absence would be a guess.

        Raises:
            ValueError: If either key is absent, or a score cannot be read.
        """
        held = read_entry(document, "scores", dict)

        return cls(
            read_entry(document, "source", str),
            {metric: _score(held, metric) for metric in METRICS},
        )


# ========================== #
#          Summaries         #
# ========================== #


@dataclass(frozen=True, slots=True)
class Measured:
    """One metric summarised over what was scored on it.

    Attributes:
        scored: How many pairs this metric was measured on, finitely.
        mean: The mean over those, or `0` where there were none.

    Raises:
        ValueError: If `scored` is negative, or the mean of nothing is not the zero that
            stands for it.
    """

    scored: int
    mean: float

    def __post_init__(self) -> None:
        """Refuse a count that cannot have been reached."""
        if self.scored < 0:
            msg = f"negative score count {self.scored}: expected 0 or more"
            raise ValueError(msg)

        if not self.scored and self.mean:
            msg = f"mean {self.mean} over nothing scored: expected 0"
            raise ValueError(msg)

    @classmethod
    def over(cls, values: Iterable[float | None]) -> Self:
        """Summarise the finite scores of one metric, leaving the rest out."""
        finite = [value for value in values if value is not None and isfinite(value)]

        return cls(len(finite), sum(finite) / len(finite) if finite else 0.0)

    def to_dict(self) -> dict[str, Any]:
        """Return the summary as plain data, ready to be written as JSON."""
        return asdict(self)

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> Self:
        """Rebuild a summary from what `to_dict` produced.

        Raises:
            ValueError: If a key it needs is absent or unreadable.
        """
        return cls(read_entry(document, "scored", int), read_number(document, "mean"))


@dataclass(frozen=True, slots=True)
class Spread(Measured):
    """One metric summarised across sequences, naming the ends.

    A mean alone cannot show the shape this search is most likely to produce: a setting
    that lifts most sequences and collapses a few. Naming the ends is what settles the
    next move, since a worst that differs per setting means the setting breaks something
    and one that stays the same means the sequence does.

    Both ends rather than the worse of them, so nothing here has to know which end is
    bad for each metric: low is bad for `ssim`, high for `mse` and `fb_error`, and the
    reader knows that where this cannot. The mean each end reached is not repeated here,
    the sequence it names carrying it already.

    Attributes:
        scored: How many pairs this metric was measured on, finitely, summed over the
            sequences.
        mean: The mean over those, weighted by each sequence's own `scored`, or `0`
            where there were none.
        min_source: The sequence whose mean was lowest, empty where none was scored.
        max_source: The sequence whose mean was highest, on the same terms.

    Raises:
        ValueError: If `scored` is negative, or the mean of nothing is not the zero that
            stands for it.
    """

    min_source: str
    max_source: str

    @override
    @classmethod
    def over(cls, values: Iterable[float | None]) -> Self:
        """Refuse the sequence-level summary, a spread being taken across sequences.

        Raises:
            TypeError: Always. Use `across`, which weights each sequence by what it
                scored rather than counting them equally.
        """
        msg = f"{cls.__name__} is taken across sequences: use `across`"
        raise TypeError(msg)

    @classmethod
    def across(cls, summarised: Mapping[str, Measured]) -> Self:
        """Summarise one metric across sequences, weighting each by what it scored.

        The weight is the sequence's own `scored` for this metric rather than the pairs
        it held, which makes the two-level summary exactly the mean over every finite
        score: weighting by pairs would count what was left out as a zero.

        Sequences that scored none are left out of the ends as well as the mean, so a
        metric nobody measured reads as absent rather than as zero everywhere.

        Args:
            summarised: What each sequence scored on this metric, by sequence name.
        """
        taken = {name: one for name, one in summarised.items() if one.scored}
        if not taken:
            return cls(0, 0.0, "", "")

        scored = sum(one.scored for one in taken.values())
        total = sum(one.scored * one.mean for one in taken.values())
        low = min(taken, key=lambda name: taken[name].mean)
        high = max(taken, key=lambda name: taken[name].mean)

        return cls(scored, total / scored, low, high)

    @override
    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> Self:
        """Rebuild a spread from what `to_dict` produced.

        Raises:
            ValueError: If a key it needs is absent or unreadable.
        """
        return cls(
            read_entry(document, "scored", int),
            read_number(document, "mean"),
            read_entry(document, "min_source", str),
            read_entry(document, "max_source", str),
        )


# ========================== #
#        Evaluations         #
# ========================== #


@dataclass(frozen=True, slots=True)
class SequenceEvaluation(SequenceResult[FrameEvaluation]):
    """What one sequence scored, over the pairs it was measured on.

    Attributes:
        source: The name the sequence has in its dataset.
        steps: What each pair scored, in the order they were measured.
        pairs: How many flows the sequence answered, which is one fewer than the frames
            it holds and is what every `scored` is read against.
        metrics: What each metric scored, by name, over the finite ones.

    Raises:
        ValueError: If there are no pairs, since a sequence that answered nothing has
            nothing to say and a result standing for it would count as covered.
    """

    pairs: int = field(init=False)
    metrics: Mapping[str, Measured] = field(init=False)

    @override
    def _aggregate(self) -> None:
        """Summarise the pairs, one metric at a time."""
        summarised = {
            metric: Measured.over(step.score(metric) for step in self.steps)
            for metric in METRICS
        }

        object.__setattr__(self, "pairs", len(self.steps))
        object.__setattr__(self, "metrics", MappingProxyType(summarised))

    def dropped(self, metric: str) -> int:
        """How many of the pairs this metric did not come back finite for."""
        return self.pairs - self.metrics[metric].scored

    @override
    def to_dict(self) -> dict[str, Any]:
        """Return the evaluation as plain data, ready to be written as JSON."""
        return {
            "source": self.source,
            "steps": [step.to_dict() for step in self.steps],
            "pairs": self.pairs,
            "metrics": {name: one.to_dict() for name, one in self.metrics.items()},
        }

    @override
    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> Self:
        """Rebuild one sequence's evaluation from its `source` and its `steps`.

        The summary is taken again rather than read back, so a document whose numbers
        were edited by hand cannot disagree with the pairs under them.

        Raises:
            ValueError: If either key is absent, or a pair cannot be read.
        """
        steps = read_entry(document, "steps", (list, tuple))

        return cls(
            read_entry(document, "source", str),
            tuple(FrameEvaluation.from_dict(step) for step in steps),
        )


@dataclass(frozen=True, slots=True)
class DatasetEvaluation(DatasetResult[SequenceEvaluation]):
    """What a dataset scored, over the sequences it covers.

    Attributes:
        source: The dataset root the run read, which is what tells two documents apart
            when someone comes to merge them.
        sequences: What each sequence scored, in the order they were summarised.
        pairs: The flows every sequence answered together.
        metrics: What each metric scored across them, with the ends and who reached
            them.

    Raises:
        ValueError: If there are no sequences, or if two are filed under one name, which
            would leave one out of every summary without saying so.
    """

    pairs: int = field(init=False)
    metrics: Mapping[str, Spread] = field(init=False)

    @override
    def _aggregate(self) -> None:
        """Summarise the sequences, one metric at a time."""
        summarised = {
            metric: Spread.across(
                {one.source: one.metrics[metric] for one in self.sequences}
            )
            for metric in METRICS
        }

        object.__setattr__(self, "pairs", sum(one.pairs for one in self.sequences))
        object.__setattr__(self, "metrics", MappingProxyType(summarised))

    def __str__(self) -> str:
        """The one axis the document exists for, shortened for reading.

        Reconstruction alone says little without what the pair scored against each
        other, so the gain over that floor is given beside it.
        """
        ssim = self.metrics["ssim"].mean
        gain = ssim - self.metrics["ssim_floor"].mean

        return f"SSIM {ssim:.4f} ({gain:+.4f})"

    def dropped(self, metric: str) -> int:
        """How many pairs this metric did not come back finite for.

        Summed over the dataset, this is how many duplicated frames and empty fields it
        holds: an exact reconstruction is the only way to reach one.
        """
        return self.pairs - self.metrics[metric].scored

    @override
    def to_dict(self) -> dict[str, Any]:
        """Return the evaluation as plain data, ready to be written as JSON."""
        return {
            "source": self.source,
            "sequences": [one.to_dict() for one in self.sequences],
            "pairs": self.pairs,
            "metrics": {name: one.to_dict() for name, one in self.metrics.items()},
        }

    @override
    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> Self:
        """Rebuild a dataset's evaluation from its `source` and its `sequences`.

        Raises:
            ValueError: If either key is absent, or a sequence cannot be read.
        """
        sequences = read_entry(document, "sequences", (list, tuple))

        return cls(
            read_entry(document, "source", str),
            tuple(SequenceEvaluation.from_dict(one) for one in sequences),
        )
