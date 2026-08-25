from __future__ import annotations

__all__ = (
    "Normalization",
    "NormalizeConfig",
    "build_normalization",
    "log_normalize_config",
)

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kaparoo.filesystem import ensure_file_extension

from iivs_cardio.common.logging import log_indented
from iivs_cardio.common.pipeline import JSON_EXT
from iivs_cardio.data.pipeline import DatasetRange
from iivs_cardio.data.transforms.normalization import (
    FrameNormalizer,
    NormalizerConfig,
    RangeLevel,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from logging import Logger

    import torch


@dataclass
class NormalizeConfig:
    """Where the range a run scales its frames from comes from.

    `"sequence"` and `"dataset"` read a layer of the range document a measuring
    run left, so they need `range_file` and refuse a `source` of their own.
    `"given"` is the other way round: it carries the span, which is what makes
    two runs under different filters comparable, since a measured range is a
    property of the filter that shaped it.

    Attributes:
        level: Which range a frame is scaled from.
        range_file: The document a measuring run wrote, given `.json` if it has
            no extension. Defaults to `None`, which only `"given"` can do.
        source: The span `"given"` scales from, as `[min, max]`. Defaults to
            `None`, which is what a measured level takes.
        target: The span the output covers, as `[min, max]`. Defaults to `None`,
            which takes the span of the dtype the estimator reads.
    """

    level: RangeLevel = "dataset"
    range_file: str | None = None
    source: list[float] | None = None
    target: list[float] | None = None


@dataclass(frozen=True, slots=True)
class Normalization:
    """The scaling a run applies, before it knows which sequences it will run.

    Settled from the configuration alone, since a measured range is read off a
    document rather than off the dataset: a run can say what it will scale by
    before it has searched for anything to scale, and a mistyped document is
    refused before the search costs anything.

    Attributes:
        described: What went into the scaling, for the settings a later run
            compares against. The ranges themselves and not the document they
            came from: the same path may hold a document that was written
            again, and a run that scaled by other numbers must not read as this
            one.
        shared: The one scaling that covers every sequence, which is what a
            given span and a dataset range both come to. Defaults to `None`,
            which is where each sequence has its own.
        per_sequence: The scaling for each sequence the document covered, by
            name. Defaults to empty, which is where one scaling covers them all.
    """

    described: Mapping[str, Any]
    shared: FrameNormalizer | None = None
    per_sequence: Mapping[str, FrameNormalizer] = field(default_factory=dict)

    def normalizers(self, names: Iterable[str]) -> dict[str, FrameNormalizer]:
        """Return the scaling for each of `names` this run is able to scale.

        A name with no range of its own is left out rather than refused here.
        Which sequences the run was actually given is settled later, and one
        the document never covered may well be one nobody asked for.
        """
        if self.shared is not None:
            return dict.fromkeys(names, self.shared)

        return {
            name: self.per_sequence[name] for name in names if name in self.per_sequence
        }


def _span(values: Sequence[float], key: str) -> tuple[float, float]:
    """Read a `[min, max]` pair written as a list, refusing any other shape."""
    if len(values) != 2:
        msg = f"`{key}` takes two numbers, [min, max]: got {len(values)}"
        raise ValueError(msg)

    return float(values[0]), float(values[1])


def _read_dataset_range(config: NormalizeConfig) -> DatasetRange:
    """Read the range document a measuring run left, refusing what is not one.

    Raises:
        ValueError: If no document was named, if the name points at nothing, or
            if what it points at is not a document a measuring run wrote. Each
            names the setting rather than the file, since that is where the fix
            goes.
    """
    if config.range_file is None:
        fix = "set `normalize.range_file`"
        msg = f"level {config.level!r} scales from a measured range: {fix}"
        raise ValueError(msg)

    path = Path(ensure_file_extension(config.range_file, JSON_EXT, add=True))
    if not path.is_file():
        msg = f"no such `normalize.range_file`: {path}"
        raise ValueError(msg)

    with path.open(encoding="utf-8") as file:
        document = json.load(file)

    if not isinstance(document, dict) or "dataset" not in document:
        fix = "point it at a document a measuring run wrote"
        msg = f"`normalize.range_file` holds no dataset range: {fix}"
        raise ValueError(msg)

    return DatasetRange.from_dict(document["dataset"])


def build_normalization(config: NormalizeConfig, dtype: torch.dtype) -> Normalization:
    """Settle the scaling a run applies, reading a document where it needs one.

    Args:
        config: Where the range comes from.
        dtype: The dtype the frames are scaled onto, which is the one the
            estimator downstream reads: `EstimatorConfig.FRAME_DTYPE`.

    Returns:
        The scaling, ready to be asked for the sequences the run turns out to
        hold.

    Raises:
        ValueError: If the level and the settings beside it disagree, if a
            measured level names no readable document, or if a span is empty.
    """
    given = None if config.source is None else _span(config.source, "normalize.source")
    target = None if config.target is None else _span(config.target, "normalize.target")

    settings = NormalizerConfig(config.level, given, target)

    def covering_all(built: FrameNormalizer) -> Normalization:
        described = {
            "level": config.level,
            "range": list(built.source),
            "target": list(built.target),
        }

        return Normalization(described, shared=built)

    if given is not None:
        return covering_all(settings.build(dtype))

    dataset = _read_dataset_range(config)

    if config.level == "dataset":
        return covering_all(
            settings.build(dtype, (dataset.min_value, dataset.max_value))
        )

    spans = {
        each.source: (each.min_value, each.max_value) for each in dataset.sequences
    }
    measured = {name: settings.build(dtype, span) for name, span in spans.items()}

    described = {
        "level": config.level,
        "ranges": {name: list(spans[name]) for name in sorted(spans)},
        "target": list(next(iter(measured.values())).target),
    }

    return Normalization(described, per_sequence=measured)


def log_normalize_config(normalization: Normalization, logger: Logger) -> None:
    """Log the range a run scales from, and what the frames come out as."""
    described = normalization.described

    log_indented(logger, "normalize: by the %s range", described["level"], depth=0)

    if (span := described.get("range")) is not None:
        log_indented(logger, "scaling from [%.4g, %.4g]", *span)
    else:
        counted = len(normalization.per_sequence)
        log_indented(logger, "scaling each of %d sequences from its own", counted)

    log_indented(logger, "onto [%.4g, %.4g]", *described["target"])
