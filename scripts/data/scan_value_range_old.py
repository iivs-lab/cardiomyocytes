"""Scan every phase sequence under a root and record its value range.

Reports `(min, max)` at three levels -- each frame, each sequence, and the whole
dataset -- so a later `FrameNormalizer` in its `injected` mode has a span to
scale from. A sequence range is not derivable from per-frame percentiles, and a
dataset range is not derivable from sequence ones, so all three come from the
same single pass rather than being reconstructed afterwards.

The filter is a config group. Left in place, frames are filtered exactly as the
pipeline would filter them, so the recorded range matches what a consumer sees;
removed, the raw frames are scanned instead:

    uv run scripts/data/scan_value_range.py data.root=<dir>
    uv run scripts/data/scan_value_range.py data.root=<dir> \\
        ~data/transforms/filtering@data.filtering

Only Koala phase `.bin` folders are discovered today. `find_sources` is the one
place that decides, so another modality is a different finder and nothing else.

Every frame is read once, which is the cost of the whole run: budget roughly a
few milliseconds per frame for the read, and add filtering on top when a kernel
is set (`device=cuda` pays for itself there).
"""

from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import hydra
from dotenv import load_dotenv
from hydra.utils import instantiate
from iivs.dhm.data.phase import search_phase_bin_folders
from kaparoo.filesystem import ensure_dir_exists

from iivs_cardio.data.sequence import FrameSequence

if TYPE_CHECKING:
    from collections.abc import Sequence

    from iivs.dhm.data.phase import PhaseBinFolder
    from omegaconf import DictConfig

    from iivs_cardio.data.transforms.filtering import KernelParams

load_dotenv()

CONFIG_PATH = os.environ["CONFIGS_ROOT"]
CONFIG_NAME = "data/value_range/config"

FORMATS = ("csv", "json")
FIELDS = ("level", "sequence", "index", "min", "max")

# A `_target_` is a dotted path the config chooses and `instantiate` imports and
# calls, so hydra 1.4 wants the callsite to say what it means to build. This one
# builds kernel recipes and nothing else.
KERNEL_TARGETS = ("iivs_cardio.data.transforms.filtering.kernel.*",)

logger = logging.getLogger(__name__)


class Range(NamedTuple):
    """One `(min, max)` and what it covers."""

    level: str
    sequence: str
    index: int | None
    minimum: float | None
    maximum: float | None

    def row(self) -> dict[str, object]:
        """Flatten to the CSV field order, with None as an empty cell."""
        values = (self.level, self.sequence, self.index, self.minimum, self.maximum)
        return {
            field: "" if value is None else value
            for field, value in zip(FIELDS, values, strict=True)
        }


def find_sources(root: Path) -> list[PhaseBinFolder]:
    """List the sequences to scan under `root`, in a stable order."""
    return search_phase_bin_folders(root)


def scan_sequence(frames: FrameSequence, name: str) -> list[Range]:
    """Range every frame of `frames`, then the sequence itself.

    Reads each frame once and folds the per-frame ranges up, rather than asking
    the sequence for its own range and paying for a second pass. A frame holding
    no finite value is recorded with empty bounds and left out of the fold, so
    one bad frame is reported rather than ending the scan.
    """
    ranges = []
    for index in range(len(frames)):
        try:
            minimum, maximum = frames.value_range(index)
        except ValueError:
            logger.warning("no finite value in %s frame %d", name, index)
            ranges.append(Range("frame", name, index, None, None))
        else:
            ranges.append(Range("frame", name, index, minimum, maximum))

    finite = [r for r in ranges if r.minimum is not None and r.maximum is not None]
    if not finite:
        logger.warning("no finite value anywhere in %s", name)
        return [*ranges, Range("sequence", name, None, None, None)]

    lowest = min(r.minimum for r in finite if r.minimum is not None)
    highest = max(r.maximum for r in finite if r.maximum is not None)
    return [*ranges, Range("sequence", name, None, lowest, highest)]


def write_csv(path: Path, ranges: Sequence[Range]) -> None:
    """Write `ranges` as one long-format row per range."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(entry.row() for entry in ranges)


def write_json(path: Path, ranges: Sequence[Range]) -> None:
    """Write `ranges` with each frame under its sequence, and the dataset above.

    Nesting is what JSON buys over the flat rows the CSV carries, so an entry
    holds only the fields its level has: no empty index on a sequence, no empty
    sequence on the dataset. A missing bound stays `null` rather than blank.
    """

    def bounds(entry: Range) -> dict[str, float | None]:
        return {"min": entry.minimum, "max": entry.maximum}

    frames: dict[str, list[Range]] = {}
    for entry in ranges:
        if entry.level == "frame":
            frames.setdefault(entry.sequence, []).append(entry)

    document = {
        "dataset": next(bounds(e) for e in ranges if e.level == "dataset"),
        "sequences": [
            {
                "sequence": e.sequence,
                **bounds(e),
                "frames": [
                    {"index": f.index, **bounds(f)} for f in frames.get(e.sequence, ())
                ],
            }
            for e in ranges
            if e.level == "sequence"
        ],
    }
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")


def resolve_format(path: Path, requested: str | None) -> str:
    """Pick the output format, from `requested` or `path`'s suffix.

    Raises:
        ValueError: If the format is neither `csv` nor `json`.
    """
    chosen = requested or path.suffix.lstrip(".").lower()
    if chosen not in FORMATS:
        allowed = ", ".join(FORMATS)
        msg = f"unsupported output format {chosen!r}: expected one of {allowed}"
        raise ValueError(msg)
    return chosen


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    root = ensure_dir_exists(cfg.data.root)

    sources = find_sources(root)
    if not sources:
        msg = f"no phase .bin sequences found in {root}"
        raise SystemExit(msg)

    params: KernelParams | None = instantiate(
        cfg.data.get("filtering"), _target_whitelist_=KERNEL_TARGETS
    )
    logger.info(
        "Scanning %d sequence(s) under %s, %s, on %s.",
        len(sources),
        root,
        "raw" if params is None else f"filtered by {type(params).__name__}",
        cfg.device,
    )

    ranges: list[Range] = []
    for source in sources:
        name = str(source.root.relative_to(root))
        frames = FrameSequence.from_params(source, params, device=cfg.device)
        ranges.extend(scan_sequence(frames, name))
        logger.info("  %s: %d frame(s)", name, len(frames))

    sequences = [r for r in ranges if r.level == "sequence" and r.minimum is not None]
    if not sequences:
        msg = f"no finite value in any sequence under {root}"
        raise SystemExit(msg)

    lowest = min(r.minimum for r in sequences if r.minimum is not None)
    highest = max(r.maximum for r in sequences if r.maximum is not None)
    ranges.append(Range("dataset", "", None, lowest, highest))
    logger.info("Dataset range: [%g, %g]", lowest, highest)

    path = Path(cfg.output.path)
    writer = (
        write_csv if resolve_format(path, cfg.output.format) == "csv" else write_json
    )
    writer(path, ranges)
    logger.info("Wrote %d row(s) to %s", len(ranges), path.resolve())


if __name__ == "__main__":
    main()
