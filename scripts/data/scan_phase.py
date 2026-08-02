from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import hydra
from dotenv import load_dotenv
from iivs.dhm.data.koala import PHASE_FLOAT_BIN
from iivs.dhm.data.phase import resolve_phase_unit, search_phase_bin_folders
from kaparoo.filesystem import StagedFile, ensure_file_extension, stringify_path
from kaparoo.filesystem.search import select
from kaparoo.utils.optional import unwrap_or_default
from mpire import WorkerPool
from omegaconf import MISSING
from tqdm import tqdm

from iivs_cardio.common.range import finite_range
from iivs_cardio.data.phase import save_phase_bin_folder
from iivs_cardio.data.transforms.filtering import FilteredSequence
from scripts._compute import ComputeConfig, pin_threads, plan_devices, report_insights
from scripts._hydra import apply_schema, is_multirun, output_directory
from scripts.data._filtering import build_filter_kernel, describe_filter_kernel

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from iivs.dhm.data.phase import PhaseFileFolder
    from omegaconf import DictConfig
    from torch import Tensor

    from iivs_cardio.common.device import Device


load_dotenv()

CONFIG_PATH = os.environ["CONFIGS_ROOT"]
CONFIG_NAME = "data/phase_scan/config"

type PhaseFilteredSequence = FilteredSequence[PhaseFileFolder, Path]


@dataclass
class SourceConfig:
    root: str = MISSING
    subpath: str | None = None
    include: list[str] | str | None = None
    exclude: list[str] | str | None = None
    phase_unit: str | None = None
    frame_step: int = 1


@dataclass
class TargetConfig:
    root: str = MISSING
    overwrite: bool = False
    save_frames: bool = False
    save_ranges: bool = True
    range_file: str = "phase_range"


class ValueRange(Protocol):
    @property
    def min_value(self) -> float: ...

    @property
    def max_value(self) -> float: ...


@dataclass(frozen=True, slots=True)
class CompositeRange(ValueRange, ABC):
    min_value: float = field(init=False)
    max_value: float = field(init=False)
    min_index: int = field(init=False)
    max_index: int = field(init=False)

    @property
    @abstractmethod
    def parts(self) -> Sequence[ValueRange]: ...

    def __post_init__(self) -> None:
        parts = self.parts
        if not parts:
            msg = f"value range is undefined: {type(self).__name__} holds nothing"
            raise ValueError(msg)

        indices = range(len(parts))
        min_index = min(indices, key=lambda i: parts[i].min_value)
        max_index = max(indices, key=lambda i: parts[i].max_value)

        object.__setattr__(self, "min_value", parts[min_index].min_value)
        object.__setattr__(self, "max_value", parts[max_index].max_value)
        object.__setattr__(self, "min_index", min_index)
        object.__setattr__(self, "max_index", max_index)


@dataclass(frozen=True, slots=True)
class FrameRange:
    source: str  # frame name relative to the sequence folder
    min_value: float
    max_value: float


@dataclass(frozen=True, slots=True)
class SequenceRange(CompositeRange):
    source: str  # folder path relative to the dataset root
    frames: tuple[FrameRange, ...]

    @property
    def parts(self) -> tuple[FrameRange, ...]:
        return self.frames


@dataclass(frozen=True, slots=True)
class DatasetRange(CompositeRange):
    sequences: tuple[SequenceRange, ...]

    @property
    def parts(self) -> tuple[SequenceRange, ...]:
        return self.sequences


def search_sources(config: SourceConfig) -> list[PhaseFileFolder]:
    subpath = unwrap_or_default(config.subpath, PHASE_FLOAT_BIN)

    folders = search_phase_bin_folders(config.root, subpath=subpath)
    if (num_folders := len(folders)) == 0:
        msg = f"no time-lapse holds a {subpath!r} folder: {config.root}"
        raise ValueError(msg)

    def folder_subpath(folder: PhaseFileFolder) -> str:
        return stringify_path(folder.root, after=config.root, before=subpath)

    sources: list[PhaseFileFolder] = select(
        folders,
        key=folder_subpath,
        include=config.include,
        exclude=config.exclude,
    )

    if not sources:
        msg = f"include/exclude left none of the {num_folders} sequences: {config.root}"
        raise ValueError(msg)

    if config.phase_unit is not None:
        unit = resolve_phase_unit(config.phase_unit)
        sources = [source.with_unit(unit) for source in sources]

    return sources


def build_sequences(
    source_config: SourceConfig,
    compute_config: ComputeConfig,
    filter_config: DictConfig | None = None,
) -> list[PhaseFilteredSequence]:
    device = plan_devices(compute_config)[0]
    kernel = build_filter_kernel(filter_config)
    sources = search_sources(source_config)
    frame_step = source_config.frame_step

    def build_sequence(source: PhaseFileFolder) -> PhaseFilteredSequence:
        return FilteredSequence(source, kernel, step=frame_step, device=device)

    return [build_sequence(source) for source in sources]


def scan_sequence(
    sequence: PhaseFilteredSequence,
    source_config: SourceConfig,
    target_config: TargetConfig | None = None,
) -> SequenceRange:
    subpath = unwrap_or_default(source_config.subpath, PHASE_FLOAT_BIN)
    source = stringify_path(
        sequence.get_meta(0).parent, after=source_config.root, before=subpath
    )
    frames: list[FrameRange] = []

    def walk() -> Iterator[Tensor]:
        # One read per frame, ranged in hand: `value_range` would read it again and
        # so re-run the kernel, which is the expensive half of a filtered read --
        # a second traversal measured +94% on a median (2, 2, 2).
        for index in range(len(sequence)):
            frame = sequence.get_item(index)
            name = sequence.get_meta(index).name

            found = finite_range(frame)
            if found is None:
                msg = f"no finite value in {source}/{name}"
                raise ValueError(msg)

            frames.append(FrameRange(name, *found))
            yield frame

    if target_config is not None and target_config.save_frames:
        header = sequence.origin.header
        save_phase_bin_folder(
            Path(target_config.root, source, subpath),
            (frame.cpu().numpy() for frame in walk()),
            pixel_size=header.pixel_size,
            height_scale=header.height_scale,
            # None means each file keeps its stored unit, which is the header's.
            unit=unwrap_or_default(sequence.origin.target_unit, header.unit),
            overwrite=target_config.overwrite,
        )
    else:
        for _ in walk():
            pass

    return SequenceRange(source, tuple(frames))


def _scan_on_worker(
    worker_id: int,
    devices: tuple[Device, ...],
    sequence: PhaseFilteredSequence,
    source_config: SourceConfig,
    target_config: TargetConfig,
) -> SequenceRange:
    device = devices[worker_id]
    device.activate()
    pin_threads(len(devices))
    sequence.device = device

    return scan_sequence(sequence, source_config, target_config)


def scan_sequences(
    sequences: Sequence[PhaseFilteredSequence],
    source_config: SourceConfig,
    target_config: TargetConfig,
    compute_config: ComputeConfig,
) -> list[SequenceRange]:
    pbar_enabled = compute_config.progress_bar
    pbar_options = {"desc": "scanning", "unit": "seq"}

    devices = plan_devices(compute_config)[: len(sequences)]

    if (workers := len(devices)) == 1:
        pbar = tqdm(sequences, disable=not pbar_enabled, **pbar_options)
        return [scan_sequence(item, source_config, target_config) for item in pbar]

    with WorkerPool(
        n_jobs=workers,
        shared_objects=devices,
        pass_worker_id=True,
        enable_insights=compute_config.insights,
    ) as pool:
        scan = partial(
            _scan_on_worker, source_config=source_config, target_config=target_config
        )

        # Each task takes one sequence, wrapped: `mpire` spreads any iterable
        # task argument across the parameters, and a sequence is one.
        scanned: list[SequenceRange] = pool.map(
            scan,
            [(sequence,) for sequence in sequences],
            chunk_size=1,
            worker_lifespan=compute_config.worker_lifespan,
            progress_bar=pbar_enabled,
            progress_bar_options=pbar_options,
        )

        if compute_config.insights:
            report_insights(pool.get_insights())

    return scanned


def save_dataset_range(
    dataset: DatasetRange,
    source_config: SourceConfig,
    target_config: TargetConfig,
    filter_config: DictConfig | None = None,
) -> Path:
    def _source_first(items: list[tuple[str, Any]]) -> dict[str, Any]:
        return dict(sorted(items, key=lambda item: item[0] != "source"))

    document = {
        "source": {
            "phase_unit": source_config.phase_unit,
            "frame_step": source_config.frame_step,
        },
        "filter": describe_filter_kernel(filter_config),
        "dataset": asdict(dataset, dict_factory=_source_first),
    }

    path = Path(output_directory(), target_config.range_file)
    path = ensure_file_extension(path, ".json", add=True)

    with StagedFile(
        path,
        overwrite=target_config.overwrite,
        make_parents=True,
        encoding="utf-8",
    ) as file:
        file.write(json.dumps(document, indent=2))

    return path


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    compute_config = apply_schema(ComputeConfig, cfg.compute)
    source_config = apply_schema(SourceConfig, cfg.source)
    target_config = apply_schema(TargetConfig, cfg.target)
    filter_config: DictConfig | None = cfg.filter

    if not (target_config.save_ranges or target_config.save_frames):
        msg = "nothing to do: set `target.save_ranges` or `target.save_frames`"
        raise ValueError(msg)

    if target_config.save_frames and is_multirun():
        msg = "cannot write frames in a sweep: run the winning config alone instead"
        raise ValueError(msg)

    sequences = build_sequences(source_config, compute_config, filter_config)
    scanned = scan_sequences(sequences, source_config, target_config, compute_config)

    if target_config.save_ranges:
        dataset_range = DatasetRange(tuple(scanned))
        save_dataset_range(dataset_range, source_config, target_config, filter_config)


if __name__ == "__main__":
    main()
