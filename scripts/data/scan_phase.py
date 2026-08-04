from __future__ import annotations

import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
from dotenv import load_dotenv
from iivs.dhm.data.koala import PHASE_FLOAT_BIN
from iivs.dhm.data.phase import resolve_phase_unit, search_phase_bin_folders
from kaparoo.filesystem import stringify_path
from kaparoo.filesystem.search import select
from kaparoo.utils.optional import unwrap_or_default
from mpire import WorkerPool
from omegaconf import MISSING
from tqdm import tqdm

from iivs_cardio.common.pipeline import Steps
from iivs_cardio.data.phase import phase_field_writer
from iivs_cardio.data.transforms.filtering import FilteredSequence
from scripts._compute import ComputeConfig, pin_threads, plan_devices
from scripts._hydra import apply_schema, is_multirun, output_directory
from scripts.data._filtering import build_filter_kernel, describe_filter_kernel
from scripts.data._range import DatasetRangeCollector

if TYPE_CHECKING:
    from collections.abc import Sequence

    from iivs.dhm.data.phase import PhaseFileFolder
    from omegaconf import DictConfig
    from torch import Tensor

    from iivs_cardio.common.device import Device
    from iivs_cardio.common.writer import FieldWriter


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


@dataclass(frozen=True, slots=True)
class DatasetFieldWriter:
    """Where a run writes its filtered fields, and the writer for each sequence.

    Holds what every sequence's writer shares -- the tree to fill, the layout
    inside a sequence, and whether an existing folder may go -- so a worker is
    handed the run's decision instead of the config it was read from. Frozen and
    made of strings, so a copy crosses to a worker for the cost of the strings.

    What differs per sequence is the destination and the calibration, and both
    follow from the sequence: the first from where it sat under the source root,
    the second from the folder it was read out of.
    """

    root: str
    subpath: str
    overwrite: bool = False

    def writer_for(
        self, source: str, origin: PhaseFileFolder
    ) -> FieldWriter[tuple[Tensor, Path]]:
        """The writer for the sequence at `source`, read out of `origin`."""
        return phase_field_writer(
            Path(self.root, source, self.subpath), origin, overwrite=self.overwrite
        )


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
    compute_config: ComputeConfig,
    source_config: SourceConfig,
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
    ranges: DatasetRangeCollector,
    writers: DatasetFieldWriter | None = None,
) -> None:
    sequence_root = sequence.get_meta(0).parent
    subpath = unwrap_or_default(source_config.subpath, PHASE_FLOAT_BIN)
    source = stringify_path(sequence_root, after=source_config.root, before=subpath)

    hooks = [ranges.collector_for(source)]
    if writers is not None:
        hooks.append(writers.writer_for(source, sequence.origin))

    Steps(sequence).attach(*hooks).run()


def _scan_on_worker(
    worker_id: int,
    devices: tuple[Device, ...],
    sequence: PhaseFilteredSequence,
    source_config: SourceConfig,
    ranges: DatasetRangeCollector,
    writers: DatasetFieldWriter | None,
) -> DatasetRangeCollector:
    device = devices[worker_id]
    device.activate()
    pin_threads(len(devices))
    sequence.device = device

    # Not the `ranges` that arrived: a worker keeps it between tasks, so
    # gathering into it would carry every earlier task's sequences home again.
    gathered = ranges.fresh()
    scan_sequence(sequence, source_config, gathered, writers)

    return gathered


def scan_sequences(
    sequences: Sequence[PhaseFilteredSequence],
    compute_config: ComputeConfig,
    source_config: SourceConfig,
    ranges: DatasetRangeCollector,
    writers: DatasetFieldWriter | None = None,
) -> None:
    pbar_enabled = compute_config.progress_bar
    pbar_options = {"desc": "scanning", "unit": "seq"}

    devices = plan_devices(compute_config)[: len(sequences)]

    if (num_workers := len(devices)) == 1:
        for item in tqdm(sequences, disable=not pbar_enabled, **pbar_options):
            scan_sequence(item, source_config, ranges, writers)
        return

    with WorkerPool(
        n_jobs=num_workers,
        shared_objects=devices,
        pass_worker_id=True,
        enable_insights=compute_config.insights,
    ) as pool:
        scan = partial(
            _scan_on_worker,
            source_config=source_config,
            ranges=ranges,
            writers=writers,
        )

        for gathered in pool.imap(
            scan,
            [(sequence,) for sequence in sequences],
            chunk_size=1,
            worker_lifespan=compute_config.worker_lifespan,
            progress_bar=pbar_enabled,
            progress_bar_options=pbar_options,
        ):
            ranges.merge(gathered)


def range_provenance(
    source_config: SourceConfig, filter_config: DictConfig | None = None
) -> dict[str, object]:
    # What a reader of the numbers cannot get from the numbers. The rest of the
    # run is in `.hydra/` beside the document.
    return {
        "source": {
            "phase_unit": source_config.phase_unit,
            "frame_step": source_config.frame_step,
        },
        "filter": describe_filter_kernel(filter_config),
    }


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

    ranges = DatasetRangeCollector(
        Path(output_directory(), target_config.range_file),
        provenance=range_provenance(source_config, filter_config),
        overwrite=target_config.overwrite,
    )
    writers = (
        DatasetFieldWriter(
            target_config.root,
            unwrap_or_default(source_config.subpath, PHASE_FLOAT_BIN),
            overwrite=target_config.overwrite,
        )
        if target_config.save_frames
        else None
    )

    sequences = build_sequences(compute_config, source_config, filter_config)
    scan_sequences(sequences, compute_config, source_config, ranges, writers)

    if target_config.save_ranges:
        ranges.save()


if __name__ == "__main__":
    main()
