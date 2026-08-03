from __future__ import annotations

import json
import os
from contextlib import ExitStack
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

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

from iivs_cardio.common.pipeline import Slot, steps
from iivs_cardio.common.range import finite_range
from iivs_cardio.data.phase import phase_field_writer
from iivs_cardio.data.transforms.filtering import FilteredSequence
from scripts._compute import ComputeConfig, pin_threads, plan_devices, report_insights
from scripts._hydra import apply_schema, is_multirun, output_directory
from scripts._range import DatasetRange, FrameRange, SequenceRange, as_dict
from scripts.data._filtering import build_filter_kernel, describe_filter_kernel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from iivs.dhm.data.phase import PhaseFileFolder
    from kaparoo.filesystem.types import StrPath
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


def _open_writer(
    sequence: PhaseFilteredSequence,
    dest: StrPath,
    target_config: TargetConfig,
) -> FieldWriter[Tensor]:
    header = sequence.origin.header

    return phase_field_writer(
        dest,
        pixel_size=header.pixel_size,
        height_scale=header.height_scale,
        # None means each field keeps its stored unit, which is the header's.
        unit=unwrap_or_default(sequence.origin.target_unit, header.unit),
        overwrite=target_config.overwrite,
    )


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

    with ExitStack() as stack:
        writer = (
            stack.enter_context(
                _open_writer(
                    sequence, Path(target_config.root, source, subpath), target_config
                )
            )
            if target_config is not None and target_config.save_frames
            else None
        )

        # One read per step, ranged in hand: reading the frame again to range it
        # would re-run the kernel, which is the expensive half of a filtered read
        # -- a second traversal measured +94% on a median (2, 2, 2).
        for slot in steps(sequence):
            frame, path = slot.require()

            found = finite_range(frame)
            if found is None:
                msg = f"no finite value in {source}/{path.name}"
                raise ValueError(msg)

            frames.append(FrameRange(path.name, *found))
            if writer is not None:
                writer.write(Slot(slot.index, frame))

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
    compute_config: ComputeConfig,
    source_config: SourceConfig,
    target_config: TargetConfig,
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
    document = {
        "source": {
            "phase_unit": source_config.phase_unit,
            "frame_step": source_config.frame_step,
        },
        "filter": describe_filter_kernel(filter_config),
        "dataset": as_dict(dataset),
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

    sequences = build_sequences(compute_config, source_config, filter_config)
    scanned = scan_sequences(sequences, compute_config, source_config, target_config)

    if target_config.save_ranges:
        dataset_range = DatasetRange(tuple(scanned))
        save_dataset_range(dataset_range, source_config, target_config, filter_config)


if __name__ == "__main__":
    main()
