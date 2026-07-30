from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from multiprocessing import Manager
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import hydra
from dotenv import load_dotenv
from iivs.dhm.data.koala import PHASE_FLOAT_BIN
from iivs.dhm.data.phase import resolve_phase_unit, search_phase_bin_folders
from kaparoo.filesystem import StagedFile, stringify_path
from kaparoo.filesystem.search import select
from kaparoo.utils.optional import unwrap_or_default
from omegaconf import MISSING

from iivs_cardio.common.device import (
    resolve_device,
    resolve_devices,
    visible_cuda_devices,
)
from iivs_cardio.data.sequence import FrameSequence
from scripts._config import apply_schema
from scripts.data._filtering import build_filter_kernel, describe_filter_kernel

if TYPE_CHECKING:
    from collections.abc import Sequence
    from queue import Queue

    import torch
    from iivs.dhm.data.phase import PhaseFileFolder
    from omegaconf import DictConfig


load_dotenv()

CONFIG_PATH = os.environ["CONFIGS_ROOT"]
CONFIG_NAME = "data/phase_range/config"

TIMESTAMP = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")  # YYYYMMDD-HHMMSS
RANGE_FILE = f"phase_range_{TIMESTAMP}.json"
RANGE_VERSION = 1

# Claimed once per worker by `_adopt_device`; the parent never leaves the default.
_WORKER_DEVICE: str | torch.device = "cpu"


@dataclass
class ComputeConfig:
    device: str = "cpu"
    workers: int | None = 0
    gpu_ids: list[int] | None = field(default_factory=lambda: [0])


@dataclass
class SourceConfig:
    root: str = MISSING
    unit: str | None = None
    subpath: str | None = None
    include: list[str] | str | None = None
    exclude: list[str] | str | None = None
    frame_step: int = 1


@dataclass
class TargetConfig:
    root: str = MISSING
    save_ranges: bool = True
    save_frames: bool = False


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


def plan_devices(compute: ComputeConfig) -> tuple[torch.device, ...]:
    # One device per worker, and a single entry means this process does it all,
    # so "sequential / many processes / many GPUs" is one length check downstream.
    if resolve_device(compute.device).type != "cuda":
        workers = os.cpu_count() or 1 if compute.workers is None else compute.workers
        if workers < 0:
            msg = f"invalid worker count {workers}: expected 0 or more, or null"
            raise ValueError(msg)

        return resolve_devices(["cpu"] * max(workers, 1))

    if not compute.gpu_ids:
        devices = visible_cuda_devices()
        if not devices:
            msg = "no CUDA device is visible: set `compute=cpu`, or check the driver"
            raise ValueError(msg)

        return devices

    return resolve_devices(f"cuda:{index}" for index in compute.gpu_ids)


def search_sources(config: SourceConfig) -> list[PhaseFileFolder]:
    subpath = unwrap_or_default(config.subpath, PHASE_FLOAT_BIN)

    def folder_subpath(folder: PhaseFileFolder) -> str:
        return stringify_path(folder.root, after=config.root, before=subpath)

    sources: list[PhaseFileFolder] = select(
        search_phase_bin_folders(config.root, subpath=subpath),
        key=folder_subpath,
        include=config.include,
        exclude=config.exclude,
    )

    if not sources:
        msg = f"no time-lapse holds a {subpath!r} folder: {config.root}"
        raise ValueError(msg)

    if config.unit is not None:
        unit = resolve_phase_unit(config.unit)
        sources = [source.with_unit(unit) for source in sources]

    return sources


def list_sequences(
    source_config: SourceConfig,
    compute_config: ComputeConfig,
    filter_config: DictConfig | None = None,
) -> list[FrameSequence[Path]]:
    device = plan_devices(compute_config)[0]
    kernel = build_filter_kernel(filter_config)
    sources = search_sources(source_config)
    frame_step = source_config.frame_step

    def open_sequence(source: PhaseFileFolder) -> FrameSequence[Path]:
        return FrameSequence(source, kernel=kernel, step=frame_step, device=device)

    return [open_sequence(source) for source in sources]


def scan_sequence(sequence: FrameSequence[Path], config: SourceConfig) -> SequenceRange:
    subpath = unwrap_or_default(config.subpath, PHASE_FLOAT_BIN)
    frames = []

    for index in range(len(sequence)):
        name = sequence.get_meta(index).name
        frame_min, frame_max = sequence.value_range(index)  # ignores non-finite values
        frames.append(FrameRange(name, frame_min, frame_max))

    folder = sequence.get_meta(0).parent
    source = stringify_path(folder, after=config.root, before=subpath)

    return SequenceRange(source, tuple(frames))


def _adopt_device(devices: Queue[torch.device]) -> None:
    # `initargs` reaches every worker with the same value, so the queue is what
    # hands each a different device; it holds exactly one per worker.
    global _WORKER_DEVICE
    _WORKER_DEVICE = devices.get()


def _scan_on_worker(
    sequence: FrameSequence[Path], config: SourceConfig
) -> SequenceRange:
    sequence.device = _WORKER_DEVICE

    return scan_sequence(sequence, config)


def scan_sequences(
    sequences: Sequence[FrameSequence[Path]],
    source: SourceConfig,
    compute: ComputeConfig,
) -> list[SequenceRange]:
    devices = plan_devices(compute)
    if (max_workers := len(devices)) == 1:
        return [scan_sequence(sequence, source) for sequence in sequences]

    # A CUDA context costs seconds to build, so a worker claims its device once
    # and keeps it. `chunksize=1` then hands out the next sequence as each worker
    # frees up, which balances on real completion times rather than on a guess at
    # how long a sequence takes; `map` still returns in the order given.
    with Manager() as manager:
        queue: Queue[torch.device] = manager.Queue()
        for device in devices:
            queue.put(device)

        with ProcessPoolExecutor(
            max_workers, initializer=_adopt_device, initargs=(queue,)
        ) as pool:
            scan = partial(_scan_on_worker, config=source)
            return list(pool.map(scan, sequences, chunksize=1))


def save_dataset_range(
    dataset: DatasetRange,
    target: TargetConfig,
    source: SourceConfig,
    filtering: DictConfig | None = None,
) -> Path:
    document = {
        "version": RANGE_VERSION,
        "source": asdict(source),
        "filter": describe_filter_kernel(filtering),
        "dataset": asdict(dataset),
    }

    path = Path(target.root, RANGE_FILE)

    with StagedFile(path, overwrite=True, make_parents=True, encoding="utf-8") as file:
        file.write(json.dumps(document, indent=2))

    return path


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    compute_config = apply_schema(ComputeConfig, cfg.compute)
    source_config = apply_schema(SourceConfig, cfg.source)
    target_config = apply_schema(TargetConfig, cfg.target)

    if not (target_config.save_ranges or target_config.save_frames):
        msg = "nothing to do: set `target.save_ranges` or `target.save_frames`"
        raise SystemExit(msg)

    sequences = list_sequences(source_config, compute_config, cfg.filter)

    if target_config.save_ranges:
        scanned = scan_sequences(sequences, source_config, compute_config)
        dataset_range = DatasetRange(tuple(scanned))
        save_dataset_range(dataset_range, target_config, source_config, cfg.filter)

    if target_config.save_frames:
        pass


if __name__ == "__main__":
    main()
