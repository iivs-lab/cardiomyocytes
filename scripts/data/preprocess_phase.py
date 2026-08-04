from __future__ import annotations

import os
from dataclasses import dataclass
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

from iivs_cardio.common.pipeline import SequenceStage
from iivs_cardio.data.phase import phase_frame_writer
from iivs_cardio.data.transforms.filtering import FilteredSequence
from scripts._compute import ComputeConfig, pin_threads, plan_devices
from scripts._hydra import apply_schema, is_multirun
from scripts.data._filtering import build_filter_kernel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from iivs.dhm.data.phase import PhaseFileFolder
    from omegaconf import DictConfig
    from torch import Tensor

    from iivs_cardio.common.device import Device
    from iivs_cardio.common.writer import KoalaFrameWriter


load_dotenv()

CONFIG_PATH = os.environ["CONFIGS_ROOT"]
CONFIG_NAME = "data/preprocess_phase/config"

type PhaseFilteredSequence = FilteredSequence[PhaseFileFolder, Path]
type PreprocessShared = tuple[tuple[Device, ...], PhaseStageFactory]


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


@dataclass(frozen=True, slots=True)
class FrameDestination:
    source_root: str
    target_root: str
    subpath: str
    overwrite: bool = False

    def hook_for(self, origin: PhaseFileFolder) -> KoalaFrameWriter[Tensor]:
        header = origin.header
        name = stringify_path(origin.root, after=self.source_root, before=self.subpath)

        return phase_frame_writer(
            Path(self.target_root, name, self.subpath),
            pixel_size=header.pixel_size,
            height_scale=header.height_scale,
            unit=unwrap_or_default(origin.target_unit, header.unit),
            overwrite=self.overwrite,
        )


class PhaseStageFactory:
    def __init__(
        self,
        sequences: Sequence[PhaseFilteredSequence],
        *hooks: FrameDestination,
    ) -> None:
        self._sequences = sequences
        self._hooks = hooks

    def __len__(self) -> int:
        return len(self._sequences)

    def stage_for(self, index: int, device: Device) -> SequenceStage[Tensor, Path]:
        sequence = self._sequences[index]
        sequence.device = device

        stage = SequenceStage(sequence)
        stage.register_hooks(*(hook.hook_for(sequence.origin) for hook in self._hooks))

        return stage


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
    filter_config: DictConfig | None = None,
) -> list[PhaseFilteredSequence]:
    kernel = build_filter_kernel(filter_config)
    sources = search_sources(source_config)
    frame_step = source_config.frame_step

    def build_sequence(source: PhaseFileFolder) -> PhaseFilteredSequence:
        return FilteredSequence(source, kernel, step=frame_step)

    return [build_sequence(source) for source in sources]


def build_phase_stages(
    source_config: SourceConfig,
    target_config: TargetConfig | None = None,
    filter_config: DictConfig | None = None,
) -> PhaseStageFactory:
    hooks: list[FrameDestination] = []
    if target_config is not None and target_config.save_frames:
        hooks.append(
            FrameDestination(
                source_config.root,
                target_config.root,
                unwrap_or_default(source_config.subpath, PHASE_FLOAT_BIN),
                overwrite=target_config.overwrite,
            )
        )
    return PhaseStageFactory(build_sequences(source_config, filter_config), *hooks)


def preprocess_sequence(worker_id: int, shared: PreprocessShared, index: int) -> None:
    devices, stages = shared

    device = devices[worker_id]
    device.activate()
    pin_threads(len(devices))

    stages.stage_for(index, device).run()


def preprocess_sequences(
    stages: PhaseStageFactory,
    compute_config: ComputeConfig,
) -> None:
    pbar_enabled = compute_config.progress_bar
    pbar_options = {"desc": "preprocessing", "unit": "seq"}

    devices = plan_devices(compute_config)[: len(stages)]
    shared: PreprocessShared = (devices, stages)

    if (num_workers := len(devices)) == 1:
        indices = tqdm(range(len(stages)), disable=not pbar_enabled, **pbar_options)
        for index in indices:
            preprocess_sequence(0, shared, index)
        return

    with WorkerPool(
        n_jobs=num_workers,
        shared_objects=shared,
        pass_worker_id=True,
        enable_insights=compute_config.insights,
    ) as pool:
        pool.map(
            preprocess_sequence,
            range(len(stages)),
            chunk_size=1,
            worker_lifespan=compute_config.worker_lifespan,
            progress_bar=pbar_enabled,
            progress_bar_options=pbar_options,
        )


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    compute_config = apply_schema(ComputeConfig, cfg.compute)
    source_config = apply_schema(SourceConfig, cfg.source)
    target_config = apply_schema(TargetConfig, cfg.target)
    filter_config: DictConfig | None = cfg.filter

    if target_config.save_frames and is_multirun():
        msg = "cannot write frames in a sweep: run the winning config alone instead"
        raise ValueError(msg)

    stages = build_phase_stages(source_config, target_config, filter_config)
    preprocess_sequences(stages, compute_config)


if __name__ == "__main__":
    main()
