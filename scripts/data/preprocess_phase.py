from __future__ import annotations

import os
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Self

import hydra
from dotenv import load_dotenv
from iivs.dhm.data.koala import PHASE_FLOAT_BIN
from iivs.dhm.data.phase import resolve_phase_unit, search_phase_bin_folders
from kaparoo.filesystem import stringify_path
from kaparoo.filesystem.search import select
from kaparoo.utils.optional import unwrap_or_default
from omegaconf import MISSING

from iivs_cardio.common.pipeline import SequenceStage, SideBranch
from iivs_cardio.data.phase import phase_frame_writer
from iivs_cardio.data.transforms.filtering import FilteredSequence
from scripts._compute import ComputeConfig, run_all
from scripts._hydra import apply_schema, is_multirun, output_directory
from scripts.data._filtering import build_filter_kernel, describe_filter_kernel
from scripts.data._range import RangeDocument
from scripts.data._source import sequence_name

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from iivs.dhm.data.phase import PhaseFileFolder
    from kaparoo.filesystem.types import StrPath
    from omegaconf import DictConfig
    from torch import Tensor

    from iivs_cardio.common.device import Device
    from iivs_cardio.common.writer import KoalaFrameWriter


load_dotenv()

CONFIG_PATH = os.environ["CONFIGS_ROOT"]
CONFIG_NAME = "data/preprocess_phase/config"

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
class FrameDestination:
    target_root: str
    subpath: str
    overwrite: bool = False

    def hook_for(self, name: str, origin: PhaseFileFolder) -> KoalaFrameWriter[Tensor]:
        header = origin.header

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
        source_root: str,
        subpath: str,
        *branches: SideBranch[PhaseFileFolder, Tensor, Path],
    ) -> None:
        self._sequences = sequences
        self._source_root = source_root
        self._subpath = subpath
        self._branches = branches

    def __len__(self) -> int:
        return len(self._sequences)

    def stage_for(self, index: int, device: Device) -> SequenceStage[Tensor, Path]:
        sequence = self._sequences[index]
        sequence.device = device

        origin = sequence.origin
        name = sequence_name(origin, self._source_root, self._subpath)

        stage = SequenceStage(sequence)
        stage.register_hooks(
            *(branch.hook_for(name, origin) for branch in self._branches)
        )

        return stage

    def run_one(self, index: int, device: Device) -> None:
        self.stage_for(index, device).run()

    @contextmanager
    def running(self) -> Iterator[Self]:
        with ExitStack() as stack:
            for branch in self._branches:
                if isinstance(branch, AbstractContextManager):
                    stack.enter_context(branch)

            yield self


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


def build_phase_stages(
    source_config: SourceConfig,
    target_config: TargetConfig | None = None,
    filter_config: DictConfig | None = None,
    *,
    output_root: StrPath | None = None,
) -> PhaseStageFactory:
    subpath = unwrap_or_default(source_config.subpath, PHASE_FLOAT_BIN)
    branches: list[SideBranch[PhaseFileFolder, Tensor, Path]] = []

    if target_config is not None and target_config.save_frames:
        branches.append(
            FrameDestination(
                target_config.root, subpath, overwrite=target_config.overwrite
            )
        )

    if target_config is not None and target_config.save_ranges:
        root = unwrap_or_default(output_root, target_config.root)
        branches.append(
            RangeDocument(
                Path(root, target_config.range_file),
                provenance=range_provenance(source_config, filter_config),
                overwrite=target_config.overwrite,
            )
        )

    return PhaseStageFactory(
        build_sequences(source_config, filter_config),
        source_config.root,
        subpath,
        *branches,
    )


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

    stages = build_phase_stages(
        source_config, target_config, filter_config, output_root=output_directory()
    )
    run_all(stages, compute_config, desc="preprocessing", unit="seq")


if __name__ == "__main__":
    main()
