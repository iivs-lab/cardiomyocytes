from __future__ import annotations

__all__ = ("apply_schema", "output_directory")

from typing import TYPE_CHECKING, cast

from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

if TYPE_CHECKING:
    from omegaconf import DictConfig


def apply_schema[T](schema: type[T], node: DictConfig) -> T:
    merged = OmegaConf.merge(OmegaConf.structured(schema), node)
    return cast("T", OmegaConf.to_object(merged))


def output_directory() -> str:
    """Hydra's own directory for this job, which a run writes its outputs into.

    `hydra.run.dir` for a single run, and `hydra.sweep.dir/<subdir>` for each job
    of a `--multirun`. A sweep runs every job in one process, so a script naming
    its output file after the run gives every job that same name; writing them to
    one configured directory would then leave only the last.
    """
    return HydraConfig.get().runtime.output_dir
