from __future__ import annotations

__all__ = ("apply_schema", "is_multirun", "output_directory", "sweep_parameters")

from typing import TYPE_CHECKING, cast

from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import OmegaConf

if TYPE_CHECKING:
    from omegaconf import DictConfig


def apply_schema[T](schema: type[T], node: DictConfig) -> T:
    """Read a configuration node as `schema`, checking it against the fields.

    What comes back holds plain values rather than configuration containers,
    so the rest of the code never has to know where its settings came from.

    Returns:
        The node as an instance of `schema`.

    Raises:
        ValidationError: If a value does not fit the field it was given for.
    """
    merged = OmegaConf.merge(OmegaConf.structured(schema), node)
    return cast("T", OmegaConf.to_object(merged))


def output_directory() -> str:
    """Return Hydra's own directory for this job, which a run writes into.

    `hydra.run.dir` for a single run, and `hydra.sweep.dir/<subdir>` for each job
    of a `--multirun`. A sweep runs every job in one process, so a script naming
    its output file after the run gives every job that same name; writing them to
    one configured directory would then leave only the last.
    """
    return HydraConfig.get().runtime.output_dir


def is_multirun() -> bool:
    """Test whether this job is one of a `--multirun` sweep, not a lone run.

    What `output_directory` already accounts for, made answerable: a step that
    writes somewhere every job shares cannot be repeated per job, and has to
    refuse the sweep rather than let the jobs race for it.
    """
    return HydraConfig.get().mode is RunMode.MULTIRUN


def sweep_parameters() -> tuple[str, ...]:
    """The settings a composed sweep would vary, empty where none was composed.

    An experiment names its jobs under `hydra.sweeper.params`, which the sweeper
    reads and a lone run never looks at. Answering it is what lets a caller
    refuse the one shape that fails silently: `+experiment=...` written without
    `--multirun` runs once, on the defaults, saying nothing about the sweep it
    was handed.
    """
    params = HydraConfig.get().sweeper.get("params")

    return () if params is None else tuple(params)
