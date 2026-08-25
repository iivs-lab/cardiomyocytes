from __future__ import annotations

__all__ = ("StageInputs",)

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from iivs_cardio.data.transforms.filtering.kernel import KernelConfig
from scripts._common.compute import ComputeConfig
from scripts._common.dataset import SequenceSelectConfig, SourceConfig
from scripts._common.filtering import parse_filter_config
from scripts._common.hydra import apply_schema

if TYPE_CHECKING:
    from omegaconf import DictConfig


@dataclass(frozen=True, slots=True)
class StageInputs[S: SourceConfig]:
    """What every stage reads out of the configuration it was composed from.

    One value rather than four locals, so a stage's `main` says which settings
    it took and nothing about how each of them is read. Every stage reads the
    same four: what to read, which of it to take, how to filter it, and what to
    run it on. A stage with more of its own subclasses this and adds them.

    The configuration itself stays a `DictConfig` in `main`, since that is what
    `hydra.main` hands a job. What is settled here is that nothing past it has
    to know: everything below holds plain values, checked against the fields
    they were read for.

    Type Parameters:
        S: The tree this stage reads, which is the one thing about the four
            that differs between stages.

    Attributes:
        source: The tree the sequences are read from.
        select: Which of its sequences to take.
        kernel: The filter each frame goes through, which is a kernel that does
            nothing where the configuration named none.
        compute: The devices to run on, and what to report.
    """

    source: S
    select: SequenceSelectConfig
    kernel: KernelConfig
    compute: ComputeConfig

    @classmethod
    def _shared(cls, schema: type[S], config: DictConfig) -> dict[str, Any]:
        """Read the four every stage takes, as the keywords they are held under.

        Kept apart from `read` so a subclass composes its own reading out of
        this one rather than restating it: `cls(**cls._shared(...), target=...)`.

        Args:
            schema: The tree config of the stage doing the reading.
            config: The node the whole job was composed into.
        """
        return {
            "source": apply_schema(schema, config.source),
            "select": apply_schema(SequenceSelectConfig, config.select),
            "kernel": parse_filter_config(config.get("filter")),
            "compute": apply_schema(ComputeConfig, config.compute),
        }
