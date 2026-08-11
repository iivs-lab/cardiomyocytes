from __future__ import annotations

__all__ = ("describe_filter_kernel", "log_filter_config", "parse_filter_config")

from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Final

from hydra.utils import instantiate

from iivs_cardio.common.logging import log_indented
from iivs_cardio.data.transforms.filtering.kernel import IdentityConfig, KernelConfig

if TYPE_CHECKING:
    from logging import Logger

    from omegaconf import DictConfig

_WHITELIST: Final = ("iivs_cardio.data.transforms.filtering.kernel.*",)

# The config group `filter` is filled from. Its name is the key it fills, so an
# override selects from it as `filter=<option>`.
_GROUP: Final = "filter"


def parse_filter_config(node: DictConfig | None) -> KernelConfig:
    """Read a kernel's settings from a configuration node.

    An absent node means no filtering, which is a kernel of its own rather
    than a missing one. What comes back holds plain values, not configuration
    containers, so it can be recorded as it stands.

    Returns:
        The settings the node describes.

    Raises:
        TypeError: If the node is a kernel's name rather than the kernel, or
            describes something that is not a kernel at all.
        InstantiationException: If the node names something that is not one
            of this project's kernels.
    """
    if not node:
        return IdentityConfig()

    if isinstance(node, str):
        msg = f"`filter` holds the name {node!r}: it has to select from `{_GROUP}`"
        raise TypeError(msg)

    config = instantiate(node, _target_whitelist_=_WHITELIST, _convert_="all")
    if not isinstance(config, KernelConfig):
        fix = f"give it a `_target_`, or select from `{_GROUP}`"
        msg = f"`filter` describes no kernel: {fix}"
        raise TypeError(msg)

    return config


def describe_filter_kernel(config: KernelConfig) -> dict[str, Any]:
    """Return a kernel's settings as plain data, with what kind it is.

    A fresh mapping each call, so a caller may change or drop keys without
    reaching the settings anyone else was given.
    """
    return {"kind": config.kind, **asdict(config)}


def log_filter_config(kernel_config: KernelConfig, logger: Logger) -> None:
    """Log the filter a run applies, with the settings that shape it."""
    described = describe_filter_kernel(kernel_config)
    kind = described.pop("kind")
    settings = ", ".join(f"{key}={value}" for key, value in described.items())
    settings = f" ({settings})" if settings else ""
    log_indented(logger, "filter: %s kernel%s", kind, settings, depth=0)
