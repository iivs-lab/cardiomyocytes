from __future__ import annotations

__all__ = ("describe_filter_kernel", "parse_filter_config")

from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Final

from hydra.utils import instantiate

from iivs_cardio.data.transforms.filtering.kernel import IdentityConfig, KernelConfig

if TYPE_CHECKING:
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
        msg = f"`filter` describes no kernel: give it a `_target_`, or select from `{_GROUP}`"
        raise TypeError(msg)

    return config


def describe_filter_kernel(config: KernelConfig) -> dict[str, Any]:
    """Return a kernel's settings as plain data, with what kind it is.

    A fresh mapping each call, so a caller may change or drop keys without
    reaching the settings anyone else was given.
    """
    return {"kind": config.kind, **asdict(config)}
