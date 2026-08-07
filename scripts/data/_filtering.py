from __future__ import annotations

__all__ = ("describe_filter_kernel", "parse_filter_config")

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from hydra.utils import instantiate

from iivs_cardio.data.transforms.filtering.kernel import IdentityConfig

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from iivs_cardio.data.transforms.filtering.kernel import KernelConfig

_WHITELIST = ("iivs_cardio.data.transforms.filtering.kernel.*",)


def parse_filter_config(node: DictConfig | None) -> KernelConfig:
    """Read a kernel's settings from a configuration node.

    An absent node means no filtering, which is a kernel of its own rather
    than a missing one. What comes back holds plain values, not configuration
    containers, so it can be recorded as it stands.

    Returns:
        The settings the node describes.

    Raises:
        InstantiationException: If the node names something that is not one
            of this project's kernels.
    """
    if not node:
        return IdentityConfig()

    return instantiate(node, _target_whitelist_=_WHITELIST, _convert_="all")


def describe_filter_kernel(config: KernelConfig) -> dict[str, Any]:
    """Return a kernel's settings as plain data, with what kind it is.

    A fresh mapping each call, so a caller may change or drop keys without
    reaching the settings anyone else was given.
    """
    return {"kind": config.kind, **asdict(config)}
