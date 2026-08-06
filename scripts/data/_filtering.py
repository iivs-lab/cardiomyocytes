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
    if not node:
        return IdentityConfig()

    return instantiate(node, _target_whitelist_=_WHITELIST, _convert_="all")


def describe_filter_kernel(config: KernelConfig) -> dict[str, Any]:
    return {"kind": config.kind, **asdict(config)}
