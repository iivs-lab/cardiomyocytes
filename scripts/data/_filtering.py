from __future__ import annotations

__all__ = ("build_filter_config", "build_filter_kernel", "describe_filter_kernel")

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from hydra.utils import instantiate

from iivs_cardio.data.transforms.filtering.kernel import IdentityConfig

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from iivs_cardio.data.transforms.filtering.kernel import FilterKernel, KernelConfig

# A `_target_` is a dotted path the config chooses and `instantiate` imports and
# calls, so hydra 1.4 wants the callsite to say what it means to build.
KERNEL_TARGETS = ("iivs_cardio.data.transforms.filtering.kernel.*",)


def build_filter_config(node: DictConfig | None) -> KernelConfig:
    # A run that filters nothing reaches here as `null`, an absent key, or an
    # empty node; all three mean the same kernel.
    if not node:
        return IdentityConfig()

    config: KernelConfig = instantiate(node, _target_whitelist_=KERNEL_TARGETS)

    return config


def build_filter_kernel(node: DictConfig | None) -> FilterKernel:
    return build_filter_config(node).build()


def describe_filter_kernel(node: DictConfig | None) -> dict[str, Any]:
    # What was filtered, not which class did it: an import path would make two
    # runs that filtered the same way compare unequal once the code moves. Read
    # off the built config rather than the node, so a config `instantiate` would
    # reject fails before a document records it.
    config = build_filter_config(node)

    return {"kind": config.kind, **asdict(config)}
