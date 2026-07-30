from __future__ import annotations

__all__ = ("build_filter_kernel", "describe_filter_kernel")

from typing import TYPE_CHECKING, Any

from hydra.utils import instantiate
from omegaconf import OmegaConf

from iivs_cardio.data.transforms.filtering.kernel import IdentityKernel, IdentityParams

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from iivs_cardio.data.transforms.filtering.kernel import FilterKernel, KernelParams

# A `_target_` is a dotted path the config chooses and `instantiate` imports and
# calls, so hydra 1.4 wants the callsite to say what it means to build.
KERNEL_TARGETS = ("iivs_cardio.data.transforms.filtering.kernel.*",)

IDENTITY_TARGET = f"{IdentityParams.__module__}.{IdentityParams.__qualname__}"


def build_filter_kernel(node: DictConfig | None) -> FilterKernel:
    if not node:
        return IdentityKernel()

    params: KernelParams = instantiate(node, _target_whitelist_=KERNEL_TARGETS)

    return params.build()


def describe_filter_kernel(node: DictConfig | None) -> dict[str, Any]:
    # Spelled out rather than passed through: a run that filters nothing reaches
    # here as `null`, an absent key, or an empty node, and a record saying any of
    # those leaves a reader guessing what actually ran.
    if not node:
        return {"_target_": IDENTITY_TARGET}

    return OmegaConf.to_container(node, resolve=True)  # ty: ignore[invalid-return-type]
