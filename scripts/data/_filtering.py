from __future__ import annotations

__all__ = ("build_filter_kernel",)

from typing import TYPE_CHECKING

from hydra.utils import instantiate

from iivs_cardio.data.transforms.filtering.kernel import IdentityKernel

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from iivs_cardio.data.transforms.filtering.kernel import FilterKernel, KernelParams

# A `_target_` is a dotted path the config chooses and `instantiate` imports and
# calls, so hydra 1.4 wants the callsite to say what it means to build.
KERNEL_TARGETS = ("iivs_cardio.data.transforms.filtering.kernel.*",)


def build_filter_kernel(node: DictConfig | None) -> FilterKernel:
    if not node:
        return IdentityKernel()

    params: KernelParams = instantiate(node, _target_whitelist_=KERNEL_TARGETS)

    return params.build()
