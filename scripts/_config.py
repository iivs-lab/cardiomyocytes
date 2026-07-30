from __future__ import annotations

__all__ = ("apply_schema",)

from typing import TYPE_CHECKING, cast

from omegaconf import OmegaConf

if TYPE_CHECKING:
    from omegaconf import DictConfig


def apply_schema[T](schema: type[T], node: DictConfig) -> T:
    merged = OmegaConf.merge(OmegaConf.structured(schema), node)
    return cast("T", OmegaConf.to_object(merged))
