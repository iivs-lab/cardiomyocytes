from __future__ import annotations

__all__ = ("apply_schema",)

from typing import TYPE_CHECKING, cast

from omegaconf import OmegaConf

if TYPE_CHECKING:
    from omegaconf import DictConfig


def apply_schema[T](schema: type[T], node: DictConfig) -> T:
    """Check `node` against `schema` and return the dataclass it describes.

    Merging onto the schema is what does the checking. An unknown key, or a value
    that cannot become its declared type, fails here rather than at whichever line
    first reads it. What comes back is a plain dataclass, not a `DictConfig` node.

    Args:
        schema: The dataclass describing the section.
        node: The config section to check.

    Returns:
        An instance of `schema`, built from `node` over the schema's defaults.

    Raises:
        ValidationError: If a value cannot be converted to its declared type.
        ConfigKeyError: If `node` holds a key `schema` does not declare.
    """
    merged = OmegaConf.merge(OmegaConf.structured(schema), node)
    return cast("T", OmegaConf.to_object(merged))
