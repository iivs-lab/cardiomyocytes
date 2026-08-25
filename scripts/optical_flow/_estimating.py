from __future__ import annotations

__all__ = (
    "describe_estimator_config",
    "log_estimator_config",
    "parse_estimator_config",
)

from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Final

from hydra.utils import instantiate

from iivs_cardio.common.logging import log_indented
from iivs_cardio.optical_flow.estimators import EstimatorConfig

if TYPE_CHECKING:
    from logging import Logger

    from omegaconf import DictConfig

_WHITELIST: Final = ("iivs_cardio.optical_flow.estimators.*",)

# The config group `estimator` is filled from. Its name is the key it fills, so
# an override selects from it as `estimator=<option>`.
_GROUP: Final = "estimator"


def parse_estimator_config(node: DictConfig | None) -> EstimatorConfig:
    """Read an estimator's settings from a configuration node.

    Unlike the filter there is no estimator that does nothing, so an absent node
    is a run with no way to produce a flow rather than a run that produces them
    unchanged.

    Returns:
        The settings the node describes.

    Raises:
        TypeError: If the node is absent, is an estimator's name rather than the
            estimator, or describes something that is not an estimator at all.
        InstantiationException: If the node names something that is not one of
            this project's estimators.
    """
    if not node:
        msg = f"`estimator` is not set: select from `{_GROUP}`"
        raise TypeError(msg)

    if isinstance(node, str):
        msg = f"`estimator` holds the name {node!r}: it has to select from `{_GROUP}`"
        raise TypeError(msg)

    config = instantiate(node, _target_whitelist_=_WHITELIST, _convert_="all")
    if not isinstance(config, EstimatorConfig):
        fix = f"give it a `_target_`, or select from `{_GROUP}`"
        msg = f"`estimator` describes no estimator: {fix}"
        raise TypeError(msg)

    return config


def describe_estimator_config(config: EstimatorConfig) -> dict[str, Any]:
    """Return an estimator's settings as plain data, with which one it is.

    The kind is read off the class rather than declared on it, since nothing in
    the package branches on it: the name exists for a reader of the document and
    for a later run comparing what wrote one, and both of those already read the
    settings beside it.

    A fresh mapping each call, so a caller may change or drop keys without
    reaching the settings anyone else was given.
    """
    kind = type(config).__name__.removesuffix("Config").lower()

    return {"kind": kind, **asdict(config)}  # ty: ignore[invalid-argument-type]


def log_estimator_config(estimator_config: EstimatorConfig, logger: Logger) -> None:
    """Log the estimator a run computes with, with the settings that shape it."""
    described = describe_estimator_config(estimator_config)
    kind = described.pop("kind")
    settings = ", ".join(f"{key}={value}" for key, value in described.items())
    settings = f" ({settings})" if settings else ""
    log_indented(logger, "estimator: %s%s", kind, settings, depth=0)
