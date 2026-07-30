from __future__ import annotations

__all__ = (
    "DEVICE_KINDS",
    "DeviceKind",
    "resolve_device",
    "resolve_devices",
    "visible_cuda_devices",
)

from typing import TYPE_CHECKING, Literal, get_args

import torch

if TYPE_CHECKING:
    from collections.abc import Iterable

DeviceKind = Literal["cpu", "cuda"]

DEVICE_KINDS: frozenset[DeviceKind] = frozenset(get_args(DeviceKind))


def resolve_device(
    spec: str | torch.device,
    supported: frozenset[DeviceKind] = DEVICE_KINDS,
) -> torch.device:
    """Parse a device spec, validate its kind, and normalize it.

    Strings are matched case-insensitively (`"cpu"`, `"cuda"`, `"cuda:N"`); a
    `torch.device` is validated as-is. `cpu` is returned unnumbered; `cuda` is
    given a concrete index (defaulting to `0`) so it compares equal to a
    tensor's `.device`, which always carries one.

    Args:
        spec: A device string or a `torch.device`.
        supported: The device kinds to accept.

    Returns:
        A normalized `torch.device` whose `type` is in `supported`.

    Raises:
        ValueError: If the device kind is not in `supported`.
    """
    device = spec if isinstance(spec, torch.device) else torch.device(spec.lower())

    if device.type not in supported:
        allowed = ", ".join(sorted(supported))
        msg = f"unsupported device {device.type!r}: expected one of {allowed}"
        raise ValueError(msg)

    if device.type == "cpu":
        return torch.device("cpu")  # cpu is unnumbered; drop any index
    return torch.device("cuda", device.index or 0)  # cuda always carries an index


def visible_cuda_devices() -> tuple[torch.device, ...]:
    """Every CUDA device this process can see, in index order.

    Empty when the driver reports none, which a caller asking to spread work
    across GPUs should treat as a configuration error rather than as zero work.
    """
    return tuple(torch.device("cuda", index) for index in range(_cuda_count()))


def resolve_devices(
    specs: Iterable[str | torch.device],
    supported: frozenset[DeviceKind] = DEVICE_KINDS,
) -> tuple[torch.device, ...]:
    """Resolve each spec, and check that every CUDA index is one this host has.

    The plural of `resolve_device`, plus the bound check a single spec cannot
    usefully make: a caller naming devices one at a time is describing what it
    already holds, while a caller naming a set is planning work across them, and
    an index that is not there has to fail now rather than when a tensor first
    moves. Duplicates are kept, so `["cpu", "cpu"]` is two workers on the CPU.

    Args:
        specs: Device strings or `torch.device`s, in the order they are wanted.
        supported: The device kinds to accept.

    Returns:
        The normalized devices, in the order given.

    Raises:
        ValueError: If a kind is not in `supported`, or a CUDA index is beyond
            what this host reports.
    """
    devices = tuple(resolve_device(spec, supported) for spec in specs)

    count = _cuda_count()
    beyond = sorted({d.index for d in devices if d.type == "cuda" and d.index >= count})
    if beyond:
        listed = ", ".join(str(index) for index in beyond)
        msg = f"no CUDA device at index {listed}: this host reports {count}"
        raise ValueError(msg)

    return devices


def _cuda_count() -> int:
    """How many CUDA devices the driver reports, 0 when there is no driver."""
    return torch.cuda.device_count() if torch.cuda.is_available() else 0
