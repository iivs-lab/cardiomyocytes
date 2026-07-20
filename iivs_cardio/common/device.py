from __future__ import annotations

__all__ = ("DEVICE_KINDS", "DeviceKind", "resolve_device")

from typing import Literal, get_args

import torch

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
