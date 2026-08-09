from __future__ import annotations

__all__ = ("DEVICE_KINDS", "Device", "DeviceKind", "DeviceLike")

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Final, Literal

import torch
from kaparoo.utils import literal_values, unwrap_or_default

if TYPE_CHECKING:
    from collections.abc import Iterable

type DeviceKind = Literal["cpu", "cuda"]

# What a caller may write for a device; `Device` is the form it is stored as.
type DeviceLike = str | torch.device | Device

# Taken off the alias rather than written twice.
DEVICE_KINDS: Final[frozenset[DeviceKind]] = frozenset(literal_values(DeviceKind))


def _cuda_count() -> int:
    """Count the CUDA devices the driver reports, 0 when there is no driver.

    Deliberately not guarded by `torch.cuda.is_available()`, which initializes
    CUDA in whichever process asks: a pool started by forking after that gives
    every worker a context it cannot use. This answers without doing so, and
    already answers 0 where the guard was there to.
    """
    return torch.cuda.device_count()


# `slots=True` is deliberately absent: it removes the instance `__dict__` that
# `cached_property` writes `torch` into. The frames of a run go through `torch`
# one at a time, so rebuilding it per read is what the cache is there to avoid.
@dataclass(frozen=True)
class Device:
    """One compute device, in the form every library in this stack must agree on.

    torch carries a device on each tensor, but `cv2.cuda` and CuPy each keep a
    process-global current device instead. Naming a device is therefore not the
    same as working on it, and this type separates the two: the value says which
    device, and pointing the libraries at it is a distinct step.

    A `cuda` device always carries a concrete index, so it compares equal to what
    a tensor reports; `cpu` is unnumbered. Construct through `resolve` to accept
    what a caller writes; the constructor is for a kind and index already known.

    Args:
        kind: The family of device.
        index: The CUDA device to name. Ignored for `cpu`, which takes no
            index. Defaults to `None`, which comes to `0` on cuda.

    Raises:
        ValueError: If `kind` is not a known kind, or `index` is negative.
    """

    kind: DeviceKind
    index: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in DEVICE_KINDS:
            allowed = ", ".join(sorted(DEVICE_KINDS))
            msg = f"unsupported device {self.kind!r}: expected one of {allowed}"
            raise ValueError(msg)

        if self.index is not None and self.index < 0:
            msg = f"negative device index {self.index}: expected 0 or more, or None"
            raise ValueError(msg)

        index = None if self.kind == "cpu" else unwrap_or_default(self.index, 0)
        if index != self.index:
            object.__setattr__(self, "index", index)  # frozen: normalize in place

    @classmethod
    def resolve(
        cls,
        spec: DeviceLike,
        supported: frozenset[DeviceKind] = DEVICE_KINDS,
    ) -> Device:
        """Parse a device spec, validate its kind, and normalize it.

        Strings are matched case-insensitively (`"cpu"`, `"cuda"`, `"cuda:N"`); a
        `torch.device` is read as-is, and a `Device` passes through, so a layer may
        re-resolve what it was handed without knowing which form it arrived in.

        A CUDA index is not checked against the host here, since a caller naming
        one device is describing what it already holds. `resolve_all` makes that check,
        since naming a set is planning work across it.

        Args:
            spec: The device to resolve, in any form a caller may write.
            supported: The device kinds to accept. Defaults to every kind.

        Returns:
            The normalized device, whose `kind` is in `supported`.

        Raises:
            ValueError: If `spec` is malformed, or its kind is not in `supported`.
        """
        kind, index = (
            (spec.kind, spec.index) if isinstance(spec, Device) else _parse(spec)
        )

        if kind not in supported:
            allowed = ", ".join(sorted(supported))
            msg = f"unsupported device {kind!r}: expected one of {allowed}"
            raise ValueError(msg)

        return cls(kind, index)

    @classmethod
    def resolve_all(
        cls,
        specs: Iterable[DeviceLike],
        supported: frozenset[DeviceKind] = DEVICE_KINDS,
    ) -> tuple[Device, ...]:
        """Resolve each spec, and check that every CUDA index is one this host has.

        The plural of `resolve`, plus the bound check a single spec cannot usefully
        make: an index that is not there has to fail now rather than when a tensor
        first moves. Duplicates are kept, so `["cpu", "cpu"]` is two workers on the
        CPU. The driver is asked only when a CUDA device is actually named.

        Args:
            specs: The devices to resolve, in the order they are wanted.
            supported: The device kinds to accept. Defaults to every kind.

        Returns:
            The normalized devices, in the order given.

        Raises:
            ValueError: If a spec is malformed, a kind is not in `supported`, or a
                CUDA index is beyond what this host reports.
        """
        devices = tuple(cls.resolve(spec, supported) for spec in specs)

        wanted = {device.index for device in devices if device.is_cuda}
        if wanted:
            count = _cuda_count()
            beyond = sorted(
                index for index in wanted if index is not None and index >= count
            )
            if beyond:
                listed = ", ".join(str(index) for index in beyond)
                msg = f"no CUDA device at index {listed}: this host reports {count}"
                raise ValueError(msg)

        return devices

    @classmethod
    def visible_cuda(cls) -> tuple[Device, ...]:
        """Every CUDA device this process can see, in index order.

        Empty when the driver reports none, which a caller asking to spread work
        across GPUs should treat as a configuration error rather than as zero work.
        """
        return tuple(cls("cuda", index) for index in range(_cuda_count()))

    @cached_property
    def as_torch(self) -> torch.device:
        """This device as torch names it, for the calls that take one.

        Named apart from the module rather than `torch`: a member of that name
        shadows it for every annotation in this class body.
        """
        if self.index is None:
            return torch.device(self.kind)
        return torch.device(self.kind, self.index)

    @property
    def is_cuda(self) -> bool:
        """Whether this is a CUDA device."""
        return self.kind == "cuda"

    def activate(self) -> None:
        """Point this process's CUDA libraries at this device.

        torch takes the device from each tensor it is given, but `cv2.cuda` and
        CuPy each read a process-global current device instead. Both default to
        device 0 and nothing else here moves them, so on any GPU but the first
        they disagree with the tensors they are handed: CuPy would label a
        pointer from device 1 as device 0's. A `cpu` device has nothing to bind.

        Cheap enough to repeat per item on a hot path rather than hoisted into
        worker setup, which a lone in-process run would then have to duplicate.
        """
        if self.index is None:  # cpu, since only a cuda device carries an index
            return

        # Imported here rather than at module scope: this module is on the pure
        # torch filtering path, which would otherwise pay to import both stacks.
        import cupy as cp
        import cv2

        torch.cuda.set_device(self.index)
        cv2.cuda.setDevice(self.index)
        cp.cuda.Device(self.index).use()

    def __str__(self) -> str:
        return self.kind if self.index is None else f"{self.kind}:{self.index}"


def _parse(spec: str | torch.device) -> tuple[str, int | None]:
    """Read a device spec as a `(kind, index)` pair, without judging the kind.

    Raises:
        ValueError: If `spec` is not a device string torch can read.
    """
    try:
        device = spec if isinstance(spec, torch.device) else torch.device(spec.lower())
    except RuntimeError:
        allowed = ", ".join(sorted(DEVICE_KINDS))
        msg = f"invalid device spec {spec!r}: expected {allowed}, or `<kind>:N`"
        raise ValueError(msg) from None

    return device.type, device.index
