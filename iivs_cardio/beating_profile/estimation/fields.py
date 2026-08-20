from __future__ import annotations

__all__ = (
    "Acceleration",
    "Displacement",
    "Displacement2D",
    "Displacement3D",
    "DryMass",
    "Force",
    "KineticEnergy",
    "OPDVariance",
    "Velocity",
)

from dataclasses import dataclass, field
from math import tau
from typing import TYPE_CHECKING, ClassVar, Final

import torch

from iivs_cardio.beating_profile.estimation.base import (
    Field,
    FlowInput,
    Need,
    PhaseInput,
)
from iivs_cardio.common.warp import BackwardWarp

if TYPE_CHECKING:
    from torch import Tensor

_YOCTO_PER_UNIT: Final = 1e-3


@dataclass(frozen=True, slots=True)
class DryMass(Field):
    NEEDS: ClassVar = (Need(PhaseInput),)

    pixel_size: float
    wavelength: float
    alpha: float

    _scale: float = field(init=False, default=1.0, repr=False)

    def __post_init__(self) -> None:
        opd_scale = self.wavelength / tau / 1000  # um
        area_scale = self.pixel_size**2  # um^2
        _scale = opd_scale * area_scale / self.alpha  # pg
        object.__setattr__(self, "_scale", _scale)

    def compute(self, phase: Tensor, /) -> Tensor | None:
        return phase * self._scale


@dataclass(frozen=True, slots=True)
class OPDVariance(Field):
    NEEDS: ClassVar = (Need(PhaseInput, next=1),)

    wavelength: float

    _scale: float = field(init=False, default=1.0, repr=False)

    def __post_init__(self) -> None:
        _scale = self.wavelength / tau  # nm
        object.__setattr__(self, "_scale", _scale)

    def compute(self, phase1: Tensor, phase2: Tensor, /) -> Tensor | None:
        change = (phase2 - phase1) * self._scale

        return torch.stack((change, change.square()))


class Displacement(Field):
    pass


@dataclass(frozen=True, slots=True)
class Displacement2D(Displacement):
    NEEDS: ClassVar = (Need(FlowInput),)

    pixel_size: float  # unit: um

    def compute(self, flow: Tensor, /) -> Tensor | None:
        return flow * self.pixel_size  # unit: um


@dataclass(frozen=True, slots=True)
class Displacement3D(Displacement):
    NEEDS: ClassVar = (Need(FlowInput), Need(PhaseInput, next=1))

    pixel_size: float  # unit: um
    wavelength: float  # unit: nm
    refractive_delta: float  # unitless

    _z_scale: float = field(init=False, default=1.0, repr=False)
    _warp: BackwardWarp = field(init=False, default_factory=BackwardWarp, repr=False)

    def __post_init__(self) -> None:
        _z_scale = self.wavelength / (tau * self.refractive_delta)  # unit: nm
        object.__setattr__(self, "_z_scale", 0.0005 * _z_scale)

    def compute(self, flow: Tensor, phase1: Tensor, phase2: Tensor, /) -> Tensor | None:
        dx_dy = flow * self.pixel_size
        dz = (self._warp(phase2, flow) - phase1) * self._z_scale
        return torch.cat((dx_dy, dz.unsqueeze(0)), dim=0)  # unit: um


@dataclass(frozen=True, slots=True)
class Velocity(Field):
    NEEDS: ClassVar = (Need(Displacement),)

    frame_rate: float  # unit: Hz

    def compute(self, displacement: Tensor, /) -> Tensor | None:
        return displacement * self.frame_rate  # unit: um/s


@dataclass(frozen=True, slots=True)
class Acceleration(Field):
    NEEDS: ClassVar = (Need(Velocity, prev=1),)

    frame_rate: float  # unit: Hz

    def compute(self, velocity1: Tensor, velocity2: Tensor, /) -> Tensor | None:
        return (velocity2 - velocity1) * self.frame_rate  # unit: um/s^2


@dataclass(frozen=True, slots=True)
class Force(Field):
    NEEDS: ClassVar = (Need(DryMass), Need(Acceleration))

    _scale: float = field(init=False, default=1.0, repr=False)

    def compute(self, drymass: Tensor, acceleration: Tensor, /) -> Tensor | None:
        return drymass * acceleration  # unit: pN


@dataclass(frozen=True, slots=True)
class KineticEnergy(Field):
    NEEDS: ClassVar = (Need(DryMass), Need(Velocity))

    _scale: float = field(init=False, default=1.0, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_scale", _YOCTO_PER_UNIT / 2)

    def compute(self, drymass: Tensor, velocity: Tensor, /) -> Tensor | None:
        return drymass * velocity.square().sum(dim=0)
