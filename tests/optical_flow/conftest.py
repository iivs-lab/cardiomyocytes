from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from iivs_cardio.optical_flow.estimators import FarnebackConfig

if TYPE_CHECKING:
    from iivs_cardio.common.device import Device


@pytest.fixture()
def algorithms_made(monkeypatch) -> list[Device]:
    """The devices a Farneback algorithm was made on, in the order it was made."""
    devices: list[Device] = []
    make = FarnebackConfig._algorithm  # noqa: SLF001

    def spy(self, device):
        devices.append(device)
        return make(self, device)

    monkeypatch.setattr(FarnebackConfig, "_algorithm", spy)

    return devices
