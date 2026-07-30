from __future__ import annotations

import pytest
import torch

from iivs_cardio.common import (
    DEVICE_KINDS,
    resolve_device,
    resolve_devices,
    visible_cuda_devices,
)


def test_resolve_cpu():
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_cuda_defaults_to_index_zero():
    assert resolve_device("cuda") == torch.device("cuda", 0)


@pytest.mark.parametrize(("spec", "index"), (("cuda:0", 0), ("cuda:1", 1)))
def test_resolve_cuda_indexed(spec, index):
    assert resolve_device(spec) == torch.device("cuda", index)


@pytest.mark.parametrize(
    ("spec", "expected"),
    (
        ("CPU", torch.device("cpu")),
        ("Cuda", torch.device("cuda", 0)),
        ("CUDA:1", torch.device("cuda", 1)),
    ),
)
def test_resolve_is_case_insensitive(spec, expected):
    assert resolve_device(spec) == expected


def test_resolve_passthrough_torch_device_normalizes_cuda():
    # a bare cuda torch.device (index None) is normalized to a concrete index
    assert resolve_device(torch.device("cuda")) == torch.device("cuda", 0)
    assert resolve_device(torch.device("cpu")) == torch.device("cpu")


def test_resolve_rejects_unsupported_kind():
    # cuda is a valid torch device but excluded by the supported set
    with pytest.raises(ValueError, match=r"unsupported device 'cuda'"):
        resolve_device("cuda", frozenset({"cpu"}))


def test_resolve_cpu_drops_index():
    # cpu is unnumbered, so any index is normalized away
    assert resolve_device(torch.device("cpu", 0)) == torch.device("cpu")


def test_device_kinds_matches_literal():
    assert frozenset({"cpu", "cuda"}) == DEVICE_KINDS


# ------------------------------- resolve_devices -------------------------------- #


def test_resolve_devices_keeps_order_and_duplicates():
    # Duplicates are the point on the CPU: N entries is N workers on one device.
    assert resolve_devices(["cpu", "cpu", "cpu"]) == (torch.device("cpu"),) * 3


def test_resolve_devices_normalizes_each_spec():
    assert (
        resolve_devices(["CPU", torch.device("cpu", 0)]) == (torch.device("cpu"),) * 2
    )


def test_resolve_devices_of_nothing_is_nothing():
    assert resolve_devices([]) == ()


def test_resolve_devices_rejects_an_unsupported_kind():
    with pytest.raises(ValueError, match=r"unsupported device 'cuda'"):
        resolve_devices(["cpu", "cuda"], frozenset({"cpu"}))


def test_resolve_devices_rejects_an_index_this_host_lacks():
    # The check a single spec cannot usefully make: naming a set is planning work
    # across it, so an absent index has to fail here rather than at the first
    # tensor move. The message names every missing index, sorted, and once each.
    beyond = torch.cuda.device_count() if torch.cuda.is_available() else 0
    specs = [f"cuda:{beyond + 1}", "cpu", f"cuda:{beyond}", f"cuda:{beyond + 1}"]

    with pytest.raises(ValueError, match=rf"no CUDA device at index {beyond}, "):
        resolve_devices(specs)


def test_visible_cuda_devices_are_indexed_from_zero():
    devices = visible_cuda_devices()

    assert all(device.type == "cuda" for device in devices)
    assert [device.index for device in devices] == list(range(len(devices)))


def test_visible_cuda_devices_agree_with_the_driver():
    expected = torch.cuda.device_count() if torch.cuda.is_available() else 0

    assert len(visible_cuda_devices()) == expected


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA-capable GPU detected"
)
def test_every_visible_cuda_device_resolves():
    devices = visible_cuda_devices()

    assert resolve_devices(devices) == devices
