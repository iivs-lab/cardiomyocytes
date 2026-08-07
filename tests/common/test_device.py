from __future__ import annotations

import pickle

import pytest
import torch

from iivs_cardio.common import DEVICE_KINDS, Device


def test_resolve_cpu():
    assert Device.resolve("cpu").as_torch == torch.device("cpu")


def test_resolve_cuda_defaults_to_index_zero():
    assert Device.resolve("cuda").as_torch == torch.device("cuda", 0)


@pytest.mark.parametrize(("spec", "index"), (("cuda:0", 0), ("cuda:1", 1)))
def test_resolve_cuda_indexed(spec, index):
    assert Device.resolve(spec).as_torch == torch.device("cuda", index)


@pytest.mark.parametrize(
    ("spec", "expected"),
    (
        ("CPU", torch.device("cpu")),
        ("Cuda", torch.device("cuda", 0)),
        ("CUDA:1", torch.device("cuda", 1)),
    ),
)
def test_resolve_is_case_insensitive(spec, expected):
    assert Device.resolve(spec).as_torch == expected


def test_resolve_passthrough_torch_device_normalizes_cuda():
    # a bare cuda torch.device (index None) is normalized to a concrete index
    assert Device.resolve(torch.device("cuda")).as_torch == torch.device("cuda", 0)
    assert Device.resolve(torch.device("cpu")).as_torch == torch.device("cpu")


def test_resolve_rejects_unsupported_kind():
    # cuda is a valid torch device but excluded by the supported set
    with pytest.raises(ValueError, match=r"unsupported device 'cuda'"):
        Device.resolve("cuda", frozenset({"cpu"}))


def test_resolve_cpu_drops_index():
    # cpu is unnumbered, so any index is normalized away
    assert Device.resolve(torch.device("cpu", 0)).as_torch == torch.device("cpu")


def test_device_kinds_matches_literal():
    assert frozenset({"cpu", "cuda"}) == DEVICE_KINDS


@pytest.mark.parametrize("spec", ("gpu", "cuda:abc", "cuda:-1", ""))
def test_resolve_rejects_a_malformed_spec(spec):
    # torch itself raises RuntimeError here, which would leak a second failure
    # type out of an API whose every other rejection is a ValueError.
    with pytest.raises(ValueError, match=r"invalid device spec"):
        Device.resolve(spec)


def test_resolve_passes_a_device_through_unchanged():
    # the third form of `DeviceLike`: a layer re-resolves what it was handed
    # without knowing which form it arrived in
    device = Device.resolve("cuda:1")

    assert Device.resolve(device) == device


def test_resolve_still_checks_the_kind_of_a_device_it_is_handed():
    with pytest.raises(ValueError, match=r"unsupported device 'cuda'"):
        Device.resolve(Device("cuda", 1), frozenset({"cpu"}))


# ---------------------------------- Device ----------------------------------- #


def test_constructor_gives_cuda_a_concrete_index():
    assert Device("cuda") == Device("cuda", 0)


def test_constructor_drops_an_index_from_cpu():
    assert Device("cpu", 3).index is None


def test_constructor_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match=r"unsupported device 'gpu'"):
        Device("gpu")  # ty: ignore[invalid-argument-type]


def test_constructor_rejects_a_negative_index():
    with pytest.raises(ValueError, match=r"negative device index -1"):
        Device("cuda", -1)


@pytest.mark.parametrize(
    ("device", "expected"),
    ((Device("cpu"), "cpu"), (Device("cuda"), "cuda:0"), (Device("cuda", 2), "cuda:2")),
)
def test_str_reads_like_a_device_spec(device, expected):
    assert str(device) == expected


def test_is_cuda_distinguishes_the_kinds():
    assert Device("cuda", 1).is_cuda
    assert not Device("cpu").is_cuda


def test_as_torch_is_built_once_and_kept():
    # `.as_torch` is read per frame on the filtering path, so it must not rebuild
    device = Device("cpu")

    assert device.as_torch is device.as_torch


def test_devices_survive_the_spawn_boundary():
    # a worker claims its device through a queue, so `Device` has to pickle
    device = Device("cuda", 1)

    assert pickle.loads(pickle.dumps(device)) == device  # noqa: S301


def test_devices_are_hashable():
    assert {Device("cuda", 0), Device("cuda", 0), Device("cpu")} == {
        Device("cuda", 0),
        Device("cpu"),
    }


# ------------------------------ Device.activate ------------------------------- #
#
# cupy is imported inside each test, matching `test_cuda_utils`: collection must
# not import it on a machine without a CUDA runtime.


def test_activate_binds_every_library_to_the_same_index(monkeypatch):
    # The contract only shows itself on a second GPU, which this host may not
    # have, so it is pinned with spies rather than by observing a real bind.
    import cupy as cp
    import cv2

    bound: dict[str, int] = {}

    class _CuPyDevice:
        def __init__(self, index: int) -> None:
            self._index = index

        def use(self) -> None:
            bound["cupy"] = self._index

    monkeypatch.setattr(
        torch.cuda, "set_device", lambda index: bound.update(torch=index)
    )
    monkeypatch.setattr(cv2.cuda, "setDevice", lambda index: bound.update(cv2=index))
    monkeypatch.setattr(cp.cuda, "Device", _CuPyDevice)

    Device("cuda", 3).activate()  # index 3 need not exist; nothing real is called

    assert bound == {"torch": 3, "cv2": 3, "cupy": 3}


def test_activate_on_the_cpu_binds_nothing(monkeypatch):
    import cupy as cp
    import cv2

    def fail(*_: object) -> None:
        pytest.fail("a cpu device reached for a CUDA library")

    monkeypatch.setattr(torch.cuda, "set_device", fail)
    monkeypatch.setattr(cv2.cuda, "setDevice", fail)
    monkeypatch.setattr(cp.cuda, "Device", fail)

    Device("cpu").activate()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA-capable GPU detected"
)
def test_activate_leaves_the_libraries_agreeing():
    # Trivially true on a single-GPU host -- every index is 0 either way. It is
    # the multi-GPU host this is written for; the spy test above covers the rest.
    import cupy as cp
    import cv2

    for device in Device.visible_cuda():
        device.activate()

        assert torch.cuda.current_device() == device.index
        assert cv2.cuda.getDevice() == device.index
        assert cp.cuda.Device().id == device.index


# ------------------------------ Device.resolve_all ---------------------------- #


def test_resolve_all_keeps_order_and_duplicates():
    # Duplicates are the point on the CPU: N entries is N workers on one device.
    assert Device.resolve_all(["cpu", "cpu", "cpu"]) == (Device("cpu"),) * 3


def test_resolve_all_normalizes_each_spec():
    assert Device.resolve_all(["CPU", torch.device("cpu", 0)]) == (Device("cpu"),) * 2


def test_resolve_all_of_nothing_is_nothing():
    assert Device.resolve_all([]) == ()


def test_resolve_all_rejects_an_unsupported_kind():
    with pytest.raises(ValueError, match=r"unsupported device 'cuda'"):
        Device.resolve_all(["cpu", "cuda"], frozenset({"cpu"}))


def test_resolve_all_rejects_an_index_this_host_lacks():
    # The check a single spec cannot usefully make: naming a set is planning work
    # across it, so an absent index has to fail here rather than at the first
    # tensor move. The message names every missing index, sorted, and once each.
    beyond = torch.cuda.device_count() if torch.cuda.is_available() else 0
    specs = [f"cuda:{beyond + 1}", "cpu", f"cuda:{beyond}", f"cuda:{beyond + 1}"]

    with pytest.raises(ValueError, match=rf"no CUDA device at index {beyond}, "):
        Device.resolve_all(specs)


def test_resolve_all_of_cpus_never_asks_the_driver(monkeypatch):
    # A CPU-only run should not pay a driver query, so the bound check is reached
    # only when a CUDA device is actually named.
    def fail() -> bool:
        pytest.fail("the driver was queried for a CPU-only set")

    monkeypatch.setattr(torch.cuda, "is_available", fail)

    assert Device.resolve_all(["cpu", "cpu"]) == (Device("cpu"),) * 2


# ------------------------------ Device.visible_cuda --------------------------- #


def test_visible_cuda_devices_are_indexed_from_zero():
    devices = Device.visible_cuda()

    assert all(device.is_cuda for device in devices)
    assert [device.index for device in devices] == list(range(len(devices)))


def test_visible_cuda_devices_agree_with_the_driver():
    expected = torch.cuda.device_count() if torch.cuda.is_available() else 0

    assert len(Device.visible_cuda()) == expected


def test_counting_the_devices_does_not_initialize_cuda_here(monkeypatch):
    # `is_available()` initializes CUDA in whoever calls it, and a pool that
    # forks after that gives every worker a context it cannot use -- which is a
    # run that walks the whole dataset and fails every item of it. Planning the
    # devices happens before the pool starts, so the count has to come from the
    # driver query, which answers the same without initializing anything.
    def fail() -> bool:
        pytest.fail("`is_available` would initialize CUDA before the pool starts")

    count = torch.cuda.device_count()
    monkeypatch.setattr(torch.cuda, "is_available", fail)

    assert len(Device.visible_cuda()) == count


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA-capable GPU detected"
)
def test_every_visible_cuda_device_resolves():
    devices = Device.visible_cuda()

    assert Device.resolve_all(devices) == devices
