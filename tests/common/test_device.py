from __future__ import annotations

import pytest
import torch

from iivs_cardio.common import DEVICE_KINDS, resolve_device


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
