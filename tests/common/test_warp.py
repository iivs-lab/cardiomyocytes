from __future__ import annotations

import numpy as np
import pytest
import torch

from iivs_cardio.common.warp import BackwardWarp, backward_warp

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="no CUDA-capable GPU detected",
)


def _textured() -> torch.Tensor:
    # A smooth sinusoidal texture (structure in both axes), so the warp has real
    # structure to sample.
    y, x = np.mgrid[0:64, 0:64]
    tex = 128 + 60 * np.sin(2 * np.pi * x / 16) + 60 * np.sin(2 * np.pi * y / 16)
    return torch.as_tensor(tex.astype(np.uint8))


def _zero_transform(h: int = 64, w: int = 64) -> torch.Tensor:
    return torch.zeros((2, h, w), dtype=torch.float32)


def _uniform_transform(dx: float, dy: float, h: int = 64, w: int = 64) -> torch.Tensor:
    t = torch.zeros((2, h, w), dtype=torch.float32)
    t[0] = dx
    t[1] = dy
    return t


# ------------------------------ backward_warp --------------------------- #


def test_backward_warp_with_zero_transform_is_identity():
    img = _textured()
    warped = backward_warp(img, _zero_transform())
    assert warped.shape == img.shape
    assert warped.dtype == torch.uint8
    assert torch.equal(warped, img)


def test_backward_warp_recovers_known_shift():
    # A transform (dx, dy) = (3, 2) samples `image` at (x - 3, y - 2), so the
    # output equals np.roll(image, (2, 3)). Check the interior against an
    # independent np.roll.
    img = _textured()
    warped = backward_warp(img, _uniform_transform(3.0, 2.0)).numpy()
    expected = np.roll(img.numpy(), shift=(2, 3), axis=(0, 1))
    assert np.array_equal(warped[8:56, 8:56], expected[8:56, 8:56])


def test_backward_warp_respects_padding_mode():
    # A large transform pushes the sample point off-grid at the border; `zeros`
    # fills those out-of-bounds samples with 0, unlike `border` replication, so
    # the two disagree on the pulled-in edge.
    img = _textured()
    far = _uniform_transform(-40.0, 0.0)  # sample far to the right, past the edge
    zeros = backward_warp(img, far, padding_mode="zeros")
    border = backward_warp(img, far, padding_mode="border")
    assert zeros[:, -1].max().item() == 0  # right column sampled out of bounds
    assert not torch.equal(zeros, border)


def test_backward_warp_float_image_preserves_dtype_and_fraction():
    # A float image keeps its dtype and its fractional interpolated values (no
    # round/clamp). A half-pixel dx samples at x - 0.5, so each output pixel is
    # the mean of its column and the one to its left.
    img = _textured().to(torch.float32)
    out = backward_warp(img, _uniform_transform(0.5, 0.0))
    assert out.dtype == torch.float32
    expected = (img[:, :-1] + img[:, 1:]) / 2  # output columns 1..W-1
    assert torch.allclose(out[:, 1:], expected, atol=1e-3)


def test_backward_warp_int16_image_preserves_dtype():
    # A non-uint8 integer image goes through the round/clamp path (clamped to
    # int16's range) and keeps its dtype; an integer shift samples exactly.
    img = _textured().to(torch.int16)
    out = backward_warp(img, _uniform_transform(3.0, 2.0))
    assert out.dtype == torch.int16
    expected = torch.as_tensor(np.roll(img.numpy(), shift=(2, 3), axis=(0, 1)))
    assert torch.equal(out[8:56, 8:56], expected[8:56, 8:56])


def test_backward_warp_batched_matches_individual():
    # A batch warps to the same result as warping each pair on its own, keeping
    # the leading batch dim.
    img0 = _textured()
    img1 = torch.as_tensor(np.roll(img0.numpy(), shift=(5, 7), axis=(0, 1)))
    images = torch.stack([img0, img1])  # (2, H, W)
    transforms = torch.stack(
        [_uniform_transform(3.0, 2.0), _uniform_transform(-2.0, 4.0)]
    )

    batched = backward_warp(images, transforms)
    assert batched.shape == (2, 64, 64)
    assert batched.dtype == torch.uint8
    assert torch.equal(batched[0], backward_warp(img0, transforms[0]))
    assert torch.equal(batched[1], backward_warp(img1, transforms[1]))


def test_backward_warp_rejects_mismatched_batch():
    images = torch.zeros((2, 64, 64), dtype=torch.uint8)
    transforms = torch.zeros((3, 2, 64, 64), dtype=torch.float32)  # batch 3 != 2
    with pytest.raises(Exception, match=r"\[3,2,64,64\]"):
        backward_warp(images, transforms)


def test_backward_warp_rejects_wrong_transform_shape():
    bad = torch.zeros((3, 64, 64), dtype=torch.float32)  # channel dim must be 2
    with pytest.raises(Exception, match=r"\[3,64,64\]"):
        backward_warp(_textured(), bad)


def test_backward_warp_rejects_non_real_image():
    # `Real` excludes bool and complex: complex would silently drop its imaginary
    # part in `.float()` and bool has no `iinfo` range, so both are rejected at
    # the boundary rather than failing (or corrupting) deep inside.
    with pytest.raises(Exception, match=r"c64\[64,64\]"):
        backward_warp(torch.zeros((64, 64), dtype=torch.complex64), _zero_transform())
    with pytest.raises(Exception, match=r"bool\[64,64\]"):
        backward_warp(torch.zeros((64, 64), dtype=torch.bool), _zero_transform())


def test_backward_warp_rejects_non_float32_transform():
    # transform is pinned to float32 (the internal grid-math dtype); an integer
    # and even another float dtype are both rejected at the boundary.
    with pytest.raises(Exception, match=r"i32\[2,64,64\]"):
        backward_warp(_textured(), torch.zeros((2, 64, 64), dtype=torch.int32))
    with pytest.raises(Exception, match=r"f64\[2,64,64\]"):
        backward_warp(_textured(), torch.zeros((2, 64, 64), dtype=torch.float64))


def test_backward_warp_rejects_unknown_padding_mode():
    # padding_mode is a Literal, so beartype rejects an unlisted value.
    with pytest.raises(Exception, match="padding_mode"):
        backward_warp(_textured(), _zero_transform(), padding_mode="wrap")


@requires_cuda
def test_backward_warp_stays_on_cuda():
    # grid_sample only (no cuDNN), so the warp must stay on-device with no host
    # offload — cheap to run even on the box with broken torch cuDNN.
    img = _textured().cuda()
    transform = _uniform_transform(3.0, 2.0).cuda()
    assert backward_warp(img, transform).device.type == "cuda"


# ------------------------------- BackwardWarp --------------------------- #


def test_backward_warp_module_matches_function():
    img = _textured()
    transform = _uniform_transform(3.0, 2.0)
    assert torch.equal(BackwardWarp()(img, transform), backward_warp(img, transform))
    far = _uniform_transform(-40.0, 0.0)
    assert torch.equal(
        BackwardWarp(padding_mode="zeros")(img, far),
        backward_warp(img, far, padding_mode="zeros"),
    )


def test_backward_warp_module_batched_matches_function():
    images = torch.stack([_textured(), _textured().flip(0)])
    transforms = torch.stack(
        [_uniform_transform(3.0, 2.0), _uniform_transform(-2.0, 4.0)]
    )
    assert torch.equal(
        BackwardWarp()(images, transforms), backward_warp(images, transforms)
    )


def test_backward_warp_module_reuses_grid_across_calls(monkeypatch):
    # The coordinate grid depends only on (H, W, device), so warping many images
    # of one size must build it exactly once. Spy on torch.meshgrid to prove it.
    real_meshgrid = torch.meshgrid
    calls = 0

    def counting(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_meshgrid(*args, **kwargs)

    monkeypatch.setattr(torch, "meshgrid", counting)
    module = BackwardWarp()
    img = _textured()
    transform = _uniform_transform(3.0, 2.0)
    for _ in range(3):
        module(img, transform)
    assert calls == 1  # built once, reused across the 3 warps


def test_backward_warp_module_rebuilds_grid_on_size_change(monkeypatch):
    real_meshgrid = torch.meshgrid
    calls = 0

    def counting(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_meshgrid(*args, **kwargs)

    monkeypatch.setattr(torch, "meshgrid", counting)
    module = BackwardWarp()
    module(_textured(), _uniform_transform(1.0, 1.0))  # 64x64
    small = torch.zeros((32, 32), dtype=torch.uint8)
    out = module(small, torch.zeros((2, 32, 32), dtype=torch.float32))  # 32x32
    assert calls == 2  # size changed -> rebuilt once
    assert out.shape == (32, 32)


@requires_cuda
def test_backward_warp_module_stays_on_cuda():
    # The grid must be built on the input's device, else grid_sample would raise
    # on a device mismatch; a successful on-device result proves it.
    out = BackwardWarp()(_textured().cuda(), _uniform_transform(3.0, 2.0).cuda())
    assert out.device.type == "cuda"
