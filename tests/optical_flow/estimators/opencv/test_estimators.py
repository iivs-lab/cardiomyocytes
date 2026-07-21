from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from iivs_cardio.optical_flow.estimators import (
    DeepFlow,
    DualTVL1,
    Farneback,
    FarnebackParams,
)

# All three OpenCV methods run on CPU, so the streaming contract is tested
# GPU-free; the CUDA path is gated on an actual device below.
CPU_METHODS = (Farneback, DualTVL1, DeepFlow)
CUDA_METHODS = (Farneback, DualTVL1)

requires_cuda = pytest.mark.skipif(
    cv2.cuda.getCudaEnabledDeviceCount() < 1,
    reason="no CUDA-capable GPU detected",
)


def _textured_base() -> np.ndarray:
    # A smooth sinusoidal texture (structure in both axes) that both Farneback and
    # the (global, variational) DualTVL1 track reliably — random noise does not.
    y, x = np.mgrid[0:64, 0:64]
    texture = 128 + 60 * np.sin(2 * np.pi * x / 16) + 60 * np.sin(2 * np.pi * y / 16)
    return texture.astype(np.uint8)


def _frames(device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    # A textured frame and a copy shifted by (2, 3), giving the estimators real
    # motion to track. Deterministic.
    base = _textured_base()
    shifted = np.roll(base, shift=(2, 3), axis=(0, 1))
    return torch.as_tensor(base, device=device), torch.as_tensor(shifted, device=device)


def _assert_recovers_shift(flow: torch.Tensor) -> None:
    # `_frames` shifts by np.roll(base, (2, 3), (0, 1)), so the flow away from the
    # wrap-around borders should recover (dx, dy) = (3, 2) — an independent ground
    # truth that also proves the I/O path returns real values, not garbage.
    # flow is (2, H, W): channel 0 = dx, channel 1 = dy.
    interior = flow[:, 16:48, 16:48].cpu()
    assert interior[0].median().item() == pytest.approx(3.0, abs=0.5)
    assert interior[1].median().item() == pytest.approx(2.0, abs=0.5)


def _sequence(n: int, device: str = "cpu") -> torch.Tensor:
    # `n` consecutive frames as (n, H, W); each consecutive pair shifts by (2, 3).
    base = _textured_base()
    seq = np.stack([np.roll(base, shift=(2 * k, 3 * k), axis=(0, 1)) for k in range(n)])
    return torch.as_tensor(seq, device=device)


class _CountingAlgorithm:
    """A stand-in cv2 flow algorithm that counts `.calc` calls (returns zeros)."""

    def __init__(self):
        self.calls = 0

    def calc(self, i0, i1, flow=None):  # cv2 DenseOpticalFlow.calc(prev, curr, flow)
        self.calls += 1
        return np.zeros((*i0.shape, 2), np.float32)


@pytest.mark.parametrize("flow_cls", CPU_METHODS)
def test_push_streams_pairwise_flow(flow_cls):
    of = flow_cls(device="cpu")
    prev, curr = _frames()

    assert of.push(prev) is None  # first frame: no previous, no flow

    flow = of.push(curr)
    assert flow is not None
    assert flow.shape == (2, *prev.shape)
    assert flow.dtype == torch.float32
    assert flow.device.type == "cpu"


@pytest.mark.parametrize("flow_cls", CPU_METHODS)
def test_reset_restarts_the_sequence(flow_cls):
    of = flow_cls(device="cpu")
    prev, curr = _frames()
    of.push(prev)
    assert of.push(curr) is not None

    of.reset()
    assert of.push(prev) is None  # previous frame forgotten


@pytest.mark.parametrize("flow_cls", CPU_METHODS)
def test_calc_is_a_stateless_one_shot(flow_cls):
    of = flow_cls(device="cpu")
    prev, curr = _frames()

    flow = of.calc(prev, curr)
    assert flow.shape == (2, *prev.shape)
    assert flow.dtype == torch.float32
    # One-shot leaves no retained state, so a following push is still "first".
    assert of.push(prev) is None


@pytest.mark.parametrize("flow_cls", CPU_METHODS)
def test_push_result_survives_the_next_push(flow_cls):
    # The returned flow must own its memory: a later push (which may reuse an
    # internal output buffer) must not mutate an already-returned flow.
    of = flow_cls(device="cpu")
    a, b = _frames()
    of.push(a)
    first = of.push(b)
    assert first is not None
    kept = first.clone()
    of.push(a)  # a third push; first must be untouched
    assert torch.equal(first, kept)


@pytest.mark.parametrize("flow_cls", (Farneback, DualTVL1))
def test_recovers_known_shift(flow_cls):
    # Farneback and DualTVL1 recover a small translation precisely; DeepFlow is
    # checked separately below.
    of = flow_cls(device="cpu")
    prev, curr = _frames()
    _assert_recovers_shift(of.calc(prev, curr))


def test_deepflow_recovers_shift():
    # DeepFlow's output values (elsewhere only its shape/dtype are asserted): zero
    # for identical frames and the known (dx, dy) = (3, 2) shift recovered — a
    # real, motion-dependent result that a garbage/no-op calc would fail.
    of = DeepFlow(device="cpu")
    base, shifted = _frames()
    assert of.calc(base, base).abs().max().item() < 1e-3  # no motion -> zero flow
    _assert_recovers_shift(of.calc(base, shifted))


def test_push_rejects_wrong_dtype():
    of = Farneback(device="cpu")
    bad = torch.zeros((64, 64), dtype=torch.float32)  # not uint8
    with pytest.raises(Exception, match=r"f32\[64,64\]"):
        of.push(bad)


def test_push_rejects_wrong_shape():
    of = Farneback(device="cpu")
    bad = torch.zeros((3, 64, 64), dtype=torch.uint8)  # not (H, W)
    with pytest.raises(Exception, match=r"\[3,64,64\]"):
        of.push(bad)


def test_deepflow_rejects_cuda():
    with pytest.raises(ValueError, match="unsupported device 'cuda'"):
        DeepFlow(device="cuda")


def test_supported_devices():
    assert frozenset({"cpu", "cuda"}) == Farneback.SUPPORTED_DEVICES
    assert frozenset({"cpu"}) == DeepFlow.SUPPORTED_DEVICES


def test_custom_params_are_retained():
    of = Farneback(FarnebackParams(num_levels=1, win_size=7), device="cpu")
    assert of.params.num_levels == 1
    assert of.params.win_size == 7


@pytest.mark.parametrize("flow_cls", (Farneback, DualTVL1))
def test_calc_batch_matches_per_pair(flow_cls):
    of = flow_cls(device="cpu")
    rng = np.random.default_rng(1)
    prevs = torch.as_tensor(rng.integers(0, 256, size=(3, 64, 64), dtype=np.uint8))
    currs = torch.as_tensor(rng.integers(0, 256, size=(3, 64, 64), dtype=np.uint8))

    batch = of.calc_batch(prevs, currs)
    assert batch.shape == (3, 2, 64, 64)
    assert batch.dtype == torch.float32
    for i in range(3):
        assert torch.equal(batch[i], of.calc(prevs[i], currs[i]))


def test_calc_batch_calls_algorithm_once_per_pair(monkeypatch):
    # Contract: exactly one core `algorithm.calc` per source pair. Verified with a
    # spy, not just by values — a redundant re-compute would still match results.
    of = Farneback(device="cpu")
    spy = _CountingAlgorithm()
    monkeypatch.setattr(of, "_algorithm", spy)
    prevs = torch.zeros((3, 64, 64), dtype=torch.uint8)
    of.calc_batch(prevs, prevs)
    assert spy.calls == 3


def test_push_chunk_calls_algorithm_once_per_consecutive_pair(monkeypatch):
    # 5 frames -> 4 flows -> exactly 4 core calls (the first frame is only retained).
    of = Farneback(device="cpu")
    spy = _CountingAlgorithm()
    monkeypatch.setattr(of, "_algorithm", spy)
    of.push_chunk(_sequence(5))
    assert spy.calls == 4


def test_calc_batch_empty_returns_empty():
    empty = torch.zeros((0, 64, 64), dtype=torch.uint8)
    out = Farneback(device="cpu").calc_batch(empty, empty)
    assert out.shape == (0, 2, 64, 64)
    assert out.dtype == torch.float32


def test_calc_batch_rejects_mismatched_batch():
    of = Farneback(device="cpu")
    prev = torch.zeros((3, 64, 64), dtype=torch.uint8)
    curr = torch.zeros((4, 64, 64), dtype=torch.uint8)  # N mismatch vs prev
    with pytest.raises(Exception, match=r"u8\[4,64,64\]"):
        of.calc_batch(prev, curr)


def test_push_chunk_matches_individual_pushes():
    frames = _sequence(5)
    individual = []
    of1 = Farneback(device="cpu")
    for i in range(5):
        flow = of1.push(frames[i])
        if flow is not None:
            individual.append(flow)

    chunk = Farneback(device="cpu").push_chunk(frames)
    assert chunk.shape == (4, 2, 64, 64)  # first chunk: 5 frames -> 4 flows
    assert torch.equal(chunk, torch.stack(individual))


def test_push_chunk_continues_across_chunks():
    frames = _sequence(5)
    of = Farneback(device="cpu")
    first = of.push_chunk(frames[:3])  # first chunk: 3 frames -> 2 flows
    rest = of.push_chunk(
        frames[3:]
    )  # continues with retained prev: 2 frames -> 2 flows
    assert first.shape == (2, 2, 64, 64)
    assert rest.shape == (2, 2, 64, 64)

    of2 = Farneback(device="cpu")
    individual = []
    for i in range(5):
        flow = of2.push(frames[i])
        if flow is not None:
            individual.append(flow)
    assert torch.equal(torch.cat([first, rest]), torch.stack(individual))


def test_push_chunk_first_single_frame_retains_without_flow():
    of = Farneback(device="cpu")
    frames = _sequence(2)
    out = of.push_chunk(frames[:1])  # first chunk of 1 -> 0 flows, retains the frame
    assert out.shape == (0, 2, 64, 64)
    assert of.push(frames[1]) is not None  # a previous frame is now retained


def test_push_chunk_empty_returns_empty():
    empty = torch.zeros((0, 64, 64), dtype=torch.uint8)
    out = Farneback(device="cpu").push_chunk(empty)
    assert out.shape == (0, 2, 64, 64)


@requires_cuda
@pytest.mark.parametrize("flow_cls", CUDA_METHODS)
def test_cuda_push_stays_on_device(flow_cls):
    of = flow_cls(device="cuda")
    prev, curr = _frames(device="cuda")

    assert of.push(prev) is None
    flow = of.push(curr)
    assert flow is not None
    assert flow.shape == (2, *prev.shape)
    assert flow.dtype == torch.float32
    assert flow.device.type == "cuda"  # device-resident, no host round trip


@requires_cuda
@pytest.mark.parametrize("flow_cls", CUDA_METHODS)
def test_cuda_recovers_known_shift(flow_cls):
    # The CUDA path (cupy <-> GpuMat) must produce a correct flow, verified
    # against the known frame shift rather than the (differently-implemented)
    # CPU result.
    prev, curr = _frames(device="cuda")
    flow = flow_cls(device="cuda").calc(prev, curr)
    assert flow.device.type == "cuda"
    _assert_recovers_shift(flow)


@requires_cuda
@pytest.mark.parametrize("flow_cls", CUDA_METHODS)
def test_cuda_reset_restarts_the_sequence(flow_cls):
    of = flow_cls(device="cuda")
    prev, curr = _frames(device="cuda")
    of.push(prev)
    assert of.push(curr) is not None

    of.reset()
    assert of.push(prev) is None


@requires_cuda
def test_cuda_push_streams_without_corrupting_retained_flows():
    # The CUDA push copies frames into an alternating double-buffer and reuses
    # the flow buffer, so a retained early flow must survive later pushes.
    of = Farneback(device="cuda")
    frames = _sequence(4, device="cuda")

    flows = []
    for i in range(4):
        flow = of.push(frames[i])
        if flow is not None:
            flows.append(flow)
    assert len(flows) == 3  # 4 frames -> 3 flows

    # every consecutive pair shifts by (3, 2); the first flow must still hold
    # after the later pushes reused the buffers.
    _assert_recovers_shift(flows[0])
    _assert_recovers_shift(flows[-1])


@requires_cuda
def test_cuda_push_chunk_streams():
    of = Farneback(device="cuda")
    chunk = of.push_chunk(_sequence(5, device="cuda"))
    assert chunk.shape == (4, 2, 64, 64)  # 5 frames -> 4 flows
    assert chunk.device.type == "cuda"
    _assert_recovers_shift(chunk[0])  # each pair shifts by (3, 2)


@requires_cuda
def test_push_rejects_tensor_on_wrong_device():
    of = Farneback(device="cuda")
    cpu_frame = torch.zeros((64, 64), dtype=torch.uint8)  # on cpu, estimator on cuda
    with pytest.raises(ValueError, match="expects a cuda:0 tensor"):
        of.push(cpu_frame)
