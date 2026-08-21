from __future__ import annotations

import logging
import weakref
from typing import TYPE_CHECKING, override

import cv2
import numpy as np
import pytest
import torch

from iivs_cardio.common import Device
from iivs_cardio.optical_flow.estimators import (
    DeepFlowConfig,
    DualTVL1Config,
    EstimatorConfig,
    FarnebackConfig,
    OpenCVConfig,
    OpenCVEstimator,
)

if TYPE_CHECKING:
    from iivs_cardio.optical_flow.estimators.opencv.base import DenseAlgorithm

# All three OpenCV methods run on CPU, so the streaming contract is tested
# GPU-free; the CUDA path is gated on an actual device below. Each is named by
# its config, which is the only thing that differs between them now.
CPU_METHODS = (FarnebackConfig, DualTVL1Config, DeepFlowConfig)
CUDA_METHODS = (FarnebackConfig, DualTVL1Config)

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
    of = flow_cls().build("cpu")
    prev, curr = _frames()

    assert of.push(prev) is None  # first frame: no previous, no flow

    flow = of.push(curr)
    assert flow is not None
    assert flow.shape == (2, *prev.shape)
    assert flow.dtype == torch.float32
    assert flow.device.type == "cpu"


@pytest.mark.parametrize("flow_cls", CPU_METHODS)
def test_reset_restarts_the_sequence(flow_cls):
    of = flow_cls().build("cpu")
    prev, curr = _frames()
    of.push(prev)
    assert of.push(curr) is not None

    of.reset()
    assert of.push(prev) is None  # previous frame forgotten


@pytest.mark.parametrize("flow_cls", CPU_METHODS)
def test_calc_is_a_stateless_one_shot(flow_cls):
    of = flow_cls().build("cpu")
    prev, curr = _frames()

    flow = of.calc(prev, curr)
    assert flow.shape == (2, *prev.shape)
    assert flow.dtype == torch.float32
    # One-shot leaves no retained state, so a following push is still "first".
    assert of.push(prev) is None


@pytest.mark.parametrize("flow_cls", CPU_METHODS)
def test_push_does_not_alias_the_callers_frame(flow_cls):
    # Streaming into one reusable buffer is the ordinary shape of frame IO, and
    # a retained reference makes the next read overwrite the retained frame:
    # `prev` and `curr` become one picture and the motion goes silently to zero.
    of = flow_cls().build("cpu")
    first, second = _frames()
    buffer = torch.zeros_like(first)

    buffer.copy_(first)
    of.push(buffer)
    buffer.copy_(second)

    flow = of.push(buffer)
    assert flow is not None
    assert flow.abs().max().item() > 0.5  # zero flow is what aliasing produces


def test_push_chunk_retains_one_frame_rather_than_the_chunk():
    # `for frame in frames` yields views, so retaining the last one would keep
    # the whole batch alive where the contract is one frame however long the
    # chunk. Asked of the batch's storage rather than of what was retained,
    # which is the estimator's own business.
    of = FarnebackConfig().build("cpu")
    frames = _sequence(8)
    batch = weakref.ref(frames.untyped_storage())

    of.push_chunk(frames)
    del frames

    assert batch() is None


@pytest.mark.parametrize("flow_cls", CPU_METHODS)
def test_push_result_survives_the_next_push(flow_cls):
    # The returned flow must own its memory: a later push (which may reuse an
    # internal output buffer) must not mutate an already-returned flow.
    of = flow_cls().build("cpu")
    a, b = _frames()
    of.push(a)
    first = of.push(b)
    assert first is not None
    kept = first.clone()
    of.push(a)  # a third push; first must be untouched
    assert torch.equal(first, kept)


@pytest.mark.parametrize("flow_cls", CUDA_METHODS)
def test_recovers_known_shift(flow_cls):
    # Farneback and DualTVL1 recover a small translation precisely; DeepFlow is
    # checked separately below.
    of = flow_cls().build("cpu")
    prev, curr = _frames()
    _assert_recovers_shift(of.calc(prev, curr))


def test_deepflow_recovers_shift():
    # DeepFlow's output values (elsewhere only its shape/dtype are asserted): zero
    # for identical frames and the known (dx, dy) = (3, 2) shift recovered — a
    # real, motion-dependent result that a garbage/no-op calc would fail.
    of = DeepFlowConfig().build("cpu")
    base, shifted = _frames()
    assert of.calc(base, base).abs().max().item() < 1e-3  # no motion -> zero flow
    _assert_recovers_shift(of.calc(base, shifted))


def test_push_rejects_wrong_dtype():
    of = FarnebackConfig().build("cpu")
    bad = torch.zeros((64, 64), dtype=torch.float32)  # not uint8
    with pytest.raises(Exception, match=r"f32\[64,64\]"):
        of.push(bad)


def test_push_rejects_wrong_shape():
    of = FarnebackConfig().build("cpu")
    bad = torch.zeros((3, 64, 64), dtype=torch.uint8)  # not (H, W)
    with pytest.raises(Exception, match=r"\[3,64,64\]"):
        of.push(bad)


class _MisreadingConfig(OpenCVConfig):
    """A config whose `_create` answers for a device other than the one it took."""

    def __init__(self, factory) -> None:
        self._factory = factory

    @override
    def _create(self, device: Device) -> DenseAlgorithm:
        return self._factory()


@requires_cuda
@pytest.mark.parametrize(
    ("factory", "device", "made"),
    (
        pytest.param(cv2.FarnebackOpticalFlow.create, "cuda", "cpu", id="cpu-for-cuda"),
        pytest.param(
            cv2.cuda.FarnebackOpticalFlow.create, "cpu", "cuda", id="cuda-for-cpu"
        ),
    ),
)
def test_a_config_creating_for_the_wrong_device_is_refused(factory, device, made):
    # What is left to catch once `SUPPORTED_DEVICES` has run and the backends
    # take the concrete cv2 types: a `_create` that read its device wrongly. A
    # CPU algorithm run as CUDA would be handed device tensors and read as host
    # memory.
    with pytest.raises(ValueError, match=f"made for {made}"):
        _MisreadingConfig(factory).build(device)


@pytest.mark.parametrize("flow_cls", CPU_METHODS)
def test_an_estimator_says_which_device_it_runs_on(flow_cls):
    # Nothing inside branches on this any more, the backend being chosen once,
    # so it is the caller's question alone and only a test keeps it answered.
    of = flow_cls().build("cpu")

    assert of.device == Device("cpu")
    assert of.is_cuda is False


@requires_cuda
@pytest.mark.parametrize("flow_cls", CUDA_METHODS)
def test_a_cuda_estimator_says_so(flow_cls):
    of = flow_cls().build("cuda")

    assert of.device == Device("cuda", 0)
    assert of.is_cuda is True


def test_deepflow_rejects_cuda():
    with pytest.raises(ValueError, match="unsupported device 'cuda'"):
        DeepFlowConfig().build("cuda")


def test_deepflow_refuses_cuda_below_the_policy_too():
    # `build` never reaches the guard, `SUPPORTED_DEVICES` refusing first, so
    # nothing else covers it. What it answers is a caller reaching past `build`,
    # who would otherwise hold a CPU algorithm paired with a CUDA device.
    with pytest.raises(ValueError, match="DeepFlow"):
        DeepFlowConfig()._create(Device("cuda"))  # noqa: SLF001


def test_supported_devices():
    # Declared by the algorithm's config, not by the estimator: OpenCV shipping
    # no CUDA DeepFlow is a fact about DeepFlow.
    assert frozenset({"cpu", "cuda"}) == FarnebackConfig.SUPPORTED_DEVICES
    assert frozenset({"cpu"}) == DeepFlowConfig.SUPPORTED_DEVICES


def test_tvl1_says_which_settings_the_device_will_not_read(caplog):
    # cv2 takes `iterations` only in its CUDA build and reports nothing about a
    # setting it ignored, so a CPU sweep over it would run to the end and find
    # no difference. Named rather than refused: a config carries both devices'
    # settings, so having them is not the same as having meant them.
    with caplog.at_level(logging.WARNING):
        DualTVL1Config(iterations=1000).build("cpu")

    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "iterations" in message
    assert "cuda" in message  # where it would have been read


@pytest.mark.parametrize(
    "config",
    (
        pytest.param(DualTVL1Config(), id="defaults"),
        pytest.param(DualTVL1Config(inner_iterations=7), id="cpu-setting-on-cpu"),
        pytest.param(DualTVL1Config(lambda_=0.1), id="setting-both-devices-read"),
    ),
)
def test_tvl1_is_quiet_about_settings_the_device_does_read(caplog, config):
    # Both sets sit at their defaults in every config, so warning about the
    # untouched ones would fire on every build and say nothing.
    with caplog.at_level(logging.WARNING):
        config.build("cpu")

    assert caplog.records == []


@requires_cuda
def test_tvl1_says_so_on_cuda_too(caplog):
    with caplog.at_level(logging.WARNING):
        DualTVL1Config(median_filtering=3, outer_iterations=9).build("cuda")

    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "median_filtering" in message
    assert "outer_iterations" in message
    assert "inner_iterations" not in message  # untouched, so nothing was meant


def test_custom_params_reach_the_algorithm():
    # Asked of cv2 rather than of a config the estimator kept: what matters is
    # that the settings arrived at the thing that computes, and an estimator
    # holding a copy of them would pass whether they did or not.
    of = FarnebackConfig(num_levels=1, win_size=7).build("cpu")

    assert of.algorithm.getNumLevels() == 1
    assert of.algorithm.getWinSize() == 7


@pytest.mark.parametrize(
    "config",
    (
        pytest.param(FarnebackConfig(win_size=21), id="farneback"),
        pytest.param(DualTVL1Config(nscales=5), id="dualtvl1"),
        pytest.param(DeepFlowConfig(), id="deepflow"),
    ),
)
def test_params_build_their_estimator_on_the_given_device(config):
    # The `EstimatorConfig.build` recipe a pool worker uses: a picklable config
    # object reconstructs the (unpicklable) estimator on the worker's device.
    # Every algorithm arrives as the same estimator, holding a different cv2
    # object, which is what taking the algorithm as a value rather than a
    # subclass buys.
    assert isinstance(config, EstimatorConfig)

    estimator = config.build("cpu")
    assert isinstance(estimator, OpenCVEstimator)
    assert estimator.device == Device("cpu")


def test_build_forwards_the_held_params_of_every_algorithm():
    # `build` must pass its own fields through, not defaults -- the one step that
    # turns a stored config back into a configured algorithm.
    farneback = FarnebackConfig(num_levels=1, win_size=7).build("cpu")
    tvl1 = DualTVL1Config(nscales=5, warps=2).build("cpu")

    assert (farneback.algorithm.getNumLevels(), farneback.algorithm.getWinSize()) == (
        1,
        7,
    )
    assert (tvl1.algorithm.getScalesNumber(), tvl1.algorithm.getWarpingsNumber()) == (
        5,
        2,
    )


def test_estimator_params_pickle_across_a_process_boundary():
    # Why the recipe is config, not a live estimator: the estimator holds a `cv2`
    # object that does not pickle, but its config do -- so they cross to a worker.
    import pickle

    for config in (FarnebackConfig(win_size=21), DualTVL1Config(), DeepFlowConfig()):
        assert pickle.loads(pickle.dumps(config)) == config  # noqa: S301


@pytest.mark.parametrize("flow_cls", CUDA_METHODS)
def test_calc_batch_matches_per_pair(flow_cls):
    of = flow_cls().build("cpu")
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
    of = FarnebackConfig().build("cpu")
    spy = _CountingAlgorithm()
    monkeypatch.setattr(of._backend, "algorithm", spy)  # noqa: SLF001
    prevs = torch.zeros((3, 64, 64), dtype=torch.uint8)
    of.calc_batch(prevs, prevs)
    assert spy.calls == 3


def test_push_chunk_calls_algorithm_once_per_consecutive_pair(monkeypatch):
    # 5 frames -> 4 flows -> exactly 4 core calls (the first frame is only retained).
    of = FarnebackConfig().build("cpu")
    spy = _CountingAlgorithm()
    monkeypatch.setattr(of._backend, "algorithm", spy)  # noqa: SLF001
    of.push_chunk(_sequence(5))
    assert spy.calls == 4


def test_calc_batch_empty_returns_empty():
    empty = torch.zeros((0, 64, 64), dtype=torch.uint8)
    out = FarnebackConfig().build("cpu").calc_batch(empty, empty)
    assert out.shape == (0, 2, 64, 64)
    assert out.dtype == torch.float32


def test_calc_batch_rejects_mismatched_batch():
    of = FarnebackConfig().build("cpu")
    prev = torch.zeros((3, 64, 64), dtype=torch.uint8)
    curr = torch.zeros((4, 64, 64), dtype=torch.uint8)  # N mismatch vs prev
    with pytest.raises(Exception, match=r"u8\[4,64,64\]"):
        of.calc_batch(prev, curr)


def test_push_chunk_matches_individual_pushes():
    frames = _sequence(5)
    individual = []
    of1 = FarnebackConfig().build("cpu")
    for i in range(5):
        flow = of1.push(frames[i])
        if flow is not None:
            individual.append(flow)

    chunk = FarnebackConfig().build("cpu").push_chunk(frames)
    assert chunk.shape == (4, 2, 64, 64)  # first chunk: 5 frames -> 4 flows
    assert torch.equal(chunk, torch.stack(individual))


def test_push_chunk_continues_across_chunks():
    frames = _sequence(5)
    of = FarnebackConfig().build("cpu")
    first = of.push_chunk(frames[:3])  # first chunk: 3 frames -> 2 flows
    rest = of.push_chunk(
        frames[3:]
    )  # continues with retained prev: 2 frames -> 2 flows
    assert first.shape == (2, 2, 64, 64)
    assert rest.shape == (2, 2, 64, 64)

    of2 = FarnebackConfig().build("cpu")
    individual = []
    for i in range(5):
        flow = of2.push(frames[i])
        if flow is not None:
            individual.append(flow)
    assert torch.equal(torch.cat([first, rest]), torch.stack(individual))


def test_push_chunk_first_single_frame_retains_without_flow():
    of = FarnebackConfig().build("cpu")
    frames = _sequence(2)
    out = of.push_chunk(frames[:1])  # first chunk of 1 -> 0 flows, retains the frame
    assert out.shape == (0, 2, 64, 64)
    assert of.push(frames[1]) is not None  # a previous frame is now retained


def test_push_chunk_empty_returns_empty():
    empty = torch.zeros((0, 64, 64), dtype=torch.uint8)
    out = FarnebackConfig().build("cpu").push_chunk(empty)
    assert out.shape == (0, 2, 64, 64)


@pytest.mark.parametrize("flow_cls", CPU_METHODS)
@pytest.mark.parametrize("chunks", ((4,), (1, 3)))
def test_a_chunk_of_flows_holds_no_more_than_it_returns(flow_cls, chunks):
    # A batch sized to the frames and sliced down to the flows would keep the
    # whole allocation alive behind the view, which `torch.save` writes out in
    # full. Both the first chunk (N - 1 flows) and a later one (N) are sized.
    of = flow_cls().build("cpu")
    frames = _sequence(sum(chunks))

    start = 0
    for length in chunks:
        flows = of.push_chunk(frames[start : start + length])
        held = flows.untyped_storage().size() // flows.element_size()
        assert held == flows.numel()
        start += length


def test_a_backend_says_whether_a_frame_is_retained():
    of = FarnebackConfig().build("cpu")
    backend = of._backend  # noqa: SLF001
    frames = _sequence(2)

    assert not backend.retained
    of.push(frames[0])
    assert backend.retained
    of.reset()
    assert not backend.retained


def test_push_writes_the_flow_into_a_given_destination():
    of = FarnebackConfig().build("cpu")
    backend = of._backend  # noqa: SLF001
    frames = _sequence(2)
    untouched = torch.full((2, 64, 64), -99.0)
    destination = untouched.clone()

    assert backend.push(frames[0], out=destination) is None
    assert torch.equal(destination, untouched)  # nothing retained, nothing written

    assert backend.push(frames[1], out=destination) is destination
    _assert_recovers_shift(destination)


@requires_cuda
@pytest.mark.parametrize("flow_cls", CUDA_METHODS)
def test_cuda_push_stays_on_device(flow_cls):
    of = flow_cls().build("cuda")
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
    flow = flow_cls().build("cuda").calc(prev, curr)
    assert flow.device.type == "cuda"
    _assert_recovers_shift(flow)


@requires_cuda
@pytest.mark.parametrize("flow_cls", CUDA_METHODS)
def test_cuda_reset_restarts_the_sequence(flow_cls):
    of = flow_cls().build("cuda")
    prev, curr = _frames(device="cuda")
    of.push(prev)
    assert of.push(curr) is not None

    of.reset()
    assert of.push(prev) is None


@requires_cuda
def test_cuda_push_streams_without_corrupting_retained_flows():
    # The CUDA push copies frames into an alternating double-buffer and reuses
    # the flow buffer, so a retained early flow must survive later pushes.
    of = FarnebackConfig().build("cuda")
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
    of = FarnebackConfig().build("cuda")
    chunk = of.push_chunk(_sequence(5, device="cuda"))
    assert chunk.shape == (4, 2, 64, 64)  # 5 frames -> 4 flows
    assert chunk.device.type == "cuda"
    _assert_recovers_shift(chunk[0])  # each pair shifts by (3, 2)


@requires_cuda
def test_cuda_reset_does_not_let_the_last_sequence_through():
    # `reset` keeps its buffers rather than dropping them, so the frame one
    # sequence leaves behind must not reach the first flow of the next.
    of = FarnebackConfig().build("cuda")
    frames = _sequence(3, device="cuda")
    of.push_chunk(frames)
    of.reset()

    assert of.push(frames[0]) is None
    after_reset = of.push(frames[1])

    fresh = FarnebackConfig().build("cuda")
    fresh.push(frames[0])
    assert torch.equal(after_reset, fresh.push(frames[1]))


@requires_cuda
def test_push_rejects_tensor_on_wrong_device():
    of = FarnebackConfig().build("cuda")
    cpu_frame = torch.zeros((64, 64), dtype=torch.uint8)  # on cpu, estimator on cuda
    with pytest.raises(ValueError, match="expects a cuda:0 tensor"):
        of.push(cpu_frame)
