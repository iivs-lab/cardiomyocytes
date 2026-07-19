# Optical Flow 모듈 설계 (핸드오프)

> 다른 컴퓨터에서 이어서 작업하기 위한 설계 스냅샷. 이 문서 + `AGENTS.md`
> 만 읽으면 바로 구현에 착수할 수 있도록 확정된 API 사실·클래스 설계·미결
> 정을 모두 담는다. 대상 모듈: **`optical_flow`** (계산 + 평가).
>
> 관련: [`filter-3d-design.md`](filter-3d-design.md) (3D 필터). **`Device`
> 값 객체는 두 모듈이 공유**한다(아래).

## 목표

Off-axis DHM/QPI phase 시퀀스에서 심근세포 dense optical flow를 계산·평가하는
재사용 클래스. 레거시 `iivs-lab/cardio-force-legacy`의 `Python/calc_optflows.py`
(OpenCV 고전 flow)를 현대화하되, **CUDA 백엔드(+ device 지정)** 와 **stateful
push API**를 추가한다. 학습형(RAFT류)은 레거시에 없던 새 확장 여지로 남겨 둔다
(현재 범위 밖).

- **데이터 시퀀스 읽기·순회는 iivs-lib 0.2.0**에서 온다 (의존성 추가 후 import).
  flow 계산·평가·정규화·3D 필터는 이 프로젝트가 소유.
- 정규화된 **uint8 단일 채널** 프레임을 입력으로 받는다(정규화는 상류 책임).

## 확정된 OpenCV API 사실 (재조사 불필요)

프로젝트 `.venv`(OpenCV 4.13.0, CUDA build)에서 실측 확인:

| 방식 | CPU | CUDA | 비고 |
|---|---|---|---|
| Farneback | `cv2.FarnebackOpticalFlow.create` | `cv2.cuda.FarnebackOpticalFlow.create` | **create 시그니처 동일**: `numLevels, pyrScale, fastPyramids, winSize, numIters, polyN, polySigma, flags` |
| Dual TV-L1 | `cv2.optflow.DualTVL1OpticalFlow.create` | `cv2.cuda.OpticalFlowDual_TVL1.create` | 공통: `tau, lambda_, theta, nscales, warps, epsilon, scaleStep, gamma`. **CPU만**: `innnerIterations`(OpenCV 오타·n 3개), `outerIterations`, `medianFiltering`. **CUDA만**: 단일 `iterations` |
| DeepFlow | `cv2.optflow.createOptFlow_DeepFlow()` | 없음 | **CPU 전용·튜닝 파라미터 없음** |

- 모든 방식 `.calc(I0, I1, flow[, stream])` → `(H, W, 2) float32` 반환(실측).
- CUDA `.calc`는 입력·출력이 `cv2.cuda.GpuMat`(8UC1 → CV_32FC2).
- 지원 매트릭스(kind): **Farneback(cpu/cuda), DualTVL1(cpu/cuda), DeepFlow(cpu만)**.
- **CUDA device 선택**: `cv2.cuda.setDevice(index)` — **프로세스 전역 setter**(per-op
  아님). GpuMat 할당·calc 전에 호출해야 그 GPU에서 수행된다. torch의 per-op
  device(`.to("cuda:1")`, filter_3d에서 사용)와 대비되지만, 사용자 API는 동일한
  `Device`로 통일한다.

## 설계 결정 (확정)

1. **device 은닉 + stateful push**: 시퀀스는 `push(frame)` 단일 프레임 스트리밍.
   추정기가 직전 프레임을 보유하고, CUDA는 프레임을 프레임당 1회만 업로드(직전
   프레임 디바이스 상주 → 재업로드 0), 프레임 더블 버퍼 + flow 출력 버퍼 재사용.
2. **지원 = kind, 선택 = index**: `supported_devices`(kind 집합) 클래스 속성 + 생성자
   검증. "지원하느냐"는 kind 문제(DeepFlow는 cuda 불가), "어느 GPU냐"는 index
   문제. 잘못된 kind는 유효 집합을 명시하는 에러(예: DeepFlow+cuda).
3. **파라미터 = frozen dataclass**: 기본값은 레거시 실험값. DualTVL1의 CPU/CUDA
   분기 파라미터는 한 dataclass에 두되 어느 백엔드에서 무시되는지 명시.
4. **정규화는 flow 클래스 밖**: 4-mode 정규화(per-frame/pairwise/sequence/dataset)는
   시퀀스 통계에 의존 → 전처리 책임. flow 클래스는 uint8만 받는다.
5. **평가 = warp 일치도**: GT flow 없음 → prev를 flow로 워핑해 curr와 비교
   (L1/L2/SSIM/PSNR). `score`(쌍별) + `MetricsAccumulator`(스트리밍 평균) 분리.

## 공통: `Device` 값 객체 (cpu / cuda:N)

`optical_flow`·`filter_3d`가 공유하는 device 지정 타입. **torch 스타일 문자열**
(`"cpu"`, `"cuda"`, `"cuda:0"`, `"cuda:1"`)을 받는다. 배치 권장:
`iivs_cardio/common/device.py`.

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class Device:
    kind: Literal["cpu", "cuda"]
    index: int | None = None  # None => 해당 kind의 기본 device

    @classmethod
    def parse(cls, spec: str | Device) -> Device:
        if isinstance(spec, Device):
            return spec
        if spec == "cpu":
            return cls("cpu")
        if spec == "cuda":
            return cls("cuda", 0)
        if isinstance(spec, str) and spec.startswith("cuda:"):
            return cls("cuda", int(spec[5:]))
        msg = f"invalid device {spec!r}: expected 'cpu', 'cuda', or 'cuda:N'"
        raise ValueError(msg)

    @property
    def is_cuda(self) -> bool:
        return self.kind == "cuda"

    def as_torch(self) -> str:  # filter_3d(torch)용
        return self.kind if self.index is None else f"{self.kind}:{self.index}"
```

- **flow(cv2.cuda)**: `cv2.cuda.setDevice(device.index)`(전역) — GpuMat 할당·calc 전.
- **filter(torch)**: `tensor.to(device.as_torch())`(per-op).
- cv2.cuda의 전역성은 멀티 GPU 데이터 병렬(프로세스당 GPU 1개 고정)과 잘 맞는다.

## 클래스 설계 — 계산

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import cv2
import numpy as np

from iivs_cardio.common.device import Device

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class FarnebackParams:
    num_levels: int = 3
    pyr_scale: float = 0.5
    fast_pyramids: bool = False
    win_size: int = 15
    num_iters: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    flags: int = 0


@dataclass(frozen=True, slots=True)
class DualTVL1Params:
    tau: float = 0.25
    lambda_: float = 0.05
    theta: float = 0.3
    nscales: int = 5
    warps: int = 5
    epsilon: float = 0.01
    scale_step: float = 0.8
    gamma: float = 0.0
    # CPU 전용 (CUDA에서 무시):
    inner_iterations: int = 30
    outer_iterations: int = 10
    median_filtering: int = 5
    # CUDA 전용 (CPU에서 무시):
    iterations: int = 300


class OpticalFlow(ABC):
    """단일 프레임을 순차로 밀어 넣는 stateful dense optical flow 추정기.

    `push(frame)`는 보유한 직전 프레임과 현재 프레임 사이 flow를 계산하고,
    현재 프레임을 다음 호출의 직전 프레임으로 보관한다. 첫 호출은 직전
    프레임이 없어 `None`을 반환한다(N 프레임 → N−1 flow). `reset()`으로
    새 시퀀스를 시작한다.

    CUDA device는 프레임당 1회만 업로드하고(직전 프레임 디바이스 상주),
    프레임 더블 버퍼와 flow 출력 버퍼를 재사용한다. 입력은 정규화된 uint8
    단일 채널이어야 한다(정규화는 상류 책임).
    """

    method: ClassVar[str]
    supported_devices: ClassVar[frozenset[str]]  # kind만: {"cpu"} 또는 {"cpu", "cuda"}

    def __init__(self, device: str | Device = "cpu") -> None:
        device = Device.parse(device)
        if device.kind not in self.supported_devices:
            ok = ", ".join(sorted(self.supported_devices))
            msg = (
                f"{self.method!r} does not support the {device.kind!r} backend; "
                f"use one of: {ok}"
            )
            raise ValueError(msg)
        self.device = device

        self._activate()  # cuda면 setDevice(index) — 빌드·할당 전에
        self._impl = self._build()

        self._prev = None
        if device.is_cuda:
            self._buffers = (cv2.cuda.GpuMat(), cv2.cuda.GpuMat())  # 프레임 더블 버퍼
            self._flow_buf = cv2.cuda.GpuMat()                     # 재사용 출력 버퍼
            self._slot = 0

    def _activate(self) -> None:
        # cv2.cuda의 device는 전역 상태 → 이 추정기의 GPU로 맞춘다.
        # 한 프로세스에서 여러 GPU 추정기를 섞어도 안전하도록 op 직전 재설정.
        if self.device.is_cuda:
            cv2.cuda.setDevice(self.device.index)

    @abstractmethod
    def _build(self):
        """선택된 `self.device`용 OpenCV 추정기를 생성한다."""

    def reset(self) -> None:
        """보유한 직전 프레임을 비워 새 시퀀스를 시작한다."""
        self._prev = None
        if self.device.is_cuda:
            self._slot = 0

    def push(
        self, frame: NDArray[np.uint8], *, download: bool = True
    ) -> NDArray[np.float32] | None:
        """직전 프레임 → `frame` 의 flow. 첫 호출은 `None`.

        `download=False`(CUDA)면 재사용 `GpuMat`을 반환하며 다음 `push`
        전까지만 유효하다 — 보관하려면 `.clone()`.
        """
        if self.device.is_cuda:
            return self._push_cuda(frame, download=download)
        return self._push_cpu(frame)

    def _push_cpu(self, frame):
        flow = None if self._prev is None else self._impl.calc(self._prev, frame, None)
        self._prev = frame  # 주의: CPU는 참조 저장 — 소스가 버퍼 재사용이면 .copy() 필요
        return flow

    def _push_cuda(self, frame, *, download):
        self._activate()        # 전역 device 상태 재확인 후 진행
        curr = self._buffers[self._slot]
        curr.upload(frame)      # 프레임당 업로드 정확히 1회
        self._slot ^= 1         # 다음 업로드는 반대 버퍼 → prev는 절대 안 덮임
        if self._prev is None:
            self._prev = curr
            return None
        flow = self._impl.calc(self._prev, curr, self._flow_buf)  # 출력 버퍼 재사용
        self._prev = curr       # 롤: 데이터 이동 없이 참조 승계
        return flow.download() if download else flow

    def calc(
        self, prev: NDArray[np.uint8], curr: NDArray[np.uint8]
    ) -> NDArray[np.float32]:
        """원샷 편의 (stateful 경로와 독립; 테스트/단발용)."""
        if self.device.is_cuda:
            self._activate()
            g_prev = cv2.cuda.GpuMat(); g_prev.upload(prev)
            g_curr = cv2.cuda.GpuMat(); g_curr.upload(curr)
            return self._impl.calc(g_prev, g_curr, None).download()
        return self._impl.calc(prev, curr, None)


class FarnebackFlow(OpticalFlow):
    method = "farneback"
    supported_devices = frozenset({"cpu", "cuda"})

    def __init__(
        self,
        params: FarnebackParams | None = None,
        *,
        device: str | Device = "cpu",
    ) -> None:
        self.params = params or FarnebackParams()
        super().__init__(device)

    def _build(self):
        p = self.params
        factory = (  # CPU·CUDA create 시그니처 동일 → 팩토리만 교체
            cv2.cuda.FarnebackOpticalFlow
            if self.device.is_cuda
            else cv2.FarnebackOpticalFlow
        )
        return factory.create(
            numLevels=p.num_levels, pyrScale=p.pyr_scale, fastPyramids=p.fast_pyramids,
            winSize=p.win_size, numIters=p.num_iters, polyN=p.poly_n,
            polySigma=p.poly_sigma, flags=p.flags,
        )


class DualTVL1Flow(OpticalFlow):
    method = "tvl1"
    supported_devices = frozenset({"cpu", "cuda"})

    def __init__(
        self,
        params: DualTVL1Params | None = None,
        *,
        device: str | Device = "cpu",
    ) -> None:
        self.params = params or DualTVL1Params()
        super().__init__(device)

    def _build(self):
        p = self.params
        if self.device.is_cuda:
            return cv2.cuda.OpticalFlowDual_TVL1.create(
                tau=p.tau, lambda_=p.lambda_, theta=p.theta, nscales=p.nscales,
                warps=p.warps, epsilon=p.epsilon, iterations=p.iterations,
                scaleStep=p.scale_step, gamma=p.gamma, useInitialFlow=False,
            )
        return cv2.optflow.DualTVL1OpticalFlow.create(
            tau=p.tau, lambda_=p.lambda_, theta=p.theta, nscales=p.nscales,
            warps=p.warps, epsilon=p.epsilon,
            innnerIterations=p.inner_iterations,  # OpenCV 파라미터명 오타(n 3개) 그대로
            outerIterations=p.outer_iterations, scaleStep=p.scale_step,
            gamma=p.gamma, medianFiltering=p.median_filtering, useInitialFlow=False,
        )


class DeepFlow(OpticalFlow):
    method = "deepflow"
    supported_devices = frozenset({"cpu"})  # CPU 전용

    def _build(self):
        return cv2.optflow.createOptFlow_DeepFlow()  # 튜닝 파라미터 없음


_REGISTRY: dict[str, type[OpticalFlow]] = {
    "farneback": FarnebackFlow,
    "tvl1": DualTVL1Flow,
    "deepflow": DeepFlow,
}


def create_optical_flow(
    method: str, *, device: str | Device = "cpu", params=None
) -> OpticalFlow:
    try:
        cls = _REGISTRY[method]
    except KeyError:
        valid = ", ".join(_REGISTRY)
        msg = f"unknown flow method {method!r}: expected one of {valid}"
        raise ValueError(msg) from None
    return cls(params, device=device) if params is not None else cls(device=device)
```

호출자 패턴 (iivs-lib 시퀀스를 직접 순회):

```python
of = create_optical_flow("tvl1", device="cuda:1")   # 2번 GPU에서 CUDA TV-L1
for frame in sequence:            # 정규화된 uint8 프레임 (iivs-lib 0.2.0)
    flow = of.push(frame)
    if flow is None:              # 첫 프레임: 직전 없음
        continue
    ...                           # flow: (i-1 → i), (H, W, 2) float32

# DeepFlow(device="cuda:0")
# -> ValueError: 'deepflow' does not support the 'cuda' backend; use one of: cpu
```

## 클래스 설계 — 평가

```python
from typing import NamedTuple

from skimage.metrics import peak_signal_noise_ratio, structural_similarity


class FlowMetrics(NamedTuple):
    l1_norm: float
    l2_norm: float
    ssim: float
    psnr: float


class OpticalFlowEvaluator:
    """flow의 warp-consistency 평가.

    `prev`를 `flow`로 워핑해 `curr`와 비교한다. GT flow가 없으므로 warp
    일치도가 표준 프록시다. SSIM/PSNR은 skimage(host)라 CPU 고정이다.
    """

    def __init__(self, data_range: float = 255.0) -> None:
        self.data_range = data_range

    @staticmethod
    def warp(image: NDArray[np.uint8], flow: NDArray[np.float32]) -> NDArray[np.uint8]:
        h, w = image.shape[:2]
        gx, gy = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (gx + flow[..., 0]).astype(np.float32)
        map_y = (gy + flow[..., 1]).astype(np.float32)
        warped = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        return warped.astype(np.uint8)

    def score(
        self,
        prev: NDArray[np.uint8],
        curr: NDArray[np.uint8],
        flow: NDArray[np.float32],
    ) -> FlowMetrics:
        warped = self.warp(prev, flow)
        diff = warped.astype(np.float32) - curr.astype(np.float32)
        l1 = float(np.mean(np.abs(diff)))
        l2 = float(np.sqrt(np.mean(diff * diff)))
        ssim = float(structural_similarity(warped, curr, data_range=self.data_range))
        psnr = float(peak_signal_noise_ratio(warped, curr, data_range=self.data_range))
        return FlowMetrics(l1, l2, ssim, psnr)


class MetricsAccumulator:
    """시퀀스 전체 metric의 스트리밍 평균 (프레임 쌍마다 `add`)."""

    def __init__(self) -> None:
        self._sums = [0.0, 0.0, 0.0, 0.0]
        self._n = 0

    def add(self, metrics: FlowMetrics) -> None:
        for i, v in enumerate(metrics):
            self._sums[i] += v
        self._n += 1

    def mean(self) -> FlowMetrics:
        if self._n == 0:
            msg = "no metrics accumulated; call add() at least once before mean()"
            raise ValueError(msg)
        return FlowMetrics(*(s / self._n for s in self._sums))
```

## 미결정 (다음에 확정할 것)

1. **배치 위치 & 타입체크**: 권장 `iivs_cardio/optical_flow/`(라이브러리, **ty 검사
   대상**), `Device`는 `iivs_cardio/common/device.py`. cv2는 타입 스텁이 없어 ty가
   untyped로 취급 → `error-on-warning` 하에 일부 `# ty: ignore`가 필요할 수 있음.
   대안: `scripts/`(ty 제외)에 임시로 둠.
2. **CPU prev 저장 방식**: `_push_cpu`는 참조 저장. iivs-lib 시퀀스가 프레임마다
   새 배열을 주면 OK, 버퍼 재사용이면 `frame.copy()` 필요 → **iivs-lib 0.2.0 시퀀스의
   버퍼 동작 확인 후 확정**.
3. **DualTVL1 divergent 파라미터 정책**: 현재 "무시 + 문서화". 더 엄격히 가려면
   백엔드에 안 맞는 비기본값 설정 시 경고/에러.
4. **평가 GPU 경로**: `download=False` 이점을 살리려면 `warp`에 `cv2.cuda.remap`
   GPU 버전 추가 가능. 단 SSIM/PSNR은 host라 이득은 warp 단계 한정.
5. **async 스트림**: `cv2.cuda.Stream`으로 업로드·calc·다운로드 파이프라이닝(추후).

## 의존성

- **추가 필요**: `iivs-lib>=0.2.0` (데이터 시퀀스 읽기·순회; 배포 형태 TBD — git/사설
  인덱스 가능성). `uv add` 로 등록.
- **이미 있음**: `opencv-contrib-python`(CUDA build 4.13.0), `numpy>=2`,
  `scikit-image>=0.24`(SSIM/PSNR + scipy 전이), `torch`(filter_3d GPU에서 사용).

## 남은 작업 (구현 순서 제안)

1. `iivs_cardio/common/device.py`(`Device`) + `iivs_cardio/optical_flow/` 배치 확정
   → 위 두 클래스 파일화 (배치 1번 결정 후).
2. **3D 필터 구현** — [`filter-3d-design.md`](filter-3d-design.md) 참조 (median/gaussian,
   streaming delay-line, ellipsoid/cuboid, cpu numba·scipy / cuda torch).
3. **4-mode 정규화** (per-frame / pairwise / sequence / dataset) — flow 전처리, uint8화.
4. **iivs-lib 0.2.0** 의존성 추가 후 시퀀스 로더 배선.
5. 임시 CUDA flow **스크립트** — 위 계산·평가 클래스 + 3D 필터 + 정규화 조립
   (`scripts/optical_flow/`). 입력 소스(.bin 시퀀스 / synthetic / avi)·처리 범위
   (단일 dir / 배치)는 그때 확정.
6. 커널·정규화·arity(T→T−1) 단위 테스트 우선.

## 참고

- 레거시 원본: `iivs-lab/cardio-force-legacy` (archived) — `Python/calc_optflows.py`(54KB,
  통합본), `calc_optflows_old.py`(Hydra 8-method 실험본), `eval_optflows.py`(평가).
  파라미터 기본값·warp 평가 방식이 여기서 유래.
- 상위 설계: `iivs-lab/new-project-DESIGN.md`(cell-dynamics 전체), `PROJECT_CONTEXT.md`.
