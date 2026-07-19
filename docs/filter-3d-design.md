# 3D 필터 모듈 설계 (핸드오프)

> 다른 컴퓨터에서 이어서 작업하기 위한 설계 스냅샷. 대상: **`filter_3d`**
> (시공간 3D 필터 — median / gaussian). optical flow 앞단 전처리(노이즈 제거)로
> 쓰이며, `push(frame)` 스트리밍·device 지정을 지원한다.
>
> 관련: [`optical-flow-design.md`](optical-flow-design.md). **`Device` 값 객체는
> 두 모듈이 공유**한다(그 문서의 "공통: Device" 참조,
> `iivs_cardio/common/device.py`).

## 목표

레거시 `cardio-force-legacy`의 `Python/calc_optflows.py`에 있던 3D median 필터
(반경 `(2,2,2)` 타원체 footprint, numba)를 일반화한다.

- **파라미터**: 커널 shape(`ellipsoid` / `cuboid`), 3방향 크기(반경 `(rz, ry, rx)`),
  알고리즘(`median` / `gaussian` / …).
- **스트리밍**: optical flow처럼 `push(frame)` 단일 프레임 방식. 단 시간축 반경
  `rz` 때문에 **rz 프레임 지연 출력 + 종료 시 `flush()`** (delay line).
- **device**: `cpu` / `cuda` / `cuda:N`. **median도 GPU 지원**(torch).

## 파이프라인 위치

```
raw phase 프레임(float32)
  → Filter3D.push(frame)  → 필터된 프레임 (rz 지연)  [float32]
  → 4-mode 정규화 → uint8
  → OpticalFlow.push → flow
```

필터는 flow 앞단, **정규화 전**의 float phase에 적용한다(레거시 순서 그대로).

## 핵심 개념 — delay line (flow와의 차이)

- flow는 `prev`+`curr`만 필요 → 지연 1(첫 push가 `None`).
- 3D 필터는 center 프레임 `t`를 확정하려면 `[t-rz, t+rz]` 필요 → **미래 rz 프레임**을
  봐야 한다. 따라서:
  - `push(frame)`는 **rz 프레임 지연** 출력(초기 rz회는 `None`).
  - 입력 종료 시 `flush()`가 남은 마지막 rz개를 (시간축 절단 윈도우로) 배출.
- ring 버퍼 용량 = `2*rz + 1`. 경계 프레임도 유효 이웃만으로 필터 → **N in → N out**
  (arity 보존; beating profile 길이 T 유지에 필요).
- **버퍼 재사용(flow처럼)**: CUDA면 ring이 GPU 텐서를 보유(프레임당 업로드 1회).

## 알고리즘 × device 지원

| 알고리즘 | shape 사용? | CPU | CUDA(torch) |
|---|---|---|---|
| **median** | ○ (footprint) | numba(최속·선택 의존성) / scipy(폴백) | **gather + `torch.median`** |
| **gaussian** | ✕ (축별 sigma) | `scipy.ndimage.gaussian_filter` | 분리형 `F.conv3d` |
| (mean/min/max) | ○ | footprint reduce | footprint reduce |

**정직한 지점**: `shape`(ellipsoid/cuboid)는 **footprint(참여 voxel 마스크)** 개념 →
median/mean/min/max에만 의미. **gaussian은 hard footprint가 아니라 축별 sigma
가중치**라 `shape`를 무시하고 크기를 sigma로 해석한다.

**CUDA median의 핵심**: OpenCV CUDA엔 3D footprint median이 없다 → **torch로 구현**.
footprint의 True 오프셋마다 shifted 슬라이스를 gather → `(K, H, W)` 스택 →
`torch.median(dim=0)`. 이 방식이 GPU median과 `"cuda:N"` device 지정을 동시에 해결.

## 클래스 설계

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from enum import Enum
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from iivs_cardio.common.device import Device

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray


class KernelShape(Enum):
    ELLIPSOID = "ellipsoid"
    CUBOID = "cuboid"


def build_footprint(
    shape: KernelShape, radius: tuple[int, int, int]
) -> NDArray[np.bool_]:
    """(shape, 반경) → 3D boolean footprint. 축 (z, y, x)."""
    rz, ry, rx = radius
    if shape is KernelShape.CUBOID:
        return np.ones((2 * rz + 1, 2 * ry + 1, 2 * rx + 1), dtype=bool)
    zz, yy, xx = np.ogrid[-rz : rz + 1, -ry : ry + 1, -rx : rx + 1]
    # (z/rz)² + (y/ry)² + (x/rx)² <= 1  (반경 0 축은 오프셋 0만 포함)
    norm = sum(
        (a.astype(np.float64) / max(r, 1)) ** 2
        for a, r in ((zz, rz), (yy, ry), (xx, rx))
    )
    return norm <= 1.0


class Filter3D(ABC):
    """단일 프레임을 순차로 밀어 넣는 stateful 3D 필터 (delay line).

    시간축 반경 `rz` 탓에 center 프레임 t는 t+rz까지 봐야 확정된다:
    `push(frame)`는 rz 프레임 지연 출력(초기 rz회 `None`), 입력 종료 후
    `flush()`가 남은 rz개를 (시간축 절단 윈도우로) 배출한다. 경계도 유효
    이웃만으로 필터 → N in / N out.

    CUDA device면 ring이 GPU 텐서를 보유(프레임당 업로드 1회)한다. 입력은
    정규화 전 float phase 프레임.
    """

    algorithm: ClassVar[str]
    supported_devices: ClassVar[frozenset[str]]  # kind: {"cpu"} / {"cpu", "cuda"}

    def __init__(
        self, radius: tuple[int, int, int], *, device: str | Device = "cpu"
    ) -> None:
        device = Device.parse(device)
        if device.kind not in self.supported_devices:
            ok = ", ".join(sorted(self.supported_devices))
            msg = (
                f"{self.algorithm!r} 3D filter does not support the "
                f"{device.kind!r} backend; use one of: {ok}"
            )
            raise ValueError(msg)
        self.device = device
        self.radius = radius
        self._rz = radius[0]
        self._ring: deque = deque(maxlen=2 * self._rz + 1)  # cpu: numpy / cuda: torch
        self._t = -1  # 마지막으로 밀어 넣은 프레임 인덱스

    def reset(self) -> None:
        self._ring.clear()
        self._t = -1

    def push(self, frame: NDArray) -> NDArray | None:
        """center = (밀어 넣은 수 − rz)의 필터 결과. 초기 rz회는 `None`."""
        self._t += 1
        self._ring.append(self._to_device(frame))
        center = self._t - self._rz
        if center < 0:
            return None
        return self._from_device(self._emit(center, hi=self._t))

    def flush(self) -> Iterator[NDArray]:
        """입력 종료 후 남은 마지막 rz개를 시간축 절단 윈도우로 배출."""
        last = self._t
        for center in range(last - self._rz + 1, last + 1):
            if center >= 0:
                yield self._from_device(self._emit(center, hi=last))
        self.reset()

    def _emit(self, center: int, *, hi: int):
        lo = max(0, center - self._rz)  # 과거측 절단(경계)
        # ring 뒤쪽에서 [lo, hi] 스택을 꺼내 center 국소 인덱스와 함께 reduce.
        offset_from_newest = self._t - hi
        n = hi - lo + 1
        window = list(self._ring)[len(self._ring) - offset_from_newest - n :
                                  len(self._ring) - offset_from_newest]
        return self._reduce(self._stack(window), center - lo)

    # device 훅 (subclass가 backend별로 구현)
    @abstractmethod
    def _to_device(self, frame): ...          # numpy → numpy(cpu) / GPU 텐서(cuda)
    @abstractmethod
    def _from_device(self, frame): ...        # 결과 → numpy
    @abstractmethod
    def _stack(self, window): ...             # 프레임 리스트 → (w, H, W)
    @abstractmethod
    def _reduce(self, stack, center_local): ...  # (w,H,W) 윈도우 → center 2D
```

### Median

```python
class MedianFilter3D(Filter3D):
    algorithm = "median"
    supported_devices = frozenset({"cpu", "cuda"})  # GPU 지원(torch)

    def __init__(
        self,
        radius: tuple[int, int, int],
        *,
        shape: KernelShape = KernelShape.ELLIPSOID,
        device: str | Device = "cpu",
        cpu_impl: str = "auto",  # "numba" | "scipy" | "auto"(numba 있으면 numba)
    ) -> None:
        self.shape = shape
        self.footprint = build_footprint(shape, radius)
        self.cpu_impl = cpu_impl
        super().__init__(radius, device=device)

    # CPU: cpu_impl에 따라 numba(레거시 njit·parallel 이식) 또는
    #   scipy.ndimage.median_filter(window, footprint=...) 후 center 슬라이스.
    # CUDA: torch. 아래 스케치.
    def _reduce_cuda(self, stack, center_local):
        import torch
        import torch.nn.functional as F
        _, ry, rx = self.radius
        padded = F.pad(stack[None], (rx, rx, ry, ry), mode="replicate")[0]
        h, w = stack.shape[-2:]
        neighbors = [
            padded[center_local + dz, ry + dy : ry + dy + h, rx + dx : rx + dx + w]
            for (dz, dy, dx) in self._offsets(w=stack.shape[0], center=center_local)
        ]
        return torch.stack(neighbors, dim=0).median(dim=0).values  # per-pixel, GPU
```

- 시간축 절단은 ring 윈도우 길이로 자연 처리, 공간 경계는 `replicate` 패딩.
- `_offsets`: `self.footprint`의 True 위치를 현재 윈도우의 시간 범위에 맞춰 crop한 오프셋.

### Gaussian

```python
class GaussianFilter3D(Filter3D):
    algorithm = "gaussian"
    supported_devices = frozenset({"cpu", "cuda"})

    def __init__(
        self,
        sigma: tuple[float, float, float],  # (sz, sy, sx)
        *,
        truncate: float = 3.0,
        device: str | Device = "cpu",
    ) -> None:
        self.sigma, self.truncate = sigma, truncate
        rz = int(np.ceil(truncate * sigma[0]))  # 시간 반경은 sigma에서 유도
        super().__init__((rz, 0, 0), device=device)  # 공간은 분리형으로 처리

    # CPU: scipy.ndimage.gaussian_filter. CUDA: 분리형 F.conv3d(축별 1D 커널).
```

### 팩토리

```python
_REGISTRY = {"median": MedianFilter3D, "gaussian": GaussianFilter3D}

def create_filter_3d(
    algorithm: str,
    radius: tuple[int, int, int] | None = None,
    *,
    shape: KernelShape = KernelShape.ELLIPSOID,
    sigma: tuple[float, float, float] | None = None,
    device: str | Device = "cpu",
    **kwargs,
) -> Filter3D:
    ...  # optical flow의 create_optical_flow와 동형. 알 수 없는 algorithm은
         # 유효 집합을 명시하는 ValueError.
```

## 사용 예

```python
filt = MedianFilter3D(radius=(2, 2, 2), shape=KernelShape.ELLIPSOID, device="cuda:0")
for frame in phase_sequence:      # float phase (iivs-lib 0.2.0)
    out = filt.push(frame)        # rz 지연
    if out is not None:
        ...                       # 정규화 → flow
for out in filt.flush():          # 마지막 rz개
    ...
```

## 미결정 (OPEN — 아직 미정)

1. **경계 정책**: truncate(N in/N out, 레거시식) vs valid-only(N−2rz). 기본 truncate 제안.
2. **Gaussian 파라미터화**: `sigma`(+truncate) 직접 지정 vs `radius` 통일 후 sigma
   유도. 두 알고리즘의 크기 의미가 달라 divergent(DualTVL1과 같은 트레이드오프).
3. **CPU median 구현체 기본값**: `auto`(numba 있으면 numba). numba를 **선택적 의존성**
   으로 추가할지(속도 vs 의존성). torch median 성능이 충분하면 CPU는 scipy로 단순화도 가능.
4. **torch ↔ cv2 GpuMat 상호운용**: 필터(torch GPU) → 정규화 → flow(cv2 GpuMat)에서
   현재는 host 왕복. 추후 zero-copy fusion 여지(우선순위 낮음).

## 의존성

- **이미 있음**: `torch`(CUDA median/gaussian), `numpy>=2`, `scipy`(scikit-image
  전이 — CPU median/gaussian).
- **추가 검토**: `numba`(선택적, CPU median 속도). 미설치 시 scipy 폴백.

## 남은 작업

1. `iivs_cardio/common/device.py`(`Device`) 확정 후 `iivs_cardio/filter_3d/` 파일화.
2. `Filter3D` base(delay line/ring) + `_to_device`/`_stack` 백엔드 훅 구현
   (cpu numpy / cuda torch).
3. `MedianFilter3D`: cpu(numba/scipy) + cuda(torch gather+median), `_offsets` 구현.
4. `GaussianFilter3D`: scipy / F.conv3d.
5. 단위 테스트: 경계 절단, footprint 정확성, cpu==cuda 일치, arity(N in/N out),
   지연·flush 길이(초기 rz `None`, flush rz개).
