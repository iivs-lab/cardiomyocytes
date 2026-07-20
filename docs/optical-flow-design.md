# Optical Flow 모듈 — 평가·통합 설계 (남은 작업)

> **계산 추정기는 구현 완료** — `iivs_cardio/optical_flow/estimators/`가 정본이다:
> `OpticalFlowEstimator` ← `OpenCVEstimator` ← `Farneback` / `DualTVL1` /
> `DeepFlow`. `(H, W)` uint8 프레임 → `(H, W, 2)` float32 flow를 **`torch.Tensor`**
> 로 주고받으며, CPU(numpy, zero-copy)·CUDA(device-resident, cupy ↔ GpuMat)를
> 하나의 인터페이스로 통일한다. streaming `push`/`push_chunk`(O(1) 상태) +
> stateless `calc`/`calc_batch`, jaxtyping+beartype 경계 검증. 파라미터 기본값은
> `cardio-force-legacy`와 일치. device는 `common.resolve_device`(`torch.device`).
>
> 이 문서는 **아직 미구현**인 부분만 다룬다: warp-consistency 평가 클래스와
> 통합(정규화 · iivs-lib 시퀀스 IO · 조립 스크립트).
>
> 관련: [`filter-3d-design.md`](filter-3d-design.md) (3D 필터, 미구현).

## 목표 (남은 범위)

flow 계산은 완료. 남은 것:

- **warp-consistency 평가** — GT flow가 없으므로 prev를 flow로 워핑해 curr와
  비교(L1/L2/SSIM/PSNR)하는 게 표준 프록시. `OpticalFlowEvaluator`(쌍별 `score`)
  와 `MetricsAccumulator`(스트리밍 평균)를 분리한다.
- **4-mode 정규화**(per-frame / pairwise / sequence / dataset) — 시퀀스 통계에
  의존하는 상류 전처리. 추정기는 정규화된 uint8만 받는다.
- **iivs-lib 0.2.0** 시퀀스 읽기·순회 + 조립 스크립트.

## 클래스 설계 — 평가 (미구현)

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

> flow 입력은 추정기 출력과 동일한 `(H, W, 2)`. `torch.Tensor`를 받는다면
> host offload(`.cpu().numpy()`) 후 skimage로 넘긴다 — 평가 관점은
> [`docs/foundations.md`](foundations.md) §3(Lagrangian/Eulerian, 정확성 3규약)
> 참조.

## 미결정

1. **평가 GPU 경로**: `warp`에 `cv2.cuda.remap` GPU 버전을 추가할 수 있다. 단
   SSIM/PSNR은 host(skimage)라 이득은 warp 단계에 한정된다.
2. **async 스트림**: 추정기의 입력 변환·calc·출력을 `cv2.cuda.Stream`으로
   파이프라이닝하는 최적화 여지(현재는 단일 default 스트림).

## 의존성

- **추가 필요**: `iivs-lib>=0.2.0` (데이터 시퀀스 읽기·순회; 배포 형태 TBD —
  git/사설 인덱스 가능성). `uv add` 로 등록.
- **이미 있음**: `opencv-contrib-python`(CUDA build 4.13.0), `numpy>=2`,
  `scikit-image>=0.24`(SSIM/PSNR + scipy 전이), `torch`, `cupy-cuda13x`,
  `jaxtyping`/`beartype`(추정기 경계 검증).

## 남은 작업 (구현 순서 제안)

1. **3D 필터 구현** — [`filter-3d-design.md`](filter-3d-design.md) 참조
   (median/gaussian, streaming delay-line, ellipsoid/cuboid, cpu numba·scipy /
   cuda torch).
2. **4-mode 정규화** (per-frame / pairwise / sequence / dataset) — flow 전처리, uint8화.
3. **iivs-lib 0.2.0** 의존성 추가 후 시퀀스 로더 배선.
4. **`OpticalFlowEvaluator` 구현** + warp/metric 단위 테스트.
5. 임시 조립 **스크립트**(`scripts/optical_flow/`) — 추정기 + 평가 + 3D 필터 +
   정규화 조립. 입력 소스(.bin 시퀀스 / synthetic / avi)·처리 범위(단일 dir /
   배치)는 그때 확정.

## 참고

- 레거시 원본: `iivs-lab/cardio-force-legacy` (archived) — `Python/calc_optflows.py`
  (통합본), `calc_optflows_old.py`(Hydra 8-method 실험본), `eval_optflows.py`(평가).
  파라미터 기본값·warp 평가 방식이 여기서 유래.
- 상위 설계: `iivs-lab/new-project-DESIGN.md`(cell-dynamics 전체), `PROJECT_CONTEXT.md`.
