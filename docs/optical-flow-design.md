# Optical Flow 모듈 — 통합 설계 (남은 작업)

> **계산 추정기 + 평가 클래스는 구현 완료.**
>
> - **추정기** — `iivs_cardio/optical_flow/estimators/`가 정본이다:
>   `OpticalFlowEstimator` ← `OpenCVEstimator` ← `Farneback` / `DualTVL1` /
>   `DeepFlow`. `(H, W)` uint8 프레임 → `(H, W, 2)` float32 flow를 **`torch.Tensor`**
>   로 주고받으며, CPU(numpy, zero-copy)·CUDA(device-resident, cupy ↔ GpuMat)를
>   하나의 인터페이스로 통일한다. streaming `push`/`push_chunk`(O(1) 상태) +
>   stateless `calc`/`calc_batch`, jaxtyping+beartype 경계 검증. 파라미터 기본값은
>   `cardio-force-legacy`와 일치. device는 `common.resolve_device`(`torch.device`).
> - **평가** — `iivs_cardio/optical_flow/evaluation.py`가 정본이다:
>   `OpticalFlowEvaluator`/`FlowMetrics`/`MetricsAccumulator` (아래 참조).
>
> 남은 부분은 **통합**: 4-mode 정규화 · iivs-lib 시퀀스 IO · 조립 스크립트.
>
> 관련: [`filter-3d-design.md`](filter-3d-design.md) (3D 필터, 미구현).

## 목표 (남은 범위)

flow 계산·평가는 완료. 남은 것:

- **4-mode 정규화**(per-frame / pairwise / sequence / dataset) — 시퀀스 통계에
  의존하는 상류 전처리. 추정기는 정규화된 uint8만 받는다.
- **iivs-lib 0.2.0** 시퀀스 읽기·순회 + 조립 스크립트.

## 클래스 설계 — 평가 (구현 완료)

`iivs_cardio/optical_flow/evaluation.py`. estimator와 동일한 `torch.Tensor`
I/O(`(H,W)` uint8 프레임, `(H,W,2)` float32 flow) + jaxtyping/beartype 경계 검증.
warp는 `torch.nn.functional.grid_sample`, 지표는 **torchmetrics** — 전 과정이
**입력 텐서의 device에 상주**(CUDA flow는 GPU, CPU flow는 CPU, host offload 없음).

- `FlowMetrics(psnr, ssim, mse, mae, lpips)` — NamedTuple. 완전 일치 시 psnr=inf.
- `OpticalFlowEvaluator(data_range=255.0, *, padding_mode="border", lpips=False,
  lpips_net="alex")`. `score(prev, curr, flow)`(쌍별) + `warp`(static, `grid − flow`).
- **지표**: PSNR/SSIM/MSE/MAE 기본, **LPIPS는 opt-in**(`lpips=True`) — CNN 백본이
  ImageNet 학습이라 grayscale 위상영상엔 out-of-domain, 탐색용. lazy 로드.
- `MetricsAccumulator` — 스트리밍 평균(`add`/`mean`/`count`), LPIPS는 존재하는
  쌍만 평균. 빈 상태 `mean`은 `ValueError`.

> **⚠️ warp 방향 교정(중요)**: estimator는 forward flow(`prev→curr`)를 내므로 curr
> 재구성은 **`remap(prev, grid − flow)`** 다. 조상 문서/레거시의 `grid + flow`는
> **틀렸다** — 실측: 알려진 shift에서 올바른 부호 SSIM ~0.98 대 잘못된 부호 ~-0.23.
> `evaluation.py`는 올바른 부호를 쓴다. (`benchmark_opencv.py`가 이 레거시 버그를
> 물려받았을 가능성 → TODO.)

> 평가 관점은 [`docs/foundations.md`](foundations.md) §3(Lagrangian/Eulerian,
> 정확성 3규약) 참조.

## 미결정 / 환경 이슈

1. **⚠️ torch cuDNN 깨짐(이 개발 머신)**: GPU `conv2d`(SSIM의 gaussian, 그리고 향후
   DL/learned-flow 전반)가 `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`로 실패한다
   (`torch.backends.cudnn.enabled=False`면 동작). 원인 추정: `setup-opencv-cuda.ps1`이
   CUDA bin에 심볼릭한 cuDNN DLL이 torch 번들 cuDNN을 가림. **코드 문제 아님** —
   평가는 CPU에선 정상, GPU SSIM만 이 머신에서 skip된다. → 머신 cuDNN 정리 필요(TODO).
2. **async 스트림**: 추정기의 입력 변환·calc·출력을 `cv2.cuda.Stream`으로
   파이프라이닝하는 최적화 여지(현재는 단일 default 스트림).

## 의존성

- **추가 필요**: `iivs-lib>=0.2.0` (데이터 시퀀스 읽기·순회; 배포 형태 TBD —
  git/사설 인덱스 가능성). `uv add` 로 등록.
- **이미 있음**: `torch`·`torchvision`, `torchmetrics`(평가 지표 + LPIPS 백본),
  `opencv-contrib-python`(CUDA build 4.13.0, estimator), `numpy>=2`, `cupy-cuda13x`,
  `jaxtyping`/`beartype`(경계 검증). 평가는 이제 skimage 대신 **torchmetrics**(GPU 상주).

## 남은 작업 (구현 순서 제안)

1. **3D 필터 구현** — [`filter-3d-design.md`](filter-3d-design.md) 참조
   (median/gaussian, streaming delay-line, ellipsoid/cuboid, cpu numba·scipy /
   cuda torch).
2. **4-mode 정규화** (per-frame / pairwise / sequence / dataset) — flow 전처리, uint8화.
3. **iivs-lib 0.2.0** 의존성 추가 후 시퀀스 로더 배선.
4. 임시 조립 **스크립트**(`scripts/optical_flow/`) — 추정기 + 평가 + 3D 필터 +
   정규화 조립. 입력 소스(.bin 시퀀스 / synthetic / avi)·처리 범위(단일 dir /
   배치)는 그때 확정.

## 참고

- 레거시 원본: `iivs-lab/cardio-force-legacy` (archived) — `Python/calc_optflows.py`
  (통합본), `calc_optflows_old.py`(Hydra 8-method 실험본), `eval_optflows.py`(평가).
  파라미터 기본값·warp 평가 방식이 여기서 유래(단 warp 방향 부호는 위 교정 참조).
- 상위 설계: `iivs-lab/new-project-DESIGN.md`(cell-dynamics 전체), `PROJECT_CONTEXT.md`.
