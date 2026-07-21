# Optical Flow 모듈 — 통합 설계 (남은 작업)

> **계산 추정기 + 백워드 워프 유틸은 구현 완료.**
>
> - **추정기** — `iivs_cardio/optical_flow/estimators/`가 정본이다:
>   `OpticalFlowEstimator` ← `OpenCVEstimator` ← `Farneback` / `DualTVL1` /
>   `DeepFlow`. `(H, W)` uint8 프레임 → `(2, H, W)` float32 flow를 **`torch.Tensor`**
>   로 주고받으며, CPU(numpy, zero-copy)·CUDA(device-resident, cupy ↔ GpuMat)를
>   하나의 인터페이스로 통일한다. streaming `push`/`push_chunk`(O(1) 상태) +
>   stateless `calc`/`calc_batch`, jaxtyping+beartype 경계 검증. 파라미터 기본값은
>   `cardio-force-legacy`와 일치. device는 `common.resolve_device`(`torch.device`).
> - **백워드 워프** — `iivs_cardio/common/warp.py`가 정본이다:
>   `backward_warp` 함수 + `BackwardWarp`(nn.Module, grid 캐시). warp-consistency
>   점수는 전용 evaluator 클래스 없이 `backward_warp` + 지표(torchmetrics 등)를
>   **호출부에서 조합**한다 (아래 참조).
>
> 남은 부분은 **통합**: 4-mode 정규화 · iivs-lib 시퀀스 IO · 조립 스크립트.
>
> 관련: [`filter-3d-design.md`](filter-3d-design.md) (3D 필터, 미구현).

## 목표 (남은 범위)

flow 계산·워프는 완료. 남은 것:

- **4-mode 정규화**(per-frame / pairwise / sequence / dataset) — 시퀀스 통계에
  의존하는 상류 전처리. 추정기는 정규화된 uint8만 받는다.
- **iivs-lib 0.2.0** 시퀀스 읽기·순회 + 조립 스크립트.

## 백워드 워프 (구현 완료)

`iivs_cardio/common/warp.py`. estimator와 정합하는 `torch.Tensor` I/O
(`(*dim, H, W)` image, `(*dim, 2, H, W)` float32 transform) + jaxtyping/beartype
경계 검증(함수). warp는 `torch.nn.functional.grid_sample`, **입력 텐서의 device에
상주**(CUDA는 GPU, CPU는 CPU, host offload 없음). optical_flow 전용이 아니라
kinematic 등도 공유하는 `common` 유틸이다.

- `backward_warp(image, transform, *, padding_mode)` — image를 `grid − transform`
  에서 bilinear 샘플. float32로 계산 후 image dtype 복원(정수 round/clamp, float
  소수 유지). `*dim` 선행 배치축을 함께 warp.
- `BackwardWarp(nn.Module)` — 동일 연산의 캐시 버전. `(H, W)` 좌표 grid는
  transform과 무관하므로 크기·device당 1회 생성·재사용(hot-path·`torch.compile`/
  `jit` 친화를 위해 런타임 타입검사는 함수 쪽에만 둔다).
- **warp-consistency 지표**: 전용 evaluator·`FlowMetrics`·`MetricsAccumulator`는
  두지 않는다. `backward_warp`로 `prev`를 warp한 뒤 PSNR/SSIM/MSE/MAE(및 opt-in
  LPIPS)를 **호출부가 직접** 계산·집계한다 — DL 학습 metric과 동일하게 필요한
  것만 골라 쓰기 위함.

> **⚠️ warp 방향(중요)**: estimator는 forward flow(`prev→curr`)를 내므로 curr
> 재구성은 **`grid − flow`**(= `grid − transform`) 샘플이다. 조상 문서/레거시의
> `grid + flow`는 **틀렸다** — 실측: 알려진 shift에서 올바른 부호 SSIM ~0.98 대
> 잘못된 부호 ~-0.23. `common/warp.py`는 올바른 부호를 쓴다.
> (`benchmark_opencv.py`가 이 레거시 버그를 물려받은 상태 → TODO.)

> 평가 관점은 [`docs/foundations.md`](foundations.md) §3(Lagrangian/Eulerian,
> 정확성 3규약) 참조.

## 미결정 / 환경 이슈

1. **⚠️ torch cuDNN 깨짐(이 개발 머신)**: GPU `conv2d`(그리고 향후 DL/learned-flow
   전반)가 `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`로 실패한다
   (`torch.backends.cudnn.enabled=False`면 동작). 원인 추정: `setup-opencv-cuda.ps1`이
   CUDA bin에 심볼릭한 cuDNN DLL이 torch 번들 cuDNN을 가림. **코드 문제 아님** —
   torch GPU convolution 전반이 이 머신에서 막힌다. → 머신 cuDNN 정리 필요(TODO).
2. **async 스트림**: 추정기의 입력 변환·calc·출력을 `cv2.cuda.Stream`으로
   파이프라이닝하는 최적화 여지(현재는 단일 default 스트림).

## 의존성

- **추가 필요**: `iivs-lib>=0.2.0` (데이터 시퀀스 읽기·순회; 배포 형태 TBD —
  git/사설 인덱스 가능성). `uv add` 로 등록.
- **이미 있음**: `torch`·`torchvision`, `torchmetrics`(warp-consistency·DL 학습
  metric, 호출부에서 사용), `opencv-contrib-python`(CUDA build 4.13.0, estimator),
  `numpy>=2`, `cupy-cuda13x`, `jaxtyping`/`beartype`(경계 검증).

## 남은 작업 (구현 순서 제안)

1. **3D 필터 구현** — [`filter-3d-design.md`](filter-3d-design.md) 참조
   (median/gaussian, streaming delay-line, ellipsoid/cuboid, cpu numba·scipy /
   cuda torch).
2. **4-mode 정규화** (per-frame / pairwise / sequence / dataset) — flow 전처리, uint8화.
3. **iivs-lib 0.2.0** 의존성 추가 후 시퀀스 로더 배선.
4. 임시 조립 **스크립트**(`scripts/optical_flow/`) — 추정기 + 워프/지표 + 3D 필터 +
   정규화 조립. 입력 소스(.bin 시퀀스 / synthetic / avi)·처리 범위(단일 dir /
   배치)는 그때 확정.

## 참고

- 레거시 원본: `iivs-lab/cardio-force-legacy` (archived) — `Python/calc_optflows.py`
  (통합본), `calc_optflows_old.py`(Hydra 8-method 실험본), `eval_optflows.py`(평가).
  파라미터 기본값·warp 평가 방식이 여기서 유래(단 warp 방향 부호는 위 교정 참조).
- 상위 설계: `iivs-lab/new-project-DESIGN.md`(cell-dynamics 전체), `PROJECT_CONTEXT.md`.
