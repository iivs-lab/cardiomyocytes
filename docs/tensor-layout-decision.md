# 텐서 레이아웃 결정 — HWC vs CHW (미결정)

> **상태: OPEN — 확정 전.** flow/벡터장 텐서의 2-벡터축을 **마지막(HWC,
> `(H,W,2)`)** 에 둘지 **앞(CHW, `(2,H,W)` / `(N,2,H,W)`)** 에 둘지의 프로젝트
> 전역 규약. 이 문서는 판단 근거를 보존한다(맑은 머리로 결정할 것).

## 문제

- **프레임은 grayscale `(H,W)`** 라 무관. 판단 대상은 **flow·속도장 등 2-벡터
  텐서의 축 위치**뿐.
- **현재 상태(암묵적 HWC)**: estimators가 `(H,W,2)`(cv2 네이티브), evaluation.py도
  `(H,W,2)`로 작성됨. DL nn.Module(RAFT 등)은 `(N,2,H,W)` = CHW.
- 목표: evaluator를 **DL Metric + estimator 평가 양쪽**에 쓰려면 한 규약이 필요.

## 분석 (연산별로 누가 permute를 무는가)

| 연산 | native | HWC 비용 | CHW 비용 |
|---|---|---|---|
| cv2 flow 출력 | HWC | 0 | permute(view) 1 |
| DL flow-net 출력 | CHW | permute(view) 1 | 0 |
| warp의 grid 생성 | grid는 항상 `(H,W,2)`, 두 성분 조립 | `[...,0]` | `[0]` — 무승부 |
| grayscale 프레임 워핑 | 벡터축 없음 | 무승부 | 무승부 |
| **벡터장(속도 v) 워핑** | **grid_sample 이미지 입력 `(N,C,H,W)`** | **매 warp permute in+out (+contiguous 복사 가능)** | **0 (native)** |
| 커널 elementwise (`force=m·a`) | 무관 | `[...,None]` 깔끔 | `[...,None,:,:]` 장황 |
| 지표(PSNR/SSIM/MSE/MAE) | 프레임만 봄 | 무승부 | 무승부 |

## 핵심 값 판단

- **경계(cv2 vs DL, 1·2행)**: 대칭 → **상쇄**. 어느 규약이든 반대쪽 경계에서
  permute-view 1회.
- **torchmetrics는 flow 레이아웃과 무관**(지표는 프레임만 봄). ← 이전 논의에서
  "CHW가 torchmetrics에 유리"는 **오류였음**, 정정.
- **HWC의 유일한 이점 = 커널 인덱싱 가독성**(`[...,0]` vs `[...,0,:,:]`) — **코스메틱**
  (성능·기능 차이 0).
- **CHW의 유일한 실질 이점 = 5행(벡터장 워핑)**: Lagrangian kinematic 체인이 속도장·
  벡터장을 `grid_sample`로 반복 워핑하는데, `grid_sample` **이미지 입력이 `(N,C,H,W)`**
  라 2-벡터장은 `(N,2,H,W)`=CHW가 native. HWC면 매 warp마다 permute in/out(+복사).
  이것만 상쇄되지 않고 **hot-path에서 반복**됨.

**결론(잠정)**:
- **전체 파이프라인(DL + kinematic 벡터장 워핑 포함)** 관점 → **CHW가 이득**
  (근거는 "DL이라서"가 아니라 **grid_sample이 벡터장을 `(N,C,H,W)`로 받기 때문**).
- **범위가 estimator + evaluator뿐**(프레임만 워핑, 벡터장 워핑 없음)이라면 5행이
  사라져 **거의 무승부**, 그땐 cv2 native를 살려 **HWC가 미세 우위**.

## CHW로 통일 시 작업 (결정되면)

1. **estimators**: `_calc_*`/`gpumat_to_tensor` 출력을 `(2,H,W)`로 permute(한 곳에서).
   estimator 테스트의 shape 단언도 `(2,H,W)`로.
2. **evaluation.py**: `(2,H,W)` 입력으로(warp는 `flow[0]`/`flow[1]`로 grid 생성 —
   동일), frames는 `(H,W)` 유지, 배치 `(N,2,H,W)` 자연 지원.
3. **커널(미구현)**: `(...,2,H,W)` 채널-첫으로 스케치. `foundations.md §2`,
   `new-project-DESIGN.md §4.1`의 채널-마지막 커널 규약 갱신 필요.
4. **DL**: 변환 0.

## 결정

**PENDING** — 위 값 판단을 근거로 CHW 통일(전체 파이프라인 기준)이 유력하나,
확정은 보류. 확정 후 이 문서를 "DECIDED: …"로 갱신하고 위 작업을 진행한다.
