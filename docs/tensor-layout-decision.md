# 텐서 레이아웃 결정 — HWC vs CHW (결정: CHW)

> **상태: DECIDED — CHW (`(2,H,W)` / `(N,2,H,W)`).** flow/벡터장 텐서의 2-벡터축을
> **앞(채널-첫)** 에 둔다. 이 문서는 판단 근거를 보존한다(§핵심 값 판단). 적용
> 현황은 §결정 참조.

## 문제

- **프레임은 grayscale `(H,W)`** 라 무관. 판단 대상은 **flow·속도장 등 2-벡터
  텐서의 축 위치**뿐.
- **현재 상태(CHW 확정)**: estimators가 `(2,H,W)`, `common/warp.py`가
  `(*dim,2,H,W)` transform. DL nn.Module(RAFT 등)도 `(N,2,H,W)` = CHW.
- 목표: estimator 출력·warp·DL을 **한 규약(CHW)** 으로 통일.

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

## CHW 적용 작업

1. ✅ **estimators**: `_calc_*` 출력이 `(2,H,W)`, 테스트 shape 단언도 `(2,H,W)`.
2. ✅ **warp**(`common/warp.py`, evaluation.py 대체): `(*dim,2,H,W)` transform
   입력, image는 `(H,W)`/`(*dim,H,W)` 유지, 배치 `(N,2,H,W)` 자연 지원.
3. ⬜ **커널(미구현)**: `(...,2,H,W)` 채널-첫으로 스케치. `foundations.md §2`,
   `new-project-DESIGN.md §4.1`의 채널-마지막 커널 규약 갱신 필요.
4. ✅ **DL**: 변환 0(이미 CHW).

## 결정

**DECIDED: CHW** — 위 §핵심 값 판단(특히 `grid_sample`이 벡터장을 `(N,C,H,W)`로
받으므로 벡터장 워핑이 CHW native)을 근거로 채널-첫으로 확정. estimator·warp·DL은
적용 완료, 남은 것은 미구현 kinematic 커널을 CHW로 작성하는 것뿐(작업 3).
