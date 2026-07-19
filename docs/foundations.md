# 프로젝트 파운데이션 (durable 설계·결정 스냅샷)

> **⚠️ 이 문서는 "의도/결정"의 스냅샷이지 "현재 코드의 사실"이 아니다.**
> 외부 컴퓨터를 포함한 **이후 작업이 이 문서를 100% 반영하지 않을 수 있다** —
> 결정은 뒤집힐 수 있고 구현은 앞서갈 수 있다. **불일치 시 우선순위:**
> 실제 코드 > `pyproject.toml`/`uv.lock`/git 이력 > 모듈별 `docs/*-design.md` >
> **이 문서**. 이 문서에 적힌 심볼·수치·경계는 **채택 전 코드로 재검증**할 것.
>
> 출처: 조상 문서 `new-project-DESIGN.md`(cell-dynamics 설계) + `PROJECT_CONTEXT.md`
> (부트스트랩 핸드오프) 중 **여전히 유효한 부분만 선별**. 낡은 사실(옛 copier ref,
> Python 버전, "bare project" 상태, `opencv-python`, `cardio_dynamics` 명명 등)은
> 의도적으로 제외했다. 모듈별 상세는 [`optical-flow-design.md`](optical-flow-design.md),
> [`filter-3d-design.md`](filter-3d-design.md).

## 1. 아키텍처 결정 (가급적 재론 금지 — 단, §상단 disclaimer 적용)

- **umbrella 패키지 + 연구별 서브패키지**: `iivs_cardio/<research>/`, 공용 코드는
  `iivs_cardio/common/` (공유 `Device`가 여기 사는 근거).
- **얇은 `scripts/`**: 인자 파싱·설정 로드·IO·device/시퀀스 배선만. **모든 실제
  로직은 `iivs_cardio/`** 에 둔다 (단일 수학 출처 원칙 보호).
- **editable install은 개발 편의용**: `import iivs_cardio` 안정화 목적. **PyPI 배포
  아님**, `py.typed` 없음, 사설 애플리케이션/모노레포.
- **console entry point(`[project.scripts]`) 기각**: CLI 코드를 패키지 안으로
  끌어들여 `scripts/` 분리와 충돌하므로.

## 2. compute 커널 — 단일 수학 출처

DL·파이프라인·2D/3D·numpy/torch가 **모두 같은 수학 함수**를 호출한다(중복 금지).
입력은 `(T, H, W[, C])` 시간축-우선 청크 가정.

```python
def opd_from_phase(phase, opd_scale):        # phase(rad) → OPD(nm)
    return phase * opd_scale

def height_from_phase(phase, height_scale):  # phase(rad) → height(m)
    return phase * height_scale

def warp(field_next, flow):                  # field_next[t]를 flow(t→t+1)로 t좌표계 정렬
    ...                                      # grid_sample / remap

def displacement_xy(flow, pixel_size):       # flow(px) → 횡변위(m). (T-1, H, W, 2)
    return flow * pixel_size

def z_displacement(height, flow, *, lagrangian=True):  # 물질 점 높이 변화. (T-1, H, W)
    if lagrangian:
        return warp(height[1:], flow) - height[:-1]
    return height[1:] - height[:-1]                    # Eulerian(소운동 근사)

def speed(displacement, dt):
    return displacement / dt

def acceleration(v, flow, dt, *, lagrangian=True):     # 물질미분
    if lagrangian:
        return (warp(v[1:], flow) - v[:-1]) / dt
    return (v[1:] - v[:-1]) / dt

def force(accel, mass):
    return mass[..., None] * accel           # 벡터 차원 D=2 or 3 무관
```

- **quantity 계열**: 분석(`OPD`, `dry mass`) / kinematic(`displacement → speed →
  acceleration`) / kinetic(`force = mass × accel`) / 통계(`OPD variance` = OPD의
  시간 reduction).
- **다중 엔드포인트**: `phase → OPD`에서 `… → force` 가지와 `OPD variance` 가지가 갈림.
- `speed`/`acceleration`/`force`는 마지막 벡터축 `D`(2 또는 3)에 **무관**.

## 3. 정확성 3규약 + Lagrangian/Eulerian (non-obvious, 필수 지식)

1. **시간 정렬(arity)**: per-interval 양은 `T → T−1`. flow·z변위·가속도가 일관 정렬.
2. **단위 정합**: flow는 픽셀 단위 → `× pixel_size`로 미터화한 뒤 z(미터)와 결합.
3. **Lagrangian 워핑**: 물질 점을 따라가는 차분(flow로 워핑 후 차분).

**왜 Lagrangian이 중요한가 (조용한 버그 방지):**
- **Acceleration**: 물질미분 `Dv/Dt = ∂v/∂t + (v·∇)v`. Eulerian 격차 = `(v·∇)v` →
  **O(1), dt 무관** → **dt를 줄여도 Eulerian으론 못 메운다**(정상류 `∂v/∂t=0`이라도
  입자는 가속). ⇒ **Lagrangian 사실상 필수.**
- **Displacement(z)**: Eulerian 오차 = `(v·∇)h·dt` → O(dt). 그래도 (u,v)가 이미
  Lagrangian이라 **일관성** 때문에 **Lagrangian 기본**(극소운동이면 보간오차 회피용
  Eulerian 허용).
- 규약 고정: 워핑 방향 = flow 규약, 경계/가림 마스크, 단위 정합. 진단:
  `‖(v·∇)v‖` 대 `‖∂v/∂t‖`.

## 4. 2D → 3D 확장 (코드 중복 방지 규칙)

- **차원이 바뀌는 건 displacement 하나뿐.** `speed/accel/force`는 `(...,D)`에 동일
  동작 → **공유**(Speed3D 등 불필요).
- `Displacement3D = concat(xy = flow×pixel_size [m], z = z_displacement [m])`.
- `z = height 시간차분`, `height = phase × height_scale`, flow와 arity 동일(T→T−1).
- **단위 정합 필수**: xy를 `pixel_size`로 미터화 후 z(미터)와 결합.

## 5. 청킹 · 스트리밍 수치 계약

- per-frame이 아니라 **청크 단위**(시간축 벡터화).
- **청크 ≥ 3**(accel 필요), **overlap = 최대 시간 윈도우 − 1 = 2**(경계 연속성).
- **OPD variance 전역값**: 청크에 걸쳐 **Welford/Chan 스트리밍 누적**(청크별
  `(n, mean, M2)` 병합 → finalize). **float64 누적.** (윈도우 단위 분산이면 청크 독립.)
- variance/dry mass 합산은 **float64**, 그 외 기본 float32(필요 시 AMP).
- cf. `filter_3d`의 delay-line도 시간 윈도우·overlap 개념을 공유.

## 6. 멀티 GPU

- **데이터 병렬 권장**: 시퀀스/청크 샤딩, **GPU마다 독립 파이프라인 인스턴스**,
  통신 0, 거의 선형 확장. `torch.multiprocessing`(GPU당 1 프로세스), stateful 그래프는
  프로세스별 별도 인스턴스. 샤드 경계는 overlap → stitch 시 dedupe.
- **모델 병렬 비권장**(교차 GPU 전송·복잡도; 단일 청크가 한 GPU에 안 들어갈 때만).
- cf. 공유 `Device`(cuda:N): cv2.cuda는 전역 `setDevice`, torch는 per-op — 프로세스당
  GPU 1개 고정 모델과 잘 맞음.

## 7. iivs-lib 소비 경계 (일부 갱신됨 — ⚠️ 코드 재검증 필수)

이 프로젝트는 iivs-lib를 **소비만** 한다(코드 침투 없음). flow·warping·kinematic·
kinetic·OPD variance·필터는 전부 이 프로젝트 소유.

- **스케일·헤더 (유효 추정)**: `pixel_size`(헤더), `height_scale`(phase→height),
  `OPDConverter.opd_scale`(phase→OPD), `DryMassCalculator.drymass_scale`.
- **데이터 시퀀스·타임스탬프 로딩**: **iivs-lib 0.2.0** 에서 온다.
  > ⚠️ **갱신 지점**: 조상 문서는 이 부분을 **kaparoo**(`WindowedSequence`/
  > `FileFolderSequence`)로 적었으나, 최근 결정은 **iivs-lib 0.2.0**. 원본에 나온
  > 구체 클래스명(`PhaseBinFolder`/`PhaseBinList`, `TimestampsTxtFile`/
  > `TimestampsFixedFPS` 등)은 **iivs-lib 0.2.0 API로 반드시 재확인**할 것.

## 8. 액션 아이템 — `.gitignore` (미완)

ML 런타임/대용량 산출물을 **구조로 커밋하지 말고 gitignore**: `data/`, `outputs/`
(또는 `runs/`/`results/`), `checkpoints/`(또는 `models/`), `logs/`, `wandb/`, `*.ckpt`.
현재 `.gitignore`는 Python + `.venv`/`.cache`만 덮으므로, 이 디렉터리들이 생기기 전에
확장 필요.

## 참고

- 모듈별 상세 설계: [`optical-flow-design.md`](optical-flow-design.md),
  [`filter-3d-design.md`](filter-3d-design.md).
- 조상 문서(일부 낡음, repo 밖): `new-project-DESIGN.md`(cell-dynamics 전체 설계 —
  두 모드(DL nn.Module / lazy 노드 DAG), flow 처리 등 더 상세하나 일부 분기),
  `PROJECT_CONTEXT.md`(부트스트랩 핸드오프 — 사실관계 상당히 낡음).
- 상위 규약: `AGENTS.md`(코딩·커밋·테스트 규약).
