# 프로젝트 파운데이션 (durable 설계·결정 스냅샷)

> **⚠️ 이 문서는 "의도/결정"의 스냅샷이지 "현재 코드의 사실"이 아니다.**
> 외부 컴퓨터를 포함한 **이후 작업이 이 문서를 100% 반영하지 않을 수 있다** —
> 결정은 뒤집힐 수 있고 구현은 앞서갈 수 있다. **불일치 시 우선순위:**
> 실제 코드 > `pyproject.toml`/`uv.lock`/git 이력 > [`TODO.md`](../TODO.md) >
> **이 문서**. 이 문서에 적힌 심볼·수치·경계는 **채택 전 코드로 재검증**할 것.
>
> 출처: 조상 문서 `new-project-DESIGN.md`(cell-dynamics 설계) + `PROJECT_CONTEXT.md`
> (부트스트랩 핸드오프) 중 **여전히 유효한 부분만 선별**. 낡은 사실(옛 copier ref,
> Python 버전, "bare project" 상태, `opencv-python`, `cardio_dynamics` 명명 등)은
> 의도적으로 제외했다.
>
> 모듈별 `*-design.md`는 **모두 제거**했다 — 구현된 부분은 코드·docstring이,
> 미구현 부분은 [`TODO.md`](../TODO.md)가 정본이다. 원문이 필요하면 git 이력에서
> 꺼낼 수 있다(`git log --diff-filter=D -- docs/`).

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
입력은 시간축-우선 블록: 스칼라장 `(T, H, W)`, 벡터장 `(T, 2, H, W)` — **채널-첫(CHW)**.
커널 시그니처는 그대로다. 바뀐 것은 그 `T`를 **누가 모으는가**로, §5대로 단계가
자기 시간 윈도우를 버퍼링해 넘긴다(전역 청크가 아니다).

> **CHW인 이유 (HWC 대비, 결정 완료).** 결정적 근거는 **하나**다: Lagrangian 체인이
> 속도장·벡터장을 `grid_sample`로 반복 워핑하는데 그 **이미지 입력이 `(N, C, H, W)`**
> 라 2-벡터장은 `(N, 2, H, W)`가 native다. HWC면 **매 warp마다 permute in/out(+복사)**
> 이 hot-path에서 반복된다. 나머지는 상쇄된다 — cv2(HWC 출력) ↔ DL(CHW 출력) 경계는
> 어느 규약이든 반대편에서 permute-view 1회로 **대칭**이고, HWC의 유일한 이점인 커널
> 인덱싱 가독성(`[..., 0]` vs `[..., 0, :, :]`)은 **코스메틱**(성능·기능 차 0)이며,
> 지표(PSNR/SSIM/MSE/MAE)는 프레임만 보므로 레이아웃과 **무관**하다.
> 적용: estimators `(2,H,W)` ✅ / `common/warp.py` `(*dim,2,H,W)` ✅ / DL ✅(원래 CHW)
> / kinematic 커널 ⬜(미구현 — 아래 스케치가 채널-첫 규약이다).

```python
def opd_from_phase(phase, opd_scale):        # phase(rad) → OPD(nm)
    return phase * opd_scale

def height_from_phase(phase, height_scale):  # phase(rad) → height(m)
    return phase * height_scale

def warp(field_next, flow):                  # field_next[t]를 flow(t→t+1)로 t좌표계 정렬
    ...                                      # grid_sample / remap

def displacement_xy(flow, pixel_size):       # flow(px) → 횡변위(m). (T-1, 2, H, W)
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

def force(accel, mass):                      # mass (T-1, H, W) → 채널축 삽입
    return mass[..., None, :, :] * accel     # 벡터 차원 D=2 or 3 무관
```

- **quantity 계열**: 분석(`OPD`, `dry mass`) / kinematic(`displacement → speed →
  acceleration`) / kinetic(`force = mass × accel`) / 통계(`OPD variance` = OPD의
  시간 reduction).
- **다중 엔드포인트**: `phase → OPD`에서 `… → force` 가지와 `OPD variance` 가지가 갈림.
- `speed`/`acceleration`/`force`는 채널축 `D`(2 또는 3)의 크기에 **무관**.

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

## 5. 스트리밍 수치 계약

> **갱신됨.** 이전 판은 "per-frame이 아니라 **청크 단위**(시간축 벡터화), 청크 ≥ 3,
> overlap = 2"였다. 파이프라인을 단계별 이터레이터로 재설계하면서 뒤집었다 —
> 아래 배치 항목이 이유다.

- **프레임 단위 스트리밍.** 청크가 아니라 프레임이 흐른다. 항목은 **자기 인덱스를
  들고 다니고**, 한 노드는 한 인덱스를 한 번만, 순서대로 낸다.
- **정렬은 스트림 위치가 아니라 인덱스로 한다.** 깊은 노드일수록 자기 창을 채우느라
  뒤처지므로 **같은 구동 스텝에서 노드마다 내놓는 인덱스가 다르다** — `OPD variance`가
  40을 낼 때 `force`는 38을 낸다. 그래서 훅도 표도 인덱스를 키로 잡는다(§6). "스텝마다
  모든 노드가 한 항목씩"을 가정하면 안 된다.
- **시간 윈도우는 노드가 자기 버퍼로 든다.** accel은 flow 3개(= phase 4개)를
  필요로 한다. 경계 연속성은 청크 overlap이 아니라 이 버퍼가 보장하며, 입력이 끝난
  뒤에도 버퍼에 남은 것을 **드레인**해야 한다(§6).
- **배치는 단계 내부의 선택이지 전역 계약이 아니다.** 900×900에서는 프레임 하나가
  이미 장치를 채워 필터는 배치로 **0.97~1.08배**에 그쳤고(측정), OpenCV 추정기는
  배치 API 자체가 없다(`calc_batch`가 파이썬 루프인 이유). DL 추정기의 coarse 레벨과
  반복 갱신처럼 **커널 실행 오버헤드가 지배하는 곳에서만** 단계가 내부적으로 N개를
  모아 한 번에 계산하고 N개를 낸다 — 개수 1:1은 그대로다. 전역 청크 크기는 이
  선택을 모든 단계에 강요하므로 쓰지 않는다. 배치 크기는 활성값 메모리와 직결되니
  설정으로 노출하고 기본값은 보수적으로 둔다(§ TODO의 CUDA 할당자 항목).
- **OPD variance 전역값**: 훅이 프레임별 `(n, mean, M2)`를 내고, **훅을 제공한 객체가**
  **Welford/Chan으로 병합**한다 → finalize. **float64 누적.** (윈도우 단위 분산이면
  프레임 독립.)
- variance/dry mass 합산은 **float64**, 그 외 기본 float32(필요 시 AMP).
- cf. `data/transforms/filtering`의 `FilteredSequence`가 같은 버퍼 개념을 먼저 쓴다
  (시간 반지름 rz면 창 = 2·rz+1).

## 6. 파이프라인 구조 — 노드 · 훅 · 조립

- **최상위는 선형 체인, DAG는 지표 단계 안에만.** `A 필터된 phase → B flow → C 지표`.
  항목은 교체가 아니라 **누적**된다 — B가 phase를 버리지 않고 flow를 얹는다. dry mass와
  OPD variance가 phase를 필요로 하기 때문이고, 덕분에 A의 소비자도 B의 소비자도 하나뿐
  이라 팬아웃이 없다. 팬아웃이 없으므로 노드 사이는 **평범한 제너레이터**로 충분하고,
  파이프라인 전역 캐시도 그 축출 정책도 필요 없다. C 안의 다이아몬드(`speed`가
  `acceleration`과 `kinetic energy`에 함께 쓰이는 것)는 **한 스텝 안에서 끝나는 지역
  메모**로 해결된다.
- **`Slot[T]` = `index` + `value: T | None`.** 인덱스는 장식이 아니다: 같은 스텝에
  계산기마다 낼 수 있는 인덱스가 다르므로(깊을수록 뒤처진다) 소비자는 자기가 받을
  번호를 미리 알 수 없다. **훅도 스텝이 아니라 인덱스로 기록한다** — 한 스텝의 결과를
  한 행으로 묶으면 서로 다른 시각의 값이 섞인다.
- **전진 규약**: `B[i] = flow(A[i] → A[i+1])`. 인덱스 i의 항목은 **시각 i의 상태**를
  기술하고 i 이후의 프레임으로 계산된다. 빈 자리와 지연이 꼬리에 모여 소비자가 첫
  항목부터 쓸모 있는 값을 받는다. ⚠️ **z 변위도 같은 규약이어야 한다** — §3–4의
  `Displacement3D`가 xy와 z를 합치므로, 한쪽만 후진 차분이면 **한 프레임 어긋난 두
  성분**이 조용히 합쳐진다.
- **훅은 A·B·C 세 곳에 붙고, 최종 소비자가 아니다.** 프레임별 결과만 반환하고 누적은
  훅을 제공한 객체의 책임이다. 프레임 저장, 범위 기록, flow 평가·저장이 전부 훅이다.
- **근원은 교체 가능하고 하류는 출처를 모른다.** A는 `원본 phase + 필터` 또는 `캐시된
  필터 폴더`, B는 `추정기` 또는 `캐시된 flow 폴더`. 체인에서 B를 빼면 C의 입력에 flow가
  없어 force·kinetic energy를 만들 수 없다 — **조립 시점에 검증해 거부**한다. 다 돌린
  뒤 빈 결과를 내는 것보다 낫다.
- **지표 계산기는 종류당 하나.** 요청 집합에서 의존성을 전이적으로 펼쳐 합집합을 만들고,
  명시적으로 요청된 것에만 출력 표시를 단다 — 명시/내부는 **출력 포함 여부**를 정할 뿐
  인스턴스 수를 정하지 않는다. 같은 지표를 다른 하이퍼파라미터로 두 번 만들지 않는다:
  그건 **실행을 분리**할 일이고, 그래야 `.hydra/config.yaml`이 무엇이 무엇인지 답한다.
- **계산기 캐시는 시퀀스 단위다.** 계산기가 창 버퍼를 들고 있으므로 모듈 전역에 두면
  앞 시퀀스의 마지막 프레임이 다음 시퀀스의 첫 지표에 섞인다 — 예외도 안 나고 값도
  그럴듯하다. 추정기 `push` 모드와 같은 실패 방식이다.
- **드레인이 필요하다.** 입력이 끝나도 깊은 계산기의 버퍼에는 낼 것이 남는다. 드라이버는
  소스가 아니라 **모든 출력이 소진될 때까지** 당긴다. 빠뜨리면 시퀀스 끝의 지표가 조용히
  사라지고, 개수를 세기 전에는 모른다.
- **조립은 워커 안에서.** 추정기는 `cv2` 객체 때문에 pickle되지 않으므로 부모는
  `*Params` 레시피만 보내고 워커가 만든다. `cv2.cuda`·CuPy가 프로세스 전역 장치를
  읽으므로(§7) 워커가 장치를 한 번 잡으면 그 안에서 조립되는 것이 **전부 같은 장치**에
  놓인다 — 단계마다 장치를 넘길 필요가 없다.

## 7. 멀티 GPU

- **데이터 병렬 권장**: **시퀀스 샤딩**(청크 샤딩이 아니다 — §5), GPU마다 독립
  파이프라인 인스턴스, 통신 0. 시퀀스가 온전히 한 워커에 가므로 **샤드 경계도,
  overlap도, stitch 시 dedupe도 없다** — 시간 윈도우는 단계 버퍼 안에서 닫힌다.
  구현은 `torch.multiprocessing`이 아니라 `mpire`(GPU당 1 프로세스, spawn)로 갔고,
  스케일은 선형이 아니었다: 7-GPU가 3.59배로, 디스크가 상한이다.
- **모델 병렬 비권장**(교차 GPU 전송·복잡도; 한 프레임이 한 GPU에 안 들어갈 때만).
- cf. 공유 `Device`(cuda:N): cv2.cuda는 전역 `setDevice`, torch는 per-op — 프로세스당
  GPU 1개 고정 모델과 잘 맞음.

## 8. iivs-lib 소비 경계 (일부 갱신됨 — ⚠️ 코드 재검증 필수)

이 프로젝트는 iivs-lib를 **소비만** 한다(코드 침투 없음). flow·warping·kinematic·
kinetic·OPD variance·필터는 전부 이 프로젝트 소유.

- **스케일·헤더**: `PhaseBinHeader`가 `pixel_size`(m)와 `height_scale`(m/rad)를
  들고 있음 — **설치본에서 확인 완료**. `OPDConverter.opd_scale`(phase→OPD),
  `DryMassCalculator.drymass_scale`은 아직 **미확인 추정**.
- **데이터 시퀀스·타임스탬프 로딩**: **iivs-lib 0.2.0**(의존성 등록 완료).
  조상 문서가 적은 kaparoo `WindowedSequence`/`FileFolderSequence`는 **폐기된 경로**.
  > ⚠️ **타입 함정**: `PhaseFloatSequence`는 **본문 없는 마커**라 `DataSequence`
  > 표면(`len`/`seq[i]`/순회/`get_item`/`get_meta`/`get_pair`)만 준다.
  > `frame_shape`·`value_range()`·`header`/`get_header()`는 **구체 클래스**
  > (`PhaseFileList`/`PhaseBinFolder`)에만 있다 — 파라미터를 마커로 넓게 받으면
  > 스케일·통계·shape를 포기하게 된다. 상세는 [`TODO.md`](../TODO.md).

## 9. 액션 아이템 — `.gitignore` (미완)

ML 런타임/대용량 산출물을 **구조로 커밋하지 말고 gitignore**: `data/`, `outputs/`
(또는 `runs/`/`results/`), `checkpoints/`(또는 `models/`), `logs/`, `wandb/`, `*.ckpt`.
현재 `.gitignore`는 Python + `.venv`/`.cache`만 덮으므로, 이 디렉터리들이 생기기 전에
확장 필요.

## 참고

- 모듈별 상세 설계 문서는 제거됨(상단 disclaimer 참조) — 구현분은 코드·docstring,
  미구현분은 [`TODO.md`](../TODO.md)가 정본.
- 조상 문서(일부 낡음, repo 밖): `new-project-DESIGN.md`(cell-dynamics 전체 설계 —
  두 모드(DL nn.Module / lazy 노드 DAG), flow 처리 등 더 상세하나 일부 분기),
  `PROJECT_CONTEXT.md`(부트스트랩 핸드오프 — 사실관계 상당히 낡음).
- 상위 규약: `AGENTS.md`(코딩·커밋·테스트 규약).
