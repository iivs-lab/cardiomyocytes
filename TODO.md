# TODO

Tracked items that are not yet captured in code or tests. Promote an
item to a CHANGELOG entry once it lands.

## Open

- **Propagate the CHW tensor layout to the kinematic kernels.** The layout is
  settled — CHW (`(2,H,W)`/`(N,2,H,W)`); the rationale now lives in
  [`docs/foundations.md`](docs/foundations.md) §2, whose kernel sketches are
  channel-first. The estimators (`(2,H,W)` flow) and `iivs_cardio/common/warp.py`
  already follow it; the (unwritten) kinematic kernels must too. The ancestor
  `new-project-DESIGN.md` §4.1 is still channel-last — it is an external
  historical document, so read it with that correction in mind.

- **Wire up the `optical_flow` pipeline (preprocessing + sequence IO + script).**
  The estimators (`iivs_cardio/optical_flow/estimators/`), the backward-warp
  utility (`iivs_cardio/common/warp.py` — `backward_warp` / `BackwardWarp`) and
  warp-consistency scoring (`iivs_cardio/optical_flow/evaluation.py` —
  `warp_consistency` / `WarpConsistency`) are done. Remaining: `data/`
  sequence IO, `data/preprocessing/`, and a thin assembly script under
  `scripts/optical_flow/`.

  Optimization left on the table: pipeline an estimator's input conversion,
  `calc` and output over a `cv2.cuda.Stream` (today everything runs on the
  single default stream).

- **Decide how much of `iivs-lib[torch]` to consume.** The dependency now pulls
  the `torch` extra, which enables `iivs.dhm.analysis.pytorch` (`phase_to_opd`,
  `opd_to_phase`, `calc_drymass`, `calc_drymass_from_phase`) and
  `iivs.common.data.pytorch` (masked `Mean` / `Std` / `Variance` / `Norm` / `Sum`
  reductions, `region_stack`, `apply_mask`) — tensor-in/tensor-out, preserving
  device and autograd. That overlaps work `docs/foundations.md` §1 claims for
  this project: `phase -> OPD`, dry mass, and *OPD variance*, and the masked
  spatial reductions are exactly the Field -> Profile summarization step. Decide
  per quantity whether to call iivs-lib or own the kernel; the single-math-source
  rule says pick one and do not mirror it. Verify the scalar calibration
  (`opd_scale`, `drymass_scale`) against the headers before adopting.

- **Build `iivs_cardio/data/` — sequence IO over `iivs-lib`.** `iivs-lib>=0.2.0`
  is now a dependency. Key API facts, verified against the installed package:
  `PhaseFloatSequence` is an **empty marker** carrying only the `DataSequence`
  surface (`len` / `seq[i]` / iteration / `get_item` / `get_meta` / `get_pair`
  + plurals). `frame_shape`, `value_range()`, `header` / `get_header()`,
  `files`, `to_image` live on the **concrete** `PhaseFileList` /
  `PhaseBinFolder` — so annotating a parameter as `PhaseFloatSequence` gives up
  scale, statistics and shape. `PhaseBinFolder(root, *, target_unit=None,
  validate="headers")`; `PhaseBinHeader` carries `pixel_size` (m) and
  `height_scale` (m/rad) — the scale source `docs/foundations.md` §7 defers to
  iivs-lib. Frames arrive as `NDArray[np.float32]`, so `data/` owns the
  numpy → torch (+ device) boundary.

- **Build `iivs_cardio/data/preprocessing/`.**

  A faithful prototype of both files already runs inside
  `scripts/optical_flow/benchmark_opencv.py` (ported from the legacy
  `Python/calc_optflows.py`) — harvest from there, and keep what it measured.

  > **Every number below comes from 20-frame excerpts** — 2 s at 10 Hz, 1 s at
  > 20 Hz, i.e. roughly one beat. The mechanical facts (footprint, border rule,
  > even-count median, the pairwise structural constraint) are exact; the
  > *quantitative* claims about radii and algorithm ranking are provisional and
  > should be re-measured on a full dataset before anything is frozen. An
  > earlier raw-frame measurement ranked Dual TV-L1 above Farneback and the
  > filtered one reversed it, which is reason enough not to trust either yet.

  **Measured: the temporal radius must scale with the frame rate.** The damage
  tracks the *time* the window spans, not its frame count, so the legacy's fixed
  `FILTER_RADIUS=(2,2,2)` is a 20 Hz value that silently over-smooths 10 Hz data.
  Relative beating amplitude (std/mean of per-pair mean `|flow|`) retained:

  | window | 10 Hz | 20 Hz |
  | --- | --- | --- |
  | ~150-300 ms (`rz=1`) | 99.8% | 99.3% |
  | 250 ms (`rz=2` @ 20 Hz) | — | 102.8% |
  | 500 ms (`rz=2` @ 10 Hz) | **68.0%** | — |
  | 700-900 ms (`rz=3..4`) | 57%, 50% | 87%, 87% |

  Keep the window under roughly 300 ms — `rz <= (0.3 * fps - 1) / 2`, i.e. 1 at
  10 Hz and 2 at 20 Hz, which fits both measurements. Beyond it the profile
  compresses and individual intervals invert (on `10hz_tif` at `rz=2` the largest
  raw peak becomes a trough, and mean `|flow|` falls 0.39 -> 0.12 px). The
  project's deliverable *is* the beating profile, so that is signal loss, not
  denoising. The effect is almost purely temporal: `(0,0,2)` alone reproduces
  nearly all of `(2,2,2)`.

  Two caveats before hard-coding anything. The 300 ms figure is really a
  *fraction of the beat period*, which varies by cell — measure it per dataset
  rather than freezing the constant. And the 20 Hz fixture is 20 frames = 1 s,
  possibly under one full beat, so its profile statistics are the weaker of the
  two (its unfiltered std/mean is 0.352 against 10 Hz's 0.578, and this data
  cannot separate a cell difference from too short a sample). Prefer reading the
  real interval from each sample's `timestamps.txt` over trusting the folder
  name — the release notes only claim "~10 Hz" and "~20 Hz".

  Meanwhile FB-err / `|flow|` sits at 4.5-5.4% for every configuration including
  none, so no filter setting improved the flow's relative quality — though that
  ratio may be tracking interpolation error rather than noise, so treat it as
  suggestive where the profile compression is direct.

  `filtering.py` — 3D spatiotemporal filter (median / gaussian),
  ellipsoid/cuboid footprint, per-axis size (radius for the ellipsoid), device
  selectable via `iivs_cardio/common/device.py` (`resolve_device`, `cuda:N`).
  Legacy semantics to keep, all verified against `scipy` in the prototype:
  ellipsoid = offsets with `(dx/rx)^2 + (dy/ry)^2 + (dz/rz)^2 <= 1` (33 taps at
  radius 2, against a cuboid's 125); border **truncate** (out-of-range
  neighbours dropped, not padded); and with an even number of valid samples the
  median **averages the middle two** — `torch.median` returns the lower one, so
  it cannot be used directly. Allow a zero radius to disable an axis, which the
  legacy could not express and which the measurement above needs.
  **Factor it as a stateless window core (`window -> centre frame`) with the
  streaming delay-line (`push` / `flush`) as a wrapper over it**, so the
  sequential pipeline and a chunked `torch.utils.data.Dataset` share one
  implementation and cannot drift apart. Decided: **torch only** on both CPU and
  CUDA (no numba / scipy dependency); CUDA median via gather + `torch.median`;
  border policy **truncate**; gaussian parametrized by per-axis `sigma` +
  `truncate`.

  `normalization.py` — 4-mode (per-frame / pairwise / sequence / dataset),
  splitting "compute stats" from "apply", emitting uint8 for the estimators.
  **Structural constraint from the prototype: `pairwise` cannot produce a single
  normalized frame list.** A frame is scaled by the joint range of whichever pair
  it is in, so it appears twice with two different uint8 encodings — the API has
  to be built around pairs (or windows), not frames, for this mode to exist. The
  legacy name for `sequence` was `sample`. Measured, the mode barely matters
  (pairwise beats sequence by ~0.0003 SSIM gain); `per_frame` is the unsafe one
  because rescaling each frame by its own extremes breaks the brightness
  constancy every estimator assumes.
  `sequence` and `per_frame` statistics can be delegated to iivs-lib's
  `value_range(index=None, unit=None)` rather than recomputed; `dataset` merges
  those per-sequence results; only `pairwise` needs its own two-frame pass.

  Dataset consumption (settled by discussion, not yet code): a ring buffer only
  works under sequential access, so per-frame `RandomSampler` would amplify I/O
  by the window size (5x at rz=2). The fix is to **sample chunks, not frames** —
  which this domain requires anyway (flow needs 2 consecutive frames, accel 3;
  cf. `foundations.md` §5 chunk >= 3, overlap 2). A chunk of K read with a 2*rz
  halo costs `(K+2rz)/K` (1.13x at K=32) and restores the buffer *within* the
  chunk. Check first whether a whole sequence simply fits in RAM
  (900x900 float32 = 3.24 MB/frame, ~1 GB per 300 frames), which makes the
  question moot.

- **Promote the identity baseline and forward-backward consistency into
  `evaluation.py`.** Both are prototyped in
  `scripts/optical_flow/benchmark_opencv.py`; they belong next to
  `warp_consistency`, because on this data warp-consistency alone is misleading
  in *both* directions.

  - **SSIM gain over identity** — score the same pair with a zero flow and
    report the difference. Real inter-frame motion is sub-pixel, so a zero flow
    already scores SSIM ~0.94; the raw number is dominated by the frames simply
    resembling each other, and only the gain reflects what the flow contributed.
  - **Forward-backward error** — `|f_fwd(x) + f_bwd(x + f_fwd(x))|` in px, 0 for
    a self-consistent flow. Needs the backward flow too, so the API differs from
    `warp_consistency`'s single-flow signature. Catches what SSIM cannot: a flow
    that matches frame2 better by fitting *noise* scores higher photometrically
    while contradicting the flow computed in the opposite direction. Measured:
    parameters that double the SSIM gain degrade FB error 8-20x.

  **Neither metric is safe alone** — a zero flow earns no gain but has zero FB
  error, and a noise-fitted flow earns gain with a large one. Whatever the API
  ends up being, it should make reporting them together the easy path, and the
  docstring must say why. Open: whether the baseline belongs inside
  `warp_consistency` (an extra returned key) or as a separate function.

- **Add opt-in real-data tests over the fixtures.** The real Koala time-lapses
  live in the private `iivs-lab/iivs-lib-fixtures` release (`gh release download
  v1 -R iivs-lab/iivs-lib-fixtures -D fixtures`); only the benchmark reads them
  today, and nothing fetches them automatically — `iivs-lib` has a
  `scripts/fixtures/fetch.py` + `lock.json` pair worth porting when this lands.
  Mirror its test pattern too: a `conftest.py` fixture parametrized over the
  time-lapses present, yielding an empty parameter set — and therefore skipping,
  not failing — when the folder is absent (CI, or a machine without the private
  data). Add the fixture directory to `.gitignore` at that point; it is ~1.2 GB.

  Worth asserting there, because the synthetic suite cannot: that an estimator's
  flow beats the identity baseline *and* stays forward-backward consistent. Both
  numbers are in `scripts/optical_flow/benchmark_opencv.py` already.
