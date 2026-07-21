# TODO

Tracked items that are not yet captured in code or tests. Promote an
item to a CHANGELOG entry once it lands.

## Open

- **Propagate the CHW tensor layout to the kinematic kernels.** The layout is
  settled — CHW (`(2,H,W)`/`(N,2,H,W)`), rationale and channel-first kernel
  sketches in [`docs/foundations.md`](docs/foundations.md) §2. The estimators and
  `iivs_cardio/common/warp.py` already follow it; the unwritten kinematic kernels
  must too. The ancestor `new-project-DESIGN.md` §4.1 is still channel-last —
  read it with that correction in mind.

- **Wire up the `optical_flow` pipeline.** Estimators, `common/warp.py` and
  `optical_flow/evaluation.py` are done. Remaining: `data/` sequence IO,
  `data/preprocessing/`, and a thin assembly script under `scripts/optical_flow/`.

  Optimization left on the table: pipeline an estimator's input conversion,
  `calc` and output over a `cv2.cuda.Stream` (today everything runs on the single
  default stream).

- **Decide how much of `iivs-lib[torch]` to consume.** The extra enables
  `iivs.dhm.analysis.pytorch` (`phase_to_opd`, `calc_drymass`) and
  `iivs.common.data.pytorch` (masked `Mean` / `Variance` / `Norm` reductions),
  tensor-in/tensor-out and preserving device and autograd. That overlaps what
  `docs/foundations.md` §1 claims for this project — `phase -> OPD`, dry mass,
  OPD variance — and the masked spatial reductions *are* the Field -> Profile
  summarization step. Pick one owner per quantity; the single-math-source rule
  forbids mirroring. Verify `opd_scale` / `drymass_scale` against the headers
  before adopting.

- **Build `iivs_cardio/data/` — sequence IO over `iivs-lib`.** `PhaseBinFolder`
  takes the `Phase/Float/Bin` folder, not the time-lapse root. Frames arrive as
  `NDArray[np.float32]`, so this layer owns the numpy → torch (+ device)
  boundary. Mind the marker-vs-concrete trap documented in `foundations.md` §7.

- **Build `iivs_cardio/data/preprocessing/`.** A faithful prototype of both files
  runs inside `scripts/optical_flow/benchmark_opencv.py`, ported from the legacy
  `Python/calc_optflows.py` — harvest from there.

  > Quantitative claims from that script are **provisional**: they come from
  > 20-frame excerpts, about one beat. Ranking flipped between raw and filtered
  > frames, so settle radii and parameters on a full dataset. The mechanical
  > facts below are exact.

  `filtering.py` — 3D spatiotemporal filter (median / gaussian),
  ellipsoid/cuboid footprint, per-axis radius, device via
  `iivs_cardio/common/device.py`. Legacy semantics, verified against `scipy`:
  ellipsoid = offsets with `(dx/rx)^2 + (dy/ry)^2 + (dz/rz)^2 <= 1` (33 taps at
  radius 2 against a cuboid's 125); border **truncate** (out-of-range neighbours
  dropped, not padded); and with an even number of valid samples the median
  **averages the middle two**, so `torch.median` — which returns the lower —
  cannot be used directly. Allow a zero radius to disable an axis; the legacy
  could not express that.

  **Factor it as a stateless window core (`window -> centre frame`) with the
  streaming delay-line (`push` / `flush`) as a wrapper**, so the sequential
  pipeline and a chunked `Dataset` share one implementation. Decided: torch only
  on CPU and CUDA (no numba / scipy), gaussian by per-axis `sigma` + `truncate`.

  **The temporal radius must scale with the frame rate.** Damage tracks the time
  the window spans, not its frame count, so the legacy's fixed `(2,2,2)` is a
  20 Hz value that over-smooths 10 Hz data — at 10 Hz it costs ~30% of the
  beating profile's relative amplitude while `rz=1` costs none. The deliverable
  *is* the beating profile, so that is signal loss, not denoising. Derive the
  radius from the beat period per dataset rather than freezing a constant, and
  read the real interval from each sample's `timestamps.txt` — the fixture names
  only claim "~10 Hz" and "~20 Hz".

  `normalization.py` — 4 modes, splitting "compute stats" from "apply", emitting
  uint8. **`pairwise` cannot produce a single normalized frame list**: a frame is
  scaled by the joint range of whichever pair it is in, so it appears twice with
  two encodings — the API must be built around pairs (or windows) for the mode to
  exist at all. `per_frame` is the unsafe mode: rescaling each frame by its own
  extremes breaks the brightness constancy every estimator assumes. `sequence`
  (the legacy's `sample`) and `per_frame` statistics can be delegated to
  iivs-lib's `value_range(index=None)`; `dataset` merges those per sequence.

  Dataset consumption, settled by discussion: a ring buffer needs sequential
  access, so per-frame `RandomSampler` would amplify I/O by the window size.
  **Sample chunks, not frames** — which this domain requires anyway (flow needs 2
  consecutive frames, accel 3; cf. `foundations.md` §5). A chunk of K with a
  `2*rz` halo costs `(K+2rz)/K` and restores the buffer within the chunk. Check
  first whether a sequence simply fits in RAM (~1 GB per 300 frames), which makes
  the question moot.

- **Promote the identity baseline and forward-backward error into
  `evaluation.py`.** Both are prototyped in
  `scripts/optical_flow/benchmark_opencv.py`, whose docstring explains why they
  are needed: with sub-pixel motion a zero flow already scores SSIM ~0.94, and
  SSIM gain alone rewards a flow that fits noise. Neither metric is safe alone —
  a zero flow earns no gain but is perfectly self-consistent — so the API should
  make reporting them together the easy path. FB error needs the backward flow,
  so it cannot reuse `warp_consistency`'s single-flow signature. Open: whether
  the baseline is an extra returned key or a separate function.

- **Build a ground-truth flow benchmark from real frames.** Warp a real DHM frame
  by a known, smooth, sub-pixel displacement field (~0.3 px, the measured scale)
  and score estimators by endpoint error against it. This keeps real image
  statistics while restoring ground truth — the old synthetic scene had ground
  truth but the wrong motion regime, at 8-14 px.

  The point is to settle which proxy to trust. SSIM gain and forward-backward
  error routinely disagree (raising TV-L1's `lambda_` doubles the gain while
  degrading FB error 14x), and with no ground truth there is no way to say which
  is right. EPE decides it directly, and shows which proxy actually correlates
  with accuracy — after which the proxies can be used on real pairs with
  justified confidence.

  Caveat to design around: warping one frame transports its noise intact, so
  brightness constancy holds exactly and the task is unrealistically easy. Add an
  independent noise realization to the warped frame, or treat the numbers as
  ranking estimators rather than as achievable accuracy.

- **Add opt-in real-data tests over the fixtures.** The Koala time-lapses live in
  the private `iivs-lab/iivs-lib-fixtures` release (`gh release download v1 -R
  iivs-lab/iivs-lib-fixtures -D fixtures`); nothing fetches them automatically —
  `iivs-lib` has a `scripts/fixtures/fetch.py` + `lock.json` pair worth porting.
  Mirror its test pattern: a `conftest.py` fixture parametrized over the
  time-lapses present, so an absent folder skips rather than fails. Gitignore the
  directory at that point; it is ~1.2 GB. Worth asserting there, as no synthetic
  suite can: that a flow beats the identity baseline *and* stays
  forward-backward consistent.
