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

  **Streaming only** — a `push` / `flush` delay line, with no random-access entry
  point. The reason is correctness rather than I/O: a filtered frame depends on
  its neighbours, so under random access "frame i" would silently depend on which
  window asked for it, while a `Dataset` advertises independent samples. Keep the
  window core a pure function of its window for testability, but do not ship a
  random-access path. The delay line also bounds memory — the prototype holds the
  whole padded volume at a measured 9.5 MB/frame, so ~1200 frames exhaust a 12 GB
  GPU, where a delay line needs `2rz+1` frames whatever the length. Cost is linear
  and flat at **3.01 ms/frame** on CUDA, 20x that on CPU. Decided: torch only on
  both (no numba / scipy), gaussian by per-axis `sigma` + `truncate`.

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
  extremes breaks the brightness constancy every estimator assumes (`sequence` is
  the legacy's `sample`). Applying stats is elementwise and local, so once they
  exist every mode is safe under random access; only computing them needs a pass.

  **Store per-frame `(min, max)`; all four modes derive from it exactly** —
  pairwise is the elementwise min/max of two neighbours, which is literally what
  the legacy computes; sequence reduces over frames, dataset over sequences.
  ~16 KB per 1000 frames, nothing redundant to disagree, and dataset composition
  stays a *view*: changing a split needs no recomputation, and training-split-only
  stats become the natural default rather than an extra mechanism (computing them
  across all splits leaks val/test into training). Storing the sequence-level pair
  as well is fine for readability and as corruption detection, but only as a
  *derived* field verified against the per-frame array on load — left
  authoritative it is a second source of truth that can silently disagree.

  This does lock in min/max semantics: sequence percentiles are **not** derivable
  from per-frame percentiles. If outlier fragility bites — one hot pixel sets the
  max and compresses everything else into part of the 256 uint8 levels — per-frame
  *histograms* compose additively and give exact percentiles at any level, at
  ~16 MB per 1000 frames. Version the sidecar so that switch stays open.

  **Do not use `phbounds.txt`.** It is Koala's uint8 *preview* range, and
  `PhaseBounds`' own docstring says the previews are never authoritative; it also
  describes raw phase, while a median can only shrink the range, so it would waste
  uint8 levels. Do not write our values into that filename either — the same name
  with different semantics is the silent-mismatch failure this project keeps
  hitting.

  **Cache format**: filtered frames as Koala `.bin` through iivs-lib's
  `save_phase_bin` / `save_phase_folder`, so `PhaseBinFolder` reads them back and
  the `pixel_size` / `height_scale` calibration travels *inside* the file instead
  of in a sidecar that can desync. Both `.bin` and iivs-lib's `.npy` are float32,
  so float16 means leaving the ecosystem entirely — revisit only if disk actually
  binds, and measure the precision loss on the physical quantities first. Our own
  sidecar carries what the ecosystem does not: the per-frame statistics and their
  unit, the filter parameters, a source hash (without which a changed radius
  silently reuses a stale cache), and a format version.

  **Whether to keep the cache is conditional**, not automatic. Regenerating costs
  ~3 s per 1000 frames on CUDA, so caching wins where the GPU is scarce or absent
  — CPU-only machines, and training loops where re-filtering every epoch competes
  with the model for the device. Regenerating wins while preprocessing parameters
  are still being explored, since every change invalidates the cache. Deleting raw
  phase to keep only filtered frames is a separate decision that should wait until
  the parameters are settled.

  A training `Dataset` therefore reads the *cache*, never raw sequences, and its
  random access is then unrestricted.

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
