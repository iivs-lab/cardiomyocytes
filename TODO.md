# TODO

Tracked items that are not yet captured in code or tests. Promote an
item to a CHANGELOG entry once it lands.

## Open

- **Propagate the CHW tensor layout to the kinematic kernels.** The layout is
  settled — CHW (`(2,H,W)`/`(N,2,H,W)`), rationale in
  [`docs/tensor-layout-decision.md`](docs/tensor-layout-decision.md). The
  estimators (`(2,H,W)` flow) and `iivs_cardio/common/warp.py` already follow it;
  the (unwritten) kinematic kernels must too, and the channel-last kernel
  sketches in [`docs/foundations.md`](docs/foundations.md) §2 and
  `new-project-DESIGN.md` §4.1 need updating to channel-first.

- **Wire up the `optical_flow` pipeline (normalization + sequence IO + script).**
  The estimators (`iivs_cardio/optical_flow/estimators/`), the backward-warp
  utility (`iivs_cardio/common/warp.py` — `backward_warp` / `BackwardWarp`) and
  warp-consistency scoring (`iivs_cardio/optical_flow/evaluation.py` —
  `warp_consistency` / `WarpConsistency`) are done. Remaining: 4-mode
  normalization (per-frame / pairwise / sequence / dataset), sequence
  read/iterate via `iivs-lib>=0.2.0`, and a thin assembly script under
  `scripts/optical_flow/`. Design in
  [`docs/optical-flow-design.md`](docs/optical-flow-design.md).

- **Build the `filter_3d` module (3D spatiotemporal filter).** Design is
  frozen in [`docs/filter-3d-design.md`](docs/filter-3d-design.md): a
  streaming `push(frame)` delay-line filter (median / gaussian) with
  ellipsoid/cuboid footprints, per-axis size, and a `Device` matrix — CPU
  (numba / scipy) and **CUDA median via torch** (gather + `torch.median`,
  which also gives `cuda:N` selection). Shared `Device` value object lives in
  `iivs_cardio/common/device.py`. Open items: border policy, gaussian
  parametrization, optional `numba` dependency.

- **Rework the optical-flow benchmark on real data.** Replace the seeded
  synthetic scene in `scripts/optical_flow/benchmark_opencv.py` with real
  cardiomyocyte imaging data and the optical-flow parameters used in prior
  experiments. The current synthetic benchmark reports Dual TV-L1 quality
  *below* Farneback — the opposite of what is expected — so its numbers
  (TV-L1's especially) are provisional and should not be trusted until the
  rework lands. **Cause still unknown.** An earlier note here blamed an
  inverted warp; that was wrong and is retracted — measured on a known shift,
  the benchmark's `remap(frame2, grid + flow)` vs `frame1` scores SSIM 1.0000,
  exactly like the evaluator's equivalent `remap(frame1, grid - flow)` vs
  `frame2` (the *legacy* `remap(frame1, grid + flow)` vs `frame2` is the broken
  one, at -0.31). Remaining suspects: the synthetic scene may simply not suit a
  global variational method, and the old benchmark fed TV-L1 float32 `[0, 255]`
  while Farneback got uint8 and let each backend use its own default
  parameters. The benchmark now runs through the shared estimators, which
  removes both confounders — and **re-measuring did not help**: at 900px TV-L1
  still scores SSIM ~0.18 against Farneback's ~0.97 (CPU and CUDA now agree to
  within 0.015, so the two backends are consistent — they are just both bad
  here). That leaves the synthetic scene itself, or TV-L1's default parameters
  on it, as the remaining suspect — which the real-data rework settles.
