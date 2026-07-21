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

- **Re-run `setup-opencv-cuda.ps1` once (elevated) on every dev machine.** The
  torch-vs-OpenCV cuDNN clash is fixed at its source. The script used to symlink
  *every* `cudnn*.dll` into the CUDA `bin` — a directory on `PATH` that torch
  also searches — so torch (which bundles its own cuDNN 9.20 in `torch/lib`)
  loaded a foreign 9.23 sub-library (`cudnn_engines_tensor_ir`) and every GPU
  `conv2d` failed with `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`. It now links
  only the core cuDNN set and *removes* stale non-core links, so one elevated
  re-run both repairs and future-proofs a machine (`-DryRun` previews it without
  elevation). Verified on the `C:\Users\kaparoo` box — GPU `conv2d` OK, full
  suite 94 passed (the CUDA SSIM test no longer skips). **The other dev machine
  still needs the re-run** (or, minimally, deleting the
  `cudnn_engines_tensor_ir64_9.dll` link).

- **Extend `.gitignore` with ML runtime artifacts.** Add `data/`,
  `outputs/` (or `runs/`/`results/`), `checkpoints/` (or `models/`),
  `logs/`, `wandb/`, `*.ckpt` before these directories appear — the current
  `.gitignore` only covers Python + `.venv`/`.cache`. See
  [`docs/foundations.md`](docs/foundations.md) §8.

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
  rework lands. **Likely cause:** building the evaluator surfaced that the
  legacy warp direction is inverted — the backward warp that reconstructs
  `curr` from `prev` is `grid - flow`, not `grid + flow` (empirically SSIM
  ~0.98 vs ~-0.23 on a known shift). `evaluation.py` uses the correct sign;
  the benchmark inherits the legacy bug, so switch it to the evaluator before
  trusting any numbers.
