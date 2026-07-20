# TODO

Tracked items that are not yet captured in code or tests. Promote an
item to a CHANGELOG entry once it lands.

## Open

- **Decide the flow/vector tensor layout: HWC vs CHW.** Whether flow and the
  kinematic vector fields put the 2-vector axis last (`(H,W,2)`, cv2-native) or
  first (`(2,H,W)`/`(N,2,H,W)`, DL-native). Analysis + the (tentative CHW) lean
  is in [`docs/tensor-layout-decision.md`](docs/tensor-layout-decision.md).
  Blocks finalizing `evaluation.py`'s flow layout and the (unwritten) kinematic
  kernels. Currently the estimators + evaluator are HWC.

- **Fix the dev machine's torch cuDNN (GPU `conv2d` fails).** On this Windows
  box every torch GPU convolution raises `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`
  (works with `torch.backends.cudnn.enabled = False`). Likely cause: the cuDNN
  DLLs `scripts/compute_env/setup-opencv-cuda.ps1` symlinks into the CUDA `bin`
  shadow torch's bundled cuDNN. It is an environment issue, not a code bug — but
  it blocks GPU SSIM in the evaluator and any future GPU DL / learned flow, so
  the setup needs reconciling (keep the OpenCV cuDNN links from overriding
  torch's, or align versions).

- **Extend `.gitignore` with ML runtime artifacts.** Add `data/`,
  `outputs/` (or `runs/`/`results/`), `checkpoints/` (or `models/`),
  `logs/`, `wandb/`, `*.ckpt` before these directories appear — the current
  `.gitignore` only covers Python + `.venv`/`.cache`. See
  [`docs/foundations.md`](docs/foundations.md) §8.

- **Wire up the `optical_flow` pipeline (normalization + sequence IO + script).**
  The estimators (`iivs_cardio/optical_flow/estimators/`) and the
  warp-consistency evaluator (`iivs_cardio/optical_flow/evaluation.py` —
  `OpticalFlowEvaluator` / `FlowMetrics` / `MetricsAccumulator`) are done.
  Remaining: 4-mode normalization (per-frame / pairwise / sequence / dataset),
  sequence read/iterate via `iivs-lib>=0.2.0`, and a thin assembly script under
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
