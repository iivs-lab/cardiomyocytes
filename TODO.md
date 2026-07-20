# TODO

Tracked items that are not yet captured in code or tests. Promote an
item to a CHANGELOG entry once it lands.

## Open

- **Extend `.gitignore` with ML runtime artifacts.** Add `data/`,
  `outputs/` (or `runs/`/`results/`), `checkpoints/` (or `models/`),
  `logs/`, `wandb/`, `*.ckpt` before these directories appear — the current
  `.gitignore` only covers Python + `.venv`/`.cache`. See
  [`docs/foundations.md`](docs/foundations.md) §8.

- **Build the `optical_flow` evaluator (warp-consistency).** The compute
  estimators (Farneback / DualTVL1 / DeepFlow over `torch.Tensor`, CPU + CUDA,
  streaming `push`/`push_chunk` and stateless `calc`/`calc_batch`) are
  implemented under `iivs_cardio/optical_flow/estimators/`. Remaining: the
  warp-consistency `OpticalFlowEvaluator` (+ `FlowMetrics` /
  `MetricsAccumulator`) and sequence IO via `iivs-lib>=0.2.0`. Design in
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
  rework lands.
