# TODO

Tracked items that are not yet captured in code or tests. Promote an
item to a CHANGELOG entry once it lands.

## Open

- **Build the `optical_flow` module (compute + eval classes).** Design is
  frozen in [`docs/optical-flow-design.md`](docs/optical-flow-design.md):
  stateful `push(frame)` estimators (Farneback / DualTVL1 / DeepFlow) over a
  `Device` (`cpu` / `cuda:N`) matrix, `FarnebackParams` / `DualTVL1Params`,
  and a warp-consistency `OpticalFlowEvaluator`. Open items there: placement
  (`iivs_cardio/optical_flow/` vs `scripts/`) and CPU `prev` copy semantics.
  Needs the `iivs-lib>=0.2.0` dependency for sequence IO.

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
