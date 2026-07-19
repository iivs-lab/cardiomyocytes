# TODO

Tracked items that are not yet captured in code or tests. Promote an
item to a CHANGELOG entry once it lands.

## Open

- **Build the `optical_flow` module (compute + eval classes).** Design is
  frozen in [`docs/optical-flow-design.md`](docs/optical-flow-design.md):
  stateful `push(frame)` estimators (Farneback / DualTVL1 / DeepFlow) with a
  CPU/CUDA backend matrix, `FarnebackParams` / `DualTVL1Params`, and a
  warp-consistency `OpticalFlowEvaluator`. Open items there: placement
  (`iivs_cardio/optical_flow/` vs `scripts/`), CPU `prev` copy semantics,
  then 3D median filter + 4-mode normalization + the CUDA flow script.
  Needs the `iivs-lib>=0.2.0` dependency for sequence IO.

- **Rework the optical-flow benchmark on real data.** Replace the seeded
  synthetic scene in `scripts/optical_flow/benchmark_opencv.py` with real
  cardiomyocyte imaging data and the optical-flow parameters used in prior
  experiments. The current synthetic benchmark reports Dual TV-L1 quality
  *below* Farneback — the opposite of what is expected — so its numbers
  (TV-L1's especially) are provisional and should not be trusted until the
  rework lands.
