# TODO

Tracked items that are not yet captured in code or tests. Promote an
item to a CHANGELOG entry once it lands.

## In flight — `scan_phase_range`

Immediate work, unlike the design questions below. The script finds sources,
opens them as `FrameSequence`s, ranges every frame, writes the filtered frames a
run asks for, and writes one JSON per run.

`save_frames`, the single traversal, the unconditional range, the trimmed
provenance block and the `mpire` pool have all landed; see the CHANGELOG. What
follows is what those left open.

- **The worker path earns its keep, and a SATA SSD is what caps it.** Measured on
  the 440-sequence / 448,800-frame set (900x900, 7x Quadro RTX 6000): seven GPUs
  ran 3.59x one, not 7x, because reading peaked at 478 MiB/s against the drive's
  560 MB/s rating. Only the two heaviest kernels are GPU-bound; the other eight
  wait on the disk. Frame records cost 168 B each, so the whole dataset is 72 MiB
  in memory and about 53 MiB of JSON at `indent=2` -- dropping the indent saves
  40%, and gzip would save far more.

  Inverting the sweep loop is what the 251 GiB of RAM bought, and it has been
  done once by hand: the sequences were split to roughly 150 GiB through
  `source.include` listings and every filter run over a chunk before moving on,
  so only the first config of each chunk read from the disk. It underdelivered --
  see the next entry -- and it is not automated; the split and the merge of 121
  output files are both manual.

- **What the 121-job sweep actually validated, and what it did not.** Its product
  was the pipeline, not the numbers -- the range table it produced is recorded
  under Analysis as a negative result. Ran 2026-07-31 08:27 to 20:48, 12 h 21 m
  against an 8.8 h estimate, so the cache inversion paid less than predicted;
  per-chunk timings would say whether the first job of each chunk is the only
  slow one, and were not collected.

  Validated on real load: hydra 1.4.0.dev6 composing and running 121 `--multirun`
  jobs on Python 3.14, config groups and `_target_whitelist_` included; the
  per-job output directory, verified by a 3-job smoke run leaving three files
  where the old code left one; spawn + `SimpleQueue` device handoff across seven
  GPUs, with `resolve_devices` / `visible_cuda_devices` against a real driver;
  `select` splitting 440 sequences over eleven `.txt` listings with no overlap or
  gap, confirmed by the merge finding zero duplicates and zero missing; and
  `FrameSequence` / `FilteredSequence` / `MedianKernel` / `IdentityKernel` over
  about 4.9 M frame-filter operations without a crash or a non-finite result.

  **That device handoff has since been rewritten**, so this run no longer
  validates the code that ships: `mpire` replaced the queue and its initializer,
  and `Device` replaced both resolvers. It stands as evidence about the
  pipeline's shape, not about its current implementation.

  Still unvalidated on real load, and none of it needs a server: the
  `save_frames` write path, exercised so far only over a synthetic tree; failure
  handling, because all 121 jobs succeeded and nothing exercised a dead job; and
  the `mpire` pool, likewise only over a synthetic tree.

- **Settle the thread share between one worker and sixty-four.** `pin_threads`
  has landed beside where a worker claims its device, giving each `torch`'s own
  default divided by the worker count and leaving a lone worker alone. Only the
  widest point of the table below is measured, though: the share between is a
  policy, not a result, and the run that would settle it is a repeat of this one
  at 2, 4, 8 and 16 workers. It also moves `torch` alone, so a stage that leaves
  `torch` for numpy still takes threads by core count.

  Measured after the sweep on the same server, 40 sequences at `frame_step=50`
  with a median `(1, 1, 1)`, 64 cores. `compute.workers=1` stays in this process
  and builds no pool, yet already runs at 1748% CPU -- stacking processes on that
  only contends:

  ```
  pinned workers=64   26.40 s   1361% cpu   <- best
  workers=1           35.60 s   1748% cpu
  pinned workers=8    38.17 s    469% cpu
  workers=4           77.55 s   4835% cpu
  workers=8           94.29 s   5055% cpu
  workers=16          95.84 s   5154% cpu
  ```

  Unpinned is therefore 2.7x *slower* at sixteen workers than at one, while
  `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` at sixty-four beats the sequential path
  by 1.35x -- which is what `pin_threads` now does in code rather than in an
  environment variable every caller has to remember. Even pinned it reaches only
  1361% of 64 cores, so it ends up waiting on the drive that also caps the GPU
  path at 3.59x, which is why the gain is 1.35x and not sixty-four.

  The same run settled three things the sweep could not. Worker count does not
  change the numbers: 1, 4, 8 and 16 returned identical dataset bounds.
  `frame_step` is exact -- zero mismatches against the full-rate GPU document.
  And **CPU and CUDA agree bit-for-bit**, 600 of 600 frames at worst delta
  `0.000e+00`, which is what ordering and averaging the middle two should give
  and had never been checked.

- **Turn the device-parity result into a test, and cover the branch it missed.**
  Nothing in `tests/` compares a kernel across devices today; copy the
  `requires_cuda` skip marker from `tests/common/test_cuda_utils.py`, which is
  there for OpenCV interop only.

  The server check is stronger than it looks and narrower than it sounds. It
  crossed the `sort` / `topk` boundary rather than merely running one kernel
  twice: `ellipsoid (1, 1, 1)` takes 7 offsets, which is outside
  `_CUDA_TOPK_SAMPLES`, so CUDA sorted while CPU -- which never tests `is_cuda`
  -- used `topk`, and 600 of 600 frames still matched exactly. Border pixels
  carry NaN padding and an even valid count, so the NaN ordering and the
  average-the-middle-two path were both exercised.

  What it did not cover: CUDA's `topk` branch, which of the eleven sweep configs
  only `ellipsoid (2, 2, 2)` reaches at 33 offsets -- every other config lands
  outside `range(33, 64)` and sorts. And it compared per-frame `(min, max)`, not
  every pixel: 600 exact matches is strong evidence, not proof. A test should
  compare whole tensors with `torch.equal` at offset counts on both sides of the
  boundary.

- **Performance: one candidate is worth writing, and the rest are dead ends.**
  Ranked by what each was measured to return, not by how interesting it is.
  *Inverting the sweep loop* is the one that paid, and it took no code at all —
  the bullet above.

  *Batching the range's device sync* is the only code change left with a number
  behind it, and that number is an estimate. `finite_range` reads
  `low.isfinite()` per frame, which pulls the value to the host and stops the CPU
  from reading the next frame while the GPU works. A `FrameSequence.frame_ranges()`
  that launches every `aminmax` first and transfers once removes the stall: the
  scalars are 8 B a frame, so a 1200-frame sequence holds 9.6 KB, and CUDA's
  launch queue self-throttles rather than growing without bound. The non-finite
  fallback cannot branch per frame without the sync it exists to remove, so it
  becomes a second pass over only the frames whose bounds came back non-finite —
  in practice none, and `numel()` is a shape query that syncs nothing. Estimated
  5-8% overall, derived from the middle-weight kernels sitting at 65%
  utilization, not measured end to end.

  Settle `value_range` while doing it. **It already has no caller outside its own
  tests** -- the single traversal took the last one, since `scan_sequence` holds
  each frame and calls `finite_range` on it directly. So the question is not
  whether `frame_ranges()` would displace it but whether it earns its place at
  all: `_global_value_range` folds into the plural, and `value_range(slice)` keeps
  a meaning of its own only if something wants a subset. Nothing does today, and
  this is an application that ships no API.

  **It pulls against the shared traversal above.** Writing a frame moves it to
  the host, which is the same sync the batching removes — so on a `save_frames`
  run the stall returns regardless, and the batching pays only on a range-only
  run. Whichever lands second has to say which path it optimizes.

  *Prefetching source reads* belongs below `FilteredSequence`, whose window
  buffer is not thread-safe; that buffer is also why a forward pass asks the
  source for exactly one new frame per step, making the pattern trivial to
  predict. It would have to serve `get_items` as well as `get_item`, since that
  is what `_window` calls. It buys little here — a cached job already reads at
  memory speed, and a cold one is capped by the drive rather than by when the
  read is issued. Revisit if the data moves to NVMe.

  *Dropping `indent=2`* is one character, takes the document from ~53 MiB to
  ~31 MiB, and costs no time. It is disk, not throughput.

- **Measured and rejected, recorded so they are not tried again.**

  - *Samples-last in `MedianKernel.apply`.* `torch.stack(..., dim=-1)` makes a
    pixel's samples contiguous and halves the sort in isolation, but the stack
    then writes each slice strided across the sample axis, and over the whole
    `apply` that costs more than the sort saves: 0.54x-0.92x on the ten configs.
  - *Refitting `_CUDA_TOPK_SAMPLES`.* The 128-element step it looked like it
    should follow appears only under that samples-last layout. Re-measured on the
    whole `apply`, one kernel per process, at every offset count the median
    configs produce (7 to 343): the committed `range(33, 64)` picks the faster
    branch for all ten, where a tier rule mispredicts at 147.
  - *Several processes per GPU.* Without MPS the device time-slices, so
    throughput is unchanged — 0.93x-0.96x.
  - *Threads instead of worker processes.* The per-frame Python loop holds the GIL.

  Three of these failed the same way: measured on a part (the sort, the branch,
  one kernel) and lost on the whole (`apply`, the sweep). Measure the whole, and
  measure one kernel per process — eleven in one process fragments the CUDA
  allocator badly enough to read `cuboid (3,3,3)` at 8.46 s against its true
  139 ms, a 60x inflation of the same code.

- **The drive is the ceiling and no code change moves it.** NVMe at 3-7 GB/s
  would release the eight configs now waiting on ~500 MB/s, which is a larger
  factor than everything above together.

- **A per-worker progress bar, on top of the pool's own.** The `mpire` pool has
  landed with `progress_bar` / `enable_insights` / `worker_lifespan` behind
  `compute.*`; this is the piece left of it. `worker_init` opens a
  `tqdm(position=worker_id + 1, leave=False)` -- the pool's bar holds position 0
  -- and each task resets it to the sequence length and renames it; `worker_exit`
  closes it. Verified rendering seven independent lines under spawn.

  Worth it only for the two heaviest kernels. Below `cuboid (3,3,1)` it buys
  nothing: seven workers finishing a sequence every 15 s already move the
  sequence-level bar every couple of seconds, where `cuboid (3,3,3)` leaves it
  still for over two minutes and looks hung. Keep it a config flag, since seven
  lines collide with anything else on the terminal.

  > Correct the note this entry used to carry: `tqdm` does **not** go quiet off a
  > tty. Measured after the migration, a redirected pool bar writes a redraw line
  > per update into the log, so `compute.progress_bar` defaulting to true costs
  > every `--multirun` job that noise. Decide the default against a real sweep.

- **Deferred by decision, recorded so they are not rediscovered.** Renaming the
  `Params` suffix to `Config` on `KernelParams` and the optical-flow params
  classes. Reading frames on multiple threads inside `FrameSequence.value_range`.
  Putting `FrameShapedMixin` / `ValueRangeMixin` on `FrameSequence` — the latter is
  unusable on tensors as written, since it tests emptiness with `finite.size == 0`
  and a `Tensor`'s `size` is a method, so the comparison is never true.

- **`run_estimator.py` still holds four real defects**, surfaced once `ty` began
  checking `scripts/`: a mismatched return type at `:39`, both arguments to
  `FrameNormalizer.apply` wrong at `:52` (against
  `iivs_cardio/data/transforms/normalization.py:142`), and a wrong argument to
  `process_sequence` at `:76`. `ruff` adds an unused loop variable at `:51` and
  two flows computed and dropped at `:53`-`:54`.

- **Where the measurements live, and how to drive the script again.** None of it
  is in this repository, and the numbers above cannot be re-derived without it.

  On the 7-GPU server, under `/sdd/results/phase_range`:
  `chunked/chunk-NN/<job>/phase_range_*.json` holds the sweep, eleven chunks of
  eleven jobs; `chunk-NN.txt` beside them are the `source.include` listings that
  split the 440 sequences; `smoke/` is the three-job run that verified the
  per-job output directory; `cpu-smoke`, `cpu-scale/w*`, `cpu-pinned/w*` and
  `cpu-check` are the CPU validation runs. The source root is
  `/sdd/data/NEXEL/Off-axis_20Hz_Long-term`.

  Merging needs no `.hydra` reading: each document carries its own resolved
  `filter` block, so grouping on that and folding `dataset.sequences` across the
  eleven chunks reconstructs a whole-dataset range per configuration. Sequence
  `source` is unique, which is what makes a duplicate or a missing chunk
  detectable. The script that did it was deliberately left out of the repository
  -- rewrite it when the analysis resumes.

  **Override syntax bites here.** `compute=cpu` works in the short form because
  the config group and its package are both `compute`, but the filter group is
  mounted at a different package and needs
  `data/transforms/filtering@filter=<name>` -- plain `filter=<name>` does not
  select it. Any job's exact invocation is recoverable from its own
  `.hydra/overrides.yaml`, which is how this was found.

## Analysis — what to settle on the full dataset

Experiments and policy rather than modules. These decide what the caches will
hold, so they gate most of the implementation below.

- **Find the configuration that best explains the data.** The project's current
  priority: sweep filtering (on/off, kernel shape, per-axis radius) against every
  estimator and parameter set over the full dataset, then cache the winning phase
  and flow.

  **Score on three axes, not one.** Warp consistency measures only how well a
  flow reconstructs a frame, and a search will happily game that proxy:

  - *bias* — endpoint error against the ground-truth benchmark below, or failing
    that the fidelity of mean `|flow|`;
  - *consistency* — forward-backward error, which catches a flow that earns its
    photometric score by fitting noise;
  - *beating-profile amplitude* — std/mean of per-pair mean `|flow|`, which is
    what caught the legacy temporal radius compressing the signal by ~30%.

  One photometric number is a mismatched objective: the deliverable is the
  beating profile and force, not a reconstructed frame.

  **Do not assume Dual TV-L1 wins.** Both it and Farneback are coarse-to-fine, so
  that is not the difference; the estimator *inside* each level is. TV-L1's L1
  data term buys robustness to occlusion and illumination change — neither occurs
  in transparent phase imaging — at the cost of efficiency under the roughly
  Gaussian phase noise, and its TV regularizer prefers piecewise-constant fields
  where the tissue deforms as a smooth continuum. At sub-pixel motion the pyramid
  is inert for both (Farneback's `num_levels` 1/3/5 measure bit-identical).
  Whether the resulting shrinkage is a defect or a useful variance reduction is
  what the three axes are there to decide. Judge speed by the cost of generating
  a cache once, not per epoch — that makes DeepFlow's CPU-only path far less
  disqualifying than it first appears.

  **Do not select per frame pair; validate per-sequence selection before trusting
  it.** Deriving a parameter from a measured covariate (temporal radius from the
  frame interval) is principled and extrapolates; picking whichever configuration
  scores highest is selection bias. Per-pair selection is the worst case — it
  maximizes exactly the noise-fitting these metrics are vulnerable to, and since
  estimators differ in bias by ~2.2x, switching mid-sequence injects a step into
  the beating profile larger than the drug-induced changes the analysis exists to
  detect.

  Test the per-sequence hypothesis with a **split-half check**: does the
  configuration that wins on half a sequence's pairs also win on the other half?
  If not, the variation is noise. If it replicates, regress the per-sequence
  winner on covariates (frame rate, mean `|flow|`, SNR, beat period) — an
  explained difference becomes a rule, an unexplained one stays a single global
  setting. Prefer parameterizing by covariate over switching estimators: one or
  two degrees of freedom instead of one per sequence, and no discontinuity.

- **Value range does not discriminate between filters — settled on the full
  dataset.** Eleven configurations over all 440 sequences and 448,800 frames,
  every one complete. Sequence length is bimodal rather than uniform -- 132 of
  600 frames and 308 of 1200, which sums to the 448,800 -- so anything reasoning
  from a 1020-frame average is reasoning about a length no sequence has. The
  whole spread is 10.6%: `identity` spans 9.1256 and the
  most aggressive `cuboid (3,3,3)` spans 8.1592, for 49x the offsets and about
  120x the time. In uint8 that is 0.0357 rad per level against 0.0319 — the same
  picture. So the filter has to be chosen on the beating profile, which is the
  deliverable; this rules the range out as the criterion rather than answering it.

  **Temporal radius is nearly free of effect here.** Holding shape and spatial
  radius, `rz` 1 -> 3 narrows the range by 0.14-0.44%, while `cuboid (3,3,1)` ->
  `(3,3,3)` costs 1.8x the time for 0.32%. Spatial radius is what moves it,
  monotonically, and `cuboid` beats `ellipsoid` at equal radius by sampling more
  of the same box. That is a second argument for `rz = 1`, independent of the
  ~30% of beating amplitude a 20 Hz radius costs at 10 Hz.

  **Two sequences own the dataset's range**, which matters more than any filter
  choice. `Isoprenaline/167nM/Treated/20260319/2` holds the minimum under all
  eleven configurations. The maximum is `E-4031/1uM/Treated/20260624/1` under ten
  — but under `identity` it is `E-4031/10uM/Treated/20260123/2`, whose peak the
  weakest median `(1,1,1)` drops by 0.4978, or 8.6%. A value a 3x3x3
  neighbourhood's median erases is one pixel: that sequence has a hot pixel, and
  it was setting the range for the entire dataset. Filtering is also asymmetric —
  that first median takes 0.4978 off the maximum and 0.0383 off the minimum, so
  93% of its effect is on the positive side.

  This is the outlier fragility the per-frame `(min, max)` note below warns about,
  now observed rather than anticipated. Before setting a normalization policy,
  look at the distribution of per-sequence ranges (every JSON carries them) and
  decide between excluding such sequences, clipping, and moving to per-frame
  histograms for exact percentiles.

- **Cache the least-biased flow; smooth at the consumer.** Bias and variance are
  not symmetric. A noisy flow can be smoothed afterwards; motion that
  regularization shrank away is unrecoverable, because every sample is displaced
  identically. Measured, Dual TV-L1 returns ~0.46x Farneback's mean `|flow|` —
  over half the motion gone.

  The planned consumers disagree, which is why this matters. Kinematics and force
  want low variance: acceleration is a second difference, multiplying independent
  noise by `sqrt(6)` while shrinking a smooth 1 Hz signal sampled at 10 Hz by
  ~0.40, so each double-differencing costs roughly 6x in SNR. Frame interpolation
  and supervised flow training want low bias: a half-magnitude flow places
  interpolated content half-way short, and a network trained on it inherits the
  shrinkage permanently. Caching the sharper flow serves both, since the force
  path can regularize at consumption.

  Use a **single** configuration for the cache. A uniform bias is
  characterizable and correctable — a constant scale factor leaves relative
  comparisons intact — while a bias that varies by frame pair is neither.

  Storage: flow is `(2, H, W)` float32 at 6.48 MB/frame, *twice* the phase,
  against ~4 s per 1000 frames to regenerate on CUDA. Same conditional rule as
  the filter cache: regenerate while exploring, cache for training loops.

  For learned flow, prefer an unsupervised photometric objective (`warp_consistency`
  is differentiable on float frames) or the ground-truth generator below over
  classical pseudo-labels, which cap the model at its teacher.

## Implementation — modules to write

Roughly in dependency order: each one is easier once the previous has landed.

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

- **`data/transforms/` has landed, both kernels included.**
  `filtering/` (`kernel/` package + `sequence.py`) and `normalization.py` are
  written, tested, and driven from `scripts/optical_flow/`. What follows records
  what was settled, so the reasons outlive the code that now encodes them.

  > Quantitative claims from the benchmark are **provisional**: they come from
  > 20-frame excerpts, about one beat. Ranking flipped between raw and filtered
  > frames, so settle radii and parameters on a full dataset. The mechanical
  > facts below are exact.

  `filtering/kernel/` — a `FilterKernel` base (`base.py`) with two reductions beside
  it, `MedianKernel` (`median.py`) and `GaussianKernel` (`gaussian.py`). Both
  read a per-axis neighbourhood and **drop** out-of-range neighbours rather than
  pad them; they differ only in the reduction. Radius is written as `r`,
  `(r_spatial, r_temporal)`, or `(rx, ry, rz)` and normalized to the triple; a
  zero axis disables it, which the legacy could not express.

  The **median** keeps legacy semantics, verified against `scipy`: ellipsoid =
  offsets with `(dx/rx)^2 + (dy/ry)^2 + (dz/rz)^2 <= 1` (33 samples at radius 2
  against a cuboid's 125); with an even number of valid samples it **averages the
  middle two**, so `torch.median` — which returns the lower — cannot be used
  directly. The **gaussian** takes per-axis `sigma` + `truncate` (radius derived
  as `int(truncate * sigma + 0.5)`, scipy's rule) and is separable; dropping a
  border neighbour would darken the edge, so it divides by the weight that
  actually landed, one division at the end, which equals the full 3D normalized
  result exactly. Decided: torch only (no numba / scipy).

  Kernel arguments also exist as records (`MedianParams` / `GaussianParams`,
  `KernelParams.build()`), so a config or the cache sidecar carries settings
  without a live object and `FilteredSequence.from_params` dispatches through
  `build()` with no per-kernel branch.

  **Reversed: `FilteredSequence` owns its source and is randomly accessible.**
  The earlier note here banned random access, on the grounds that "frame i" would
  depend on which window asked for it. That is a property of a delay line, which
  cannot look back. Owning the source removes it — the window is always
  re-readable, so `filtered[i]` is the reduction over `source[i-rz .. i+rz]` and
  nothing else. Verified: a shuffled pass with repeats returns exactly what a
  forward pass returned. Memory is bounded the same way a delay line would bound
  it, at `2rz+1` frames, and measured flat from 100 to 2400 frames — the length
  of a sequence does not enter.

  Two costs come with it, both measured. A forward pass reads each source frame
  **once**; random access misses the buffer and reads about `2rz+1` times as much
  (5.7x at `rz=2`). And **one instance is not safe across threads** — concurrent
  `get_item` calls evict each other's buffered frames and raise `KeyError`,
  though never a wrong value. Worker *processes* each hold their own instance, so
  a `DataLoader` is unaffected; a thread pool is not. Neither is a defect to fix
  until something needs it: the contract is a sequential pass that builds a
  cache, and a shuffled consumer should read the finished cache. **Not
  implemented, recorded here on purpose.** When a thread-pool consumer does need
  it, a `threading.local()` buffer isolates threads while keeping the sequential
  one-read-per-frame; a lock (serialises `_window`) or a per-call buffer (drops
  the buffer's whole point) are the cruder fallbacks.

  **The temporal radius must scale with the frame rate.** Damage tracks the time
  the window spans, not its frame count, so the legacy's fixed `(2,2,2)` is a
  20 Hz value that over-smooths 10 Hz data — at 10 Hz it costs ~30% of the
  beating profile's relative amplitude while `rz=1` costs none. The deliverable
  *is* the beating profile, so that is signal loss, not denoising. Derive the
  radius from the beat period per dataset rather than freezing a constant, and
  read the real interval from each sample's `timestamps.txt` — the fixture names
  only claim "~10 Hz" and "~20 Hz".

  `normalization.py` — three modes on a pair-shaped API, emitting uint8.
  **`pairwise` cannot produce a single normalized frame list**: a frame is scaled
  by the joint range of whichever pair it is in, so it appears twice with two
  encodings — the API must be built around pairs (or windows) for the mode to
  exist at all. `perframe` is the unsafe mode: rescaling each frame by its own
  extremes breaks the brightness constancy every estimator assumes. The four
  scopes collapsed to three modes: `injected` takes a range measured outside the
  call, so sequence and dataset scope are the same code and the caller's choice.
  Applying stats is elementwise and local, so once they exist every mode is safe
  under random access; only computing them needs a pass.

  It **rounds where the legacy truncated**, moving half the pixels by one level
  of 256. That removes a systematic downward bias rather than adding noise, but
  quality columns from before the switch are not comparable at the fifth decimal.

  **`pairwise` forecloses the estimator's streaming path.** `push` retains the
  *normalized* previous frame, and under `pairwise` that encoding is stale by the
  time the next frame arrives — the two frames being compared end up on different
  scales, which is the brightness-constancy break `perframe` is rejected for.
  Measured: a frame's two encodings differ. So `pairwise` implies `calc`, and
  only `injected` is both safe and compatible with `push`. That trade is worth
  little at present — `push` measured 1.05x over `calc` on CUDA, because the
  host-device conversion it saves is small next to the flow itself.

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

  **The cache boundary is after filtering and before normalization**, in float32.
  This is forced, not chosen: `pairwise` gives a frame two uint8 encodings, so
  normalized frames cannot be stored at all without giving that mode up
  permanently. Storing float32 filtered frames also makes the cached form and the
  live `FilteredSequence` bit-identical, which is what lets them substitute for
  each other.

  That leaves three shapes a consumer may be handed — a raw bin folder, a
  `FilteredSequence` over one, or a bin folder of cached filtered frames — and
  they **do not share a type**. The bin folders yield `NDArray[float32]`;
  `FilteredSequence` yields `Tensor`, because it is also where the numpy → torch
  boundary happens to sit. `FrameNormalizer` takes tensors, so the two folder
  shapes need that conversion made explicit rather than inherited by accident.

- **One device per run, and watch the CUDA allocator once stages share a
  process.** Neither bites today; both do once `filter -> flow -> kinematics`
  runs as one chain.

  **Choose the device per run, not per stage.** `FilteredSequence` yields a
  `Tensor` and `tensor_to_gpumat` hands it to an OpenCV CUDA estimator without a
  copy -- `tests/common/test_cuda_utils.py:28` asserts the pointer is the same --
  so a filter and a flow on one device never move a frame. Splitting them costs
  3.24 MB per frame and, worse, the sync that transfer forces, which is the same
  stall `frame_ranges()` exists to remove. Worker count has no single answer
  across devices either: seven for seven GPUs, sixty-four for sixty-four cores,
  both measured. One pool cannot serve both. So when a stage cannot take the
  chosen device -- DeepFlow is CPU-only -- split the *runs* at the cache boundary
  instead of mixing devices inside one pool, and have the stage refuse the device
  loudly rather than fall back, since a silent fallback reads as "why is this
  slow" months later. Kinematics gets no say: differencing a `(2, H, W)` field
  costs less than moving it.

  **The allocator.** `MedianKernel.apply` holds `(S, H, W)` and `(S//2+1, H, W)`
  at once -- 1.11 GB and 557 MB at 900x900 with 343 offsets. Torch's caching
  allocator keeps freed blocks per segment, so a run of lighter kernels first
  leaves the pool holding plenty of memory in pieces too small to serve the next
  large request. It then calls `release_cached_blocks`, whose `cudaFree`
  synchronizes the device, and retries -- once per frame. That is the 60x above.
  `torch.cuda.memory_stats()["num_alloc_retries"]` is the tell; print it once
  rather than mistake the reading for the kernel's true cost, which is what
  happened here.

  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is the cheap fix and torch
  2.12 has it: segments grow in place, so no new contiguous `cudaMalloc` is
  needed. `scan_phase_range` is already safe without it, because
  `scan_sequences` builds and tears down its pool inside each job -- a
  `--multirun` sharing one parent process therefore carries no fragmentation
  between configurations. Keep that property.

  **OpenCV allocates separately.** `cv2.cuda` knows nothing about torch's cache,
  so torch can hold memory an estimator then fails to get; `tensor_to_gpumat`
  avoids copying the frame but not the estimator's own workspace. An
  `empty_cache()` at the *sequence* boundary -- never the frame boundary, which
  would perform the expensive release by hand every frame -- is where to give it
  back.

- **Rewrite the benchmark as `scripts/optical_flow/run_estimator.py`.**
  Estimators, `common/warp.py`, `optical_flow/evaluation.py` and
  `data/transforms/` are done; `benchmark_opencv.py` runs on them but scores a
  single sample against a CPU-vs-CUDA verdict. The replacement sweeps every
  sequence under a root, aggregating per sequence, which is the shape the
  parameter search needs.

  **Parallelise over sequences, not over frames.** Measured, batching buys
  nothing anywhere in this pipeline: filtering a batch of frames instead of one
  runs at 0.97-1.08x, because a single frame already saturates the device and
  there is no per-call overhead to amortise; `calc_batch` is a Python loop
  because OpenCV exposes no batched optical flow; and a `DataLoader`'s
  `batch_size` would only loop `__getitem__` anyway, since neither
  `FilteredSequence` nor `DataSequence` defines `__getitems__`. Sequences are
  independent, of uneven length, and there are dozens — that is where the
  parallelism is.

  Constraints that shape the worker API, all verified:

  - **No estimator survives a process boundary.** Every one fails to pickle on
    its `cv2` object. Workers must be handed a recipe — params, device, paths —
    and build their own. `FarnebackParams`, `MedianKernel` and `FrameNormalizer`
    all pickle, but a normalizer holds mutable state and a `FilteredSequence`
    carries both a warm buffer and a baked-in device, so neither should be sent
    either.
  - **A pool `initializer` is what makes a worker cheap**, building one estimator
    per worker rather than per sequence, and giving each worker its own device.
  - **Sort longest-first and dispatch dynamically.** Static length-balanced
    subsets and `imap_unordered` over a longest-first order measured identically
    (both 1.07x of a perfect split); the sort is one line where the split is a
    bin-packing function, and dynamic dispatch also absorbs estimate error.
    Unsorted costs 1.10x, longest-last 1.15x — so the sort earns its place, and
    the partitioning does not.
  - **Worker count follows the estimator's device**: cores for a CPU estimator,
    where the gain is near-linear, but only as many as there are GPUs for a CUDA
    one, which serialises on the device no matter how many processes queue on it.
  - **`hydra.instantiate` warns without a target whitelist (1.4).** Building a
    config-controlled `_target_` with no `_target_whitelist_` is deprecated,
    because the dotted path is arbitrary code — the config decides what gets
    imported and called. Our configs are our own, so the risk is low, but the
    driver's `instantiate` calls should pass a whitelist of the estimator/kernel
    classes they mean to build (or `UNSAFE_ALLOW_ALL_TARGETS` to keep the legacy
    any-target behaviour and silence the warning). The whitelist is the honest
    form; wire it in when `run_estimator.py` starts instantiating.

  Where the time actually goes, per frame pair: the flow dominates everything.
  CPU Dual TV-L1 is ~150x the median filter and ~15x CPU Farneback; on CUDA the
  ordering inverts against intuition, with `warp_consistency` costing ~10x a
  Farneback `calc`. Optimising preprocessing is not where throughput is.

  Optimization left on the table: pipeline an estimator's input conversion,
  `calc` and output over a `cv2.cuda.Stream` (today everything runs on the single
  default stream). And evaluation, now that it outweighs the flow it scores.

- **`hydra` decided: `hydra-core >= 1.4.0.dev6`, in a `scripts` dependency
  group.** The config-driven driver (`conf/` plus a `hydra.main` entry point)
  was committed unrunnable on Python 3.14: `hydra.main` builds an `argparse`
  parser with a non-string `help`, which 3.14 rejects, so `hydra-core` 1.3.4
  cannot start a job. 1.4.0.dev6 fixes it, and under it the config composes,
  config groups select, and `_target_` with `_partial_` builds each estimator
  with the script supplying `device` — the same recipe shape the parallel
  workers need. Of the three ways out — pin the dev release, drop to 3.13, or
  give up `--multirun` — the dev pin was taken: dropping the interpreter and
  losing the sweep both cost more than depending on a pre-release does. Only
  `scripts/` import `hydra`, not `iivs_cardio/`, so it lives in a PEP 735
  dependency group beside `dev` (`uv sync --group scripts`) rather than the
  runtime dependencies -- not published, so a group fits better than an extra,
  and a default sync stays lean and free of the pre-release. The spec is
  explicit about the pre-release, so `uv sync` resolves it without a
  project-wide prerelease mode. Revisit when a 1.4 stable ships.

- **Propagate the CHW tensor layout to the kinematic kernels.** The layout is
  settled — CHW (`(2,H,W)`/`(N,2,H,W)`), rationale and channel-first kernel
  sketches in [`docs/foundations.md`](docs/foundations.md) §2. The estimators and
  `iivs_cardio/common/warp.py` already follow it; the unwritten kinematic kernels
  must too. The ancestor `new-project-DESIGN.md` §4.1 is still channel-last —
  read it with that correction in mind.

## Evaluation — how a result earns trust

Measurement the analysis above leans on. Each entry exists because a proxy metric
has already misled this project at least once.

- **Promote the identity baseline and forward-backward error into
  `evaluation.py`.** Both are prototyped in `benchmark_opencv.py`, which the
  rewrite above will retire — move them before it goes. Its docstring explains
  why they are needed: with sub-pixel motion a zero flow already scores SSIM
  ~0.94, and SSIM gain alone rewards a flow that fits noise. Neither is safe —
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

  The same construction doubles as a **supervised training-data generator** for a
  learned flow model — real image statistics with exact labels — which is the
  alternative to training on classical pseudo-labels.

- **Add opt-in real-data tests over the fixtures.** The Koala time-lapses live in
  the private `iivs-lab/iivs-lib-fixtures` release (`gh release download v1 -R
  iivs-lab/iivs-lib-fixtures -D fixtures`); nothing fetches them automatically —
  `iivs-lib` has a `scripts/fixtures/fetch.py` + `lock.json` pair worth porting.
  Mirror its test pattern: a `conftest.py` fixture parametrized over the
  time-lapses present, so an absent folder skips rather than fails. Gitignore the
  directory at that point; it is ~1.2 GB. Worth asserting there, as no synthetic
  suite can: that a flow beats the identity baseline *and* stays
  forward-backward consistent.
