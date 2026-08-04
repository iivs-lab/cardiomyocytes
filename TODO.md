# TODO

Tracked items that are not yet captured in code or tests. Promote an
item to a CHANGELOG entry once it lands.

## In flight — `scan_phase`

Immediate work, unlike the design questions below. The script finds sources,
builds a `FilteredSequence` over each, ranges every frame, writes the filtered
frames a run asks for, and writes one JSON per run.

**The 2026-07-31 sweep and everything measured on it have been dropped.**
`Device`, the single traversal, the `mpire` pool and `pin_threads` all landed
after that run, so its timings describe a worker path that no longer exists and
its outputs a document shape that has since changed. What follows is what is
still to build or decide.

- **Turn device parity into a test, and cover the branch it misses.** Nothing in
  `tests/` compares a kernel across devices; copy the `requires_cuda` marker from
  `tests/common/test_cuda_utils.py`, which is there for OpenCV interop only.

  The interesting boundary is `sort` against `topk`. `MedianKernel` picks between
  them on offset count, and the median configs produce 7 to 343 offsets while the
  committed threshold is `range(33, 64)` -- so `ellipsoid (1, 1, 1)` at 7 offsets
  and `ellipsoid (2, 2, 2)` at 33 land on opposite sides. Border pixels carry NaN
  padding and an even valid count, so a test over them also exercises the NaN
  ordering and the average-the-middle-two path. Compare whole tensors with
  `torch.equal` at offset counts either side of the boundary.

- **Batch the range's device sync.** `finite_range` reads `low.isfinite()` per
  frame, which pulls a value to the host and stops the CPU reading the next frame
  while the GPU works. A `FilteredSequence.frame_ranges()` that launches every
  `aminmax` first and transfers once removes the stall: the scalars are 8 B a
  frame, so a 1200-frame sequence holds 9.6 KB, and CUDA's launch queue
  self-throttles rather than growing without bound. The non-finite fallback cannot
  branch per frame without the sync it exists to remove, so it becomes a second
  pass over only the frames whose bounds came back non-finite -- in practice none,
  and `numel()` syncs nothing.

  **It pulls against the shared traversal.** Writing a frame moves it to the host,
  which is the same sync the batching removes, so on a `save_frames` run the stall
  returns regardless and the batching pays only on a range-only run.

- **Drop `indent=2` from the range document.** One character, a large fraction
  of the file, and no time cost. It is disk, not throughput, so it only matters
  once the documents are kept.

- **A per-worker progress bar, on top of the pool's own.** The `mpire` pool has
  landed with `progress_bar` / `insights` / `worker_lifespan` behind `compute.*`,
  and both paths now draw the pool-level bar; this is the piece left. `worker_init`
  opens a `tqdm(position=worker_id + 1, leave=False)` -- the pool's bar holds
  position 0 -- and each task resets it to the sequence length and renames it;
  `worker_exit` closes it. Verified rendering seven independent lines under spawn.

  Worth it only for the heaviest kernels: a config that finishes a sequence every
  few seconds already moves the sequence-level bar, where the heaviest leave it
  still for minutes and look hung. Keep it a config flag, since seven lines
  collide with anything else on the terminal.

  `tqdm` does **not** go quiet off a tty: a redirected bar writes a redraw line
  per update into the log, so `compute.progress_bar` defaulting to true costs
  every `--multirun` job that noise. Decide the default against a real sweep.

- **Nothing logs.** `hydra` opens a `scan_phase.log` in every job directory and it
  is 0 bytes on every run: the script writes nothing, and `report_insights`
  `print`s, which hydra does not capture. So a sweep leaves no record of which
  sequences a job read, how long each took, or why one failed -- only the range
  document, which a run that dies never reaches. `benchmark_opencv.py` already
  takes a `logging.getLogger(__name__)` and hydra configures the root handler, so
  a module logger is all a script needs. What to settle is the level split: per
  sequence at `INFO`, per frame at `DEBUG` where 1200 frames a sequence would
  otherwise swamp the file, and `report_insights` routed there instead of stdout.
  A worker logs from its own process, so the handler has to be one a spawned
  process inherits or rebuilds.

- **One bad sequence must not cost the run.** `scan_sequence` raises on a frame
  holding no finite value, and `save_phase_bin_folder` raises when the destination
  exists and `target.overwrite` is false. Either ends everything: `mpire` re-raises
  a task's exception in the parent and tears the pool down, so every sequence
  already finished is lost to the one that failed -- after hours, on the full
  dataset. The scan should carry a failure as a result rather than an exception,
  finish what can be finished, and report at the end which sequences were skipped
  and why.

  **A partial run must say so.** A range folded over a subset is not the dataset's
  range, and a consumer setting a normalization policy from it would be reading a
  hole as data. Whatever the document gains -- a skipped list, a count against the
  number found -- has to be something a reader cannot miss.

- **The source search is moving to `iivs-lib`.** Decided, not yet scheduled.
  `search_phase_bin_folders` already lives there (`iivs.dhm.data.phase.layout`);
  what would follow it is the layer `search_sources` wraps around it -- the
  `include` / `exclude` selection, the unit override, and the two failures it
  tells apart. Move it before a second script grows its own copy.

- **Not every config group takes the short override form.** `compute=cpu` works
  because the group and its package are both `compute`, but the filter group is
  mounted at another package and needs
  `data/transforms/filtering@filter=<name>` -- plain `filter=<name>` is read as a
  value and fails at instantiation. A job's exact invocation is recoverable from
  its own `.hydra/overrides.yaml`.

- **Deferred by decision, recorded so they are not rediscovered.** Renaming the
  `Params` suffix to `Config` on `KernelParams` and the optical-flow params
  classes. Reading frames on multiple threads while ranging them. Putting
  `FrameShapedMixin` / `ValueRangeMixin` on `FilteredSequence` — the latter is
  unusable on tensors as written, since it tests emptiness with `finite.size == 0`
  and a `Tensor`'s `size` is a method, so the comparison is never true, which is
  why `finite_range` asks `numel()`.

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

- **Whether the value range can discriminate between filters is open again.**
  The full-dataset answer was produced by the retired sweep, so it goes with it.
  Two questions it raised stand on their own and should be asked of the next run:
  whether the range separates the configurations at all, and how a single hot
  pixel in one sequence moves a dataset-wide bound -- which is what a
  normalization policy has to survive. Every document carries per-sequence
  ranges, so both are answerable from the outputs without a second pass.

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

- **One device per run, and watch the CUDA allocator once stages share a
  process.** Neither bites today; both do once `filter -> flow -> kinematics`
  runs as one chain.

  **Choose the device per run, not per stage.** `FilteredSequence` yields a
  `Tensor` and `tensor_to_gpumat` hands it to an OpenCV CUDA estimator without a
  copy -- `tests/common/test_cuda_utils.py:28` asserts the pointer is the same --
  so a filter and a flow on one device never move a frame. Splitting them costs
  3.24 MB per frame and, worse, the sync that transfer forces, which is the same
  stall `frame_ranges()` exists to remove. Worker count has no single answer
  across devices either -- one per GPU against one per core -- so one pool cannot
  serve both. So when a stage cannot take the
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
  synchronizes the device, and retries -- once per frame, which is enough to read a
  kernel at many times its true cost.
  `torch.cuda.memory_stats()["num_alloc_retries"]` is the tell; print it once
  rather than mistake the reading for the kernel's true cost, which is what
  happened here.

  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is the cheap fix and torch
  2.12 has it: segments grow in place, so no new contiguous `cudaMalloc` is
  needed. `scan_phase` is already safe without it, because
  `scan_sequences` builds and tears down its pool inside each job -- a
  `--multirun` sharing one parent process therefore carries no fragmentation
  between configurations. Keep that property.

  **OpenCV allocates separately.** `cv2.cuda` knows nothing about torch's cache,
  so torch can hold memory an estimator then fails to get; `tensor_to_gpumat`
  avoids copying the frame but not the estimator's own workspace. An
  `empty_cache()` at the *sequence* boundary -- never the frame boundary, which
  would perform the expensive release by hand every frame -- is where to give it
  back.

- **`run_estimator.py` holds four real defects until that rewrite lands**,
  surfaced once `ty` began checking `scripts/`: a mismatched return type at
  `:39`, both arguments to `FrameNormalizer.apply` wrong at `:52` (against
  `iivs_cardio/data/transforms/normalization.py:142`), and a wrong argument to
  `process_sequence` at `:76`. `ruff` adds an unused loop variable and two flows
  computed and dropped. They are the whole of what `ruff check` and `ty check`
  report today, so the two commands only read as clean by comparison until this
  file is replaced.

- **The per-sequence pipeline and the dataset-level containers are being
  redesigned from the problem up.** What is in the tree -- `Slot`, `Node`,
  `Steps`, `FieldWriter`, the range collectors -- came out of a conversation
  that kept accreting rather than converging, and the records it left in
  `docs/foundations.md` were steering later work toward it. Those records are
  gone; the code is still there and passing, and is to be judged fresh rather
  than extended.

  What a redesign still has to answer, none of it settled: what a dataset-level
  container owns and whether it is the same object that dispatches the run; how
  a per-sequence result crosses back from a worker; whether there is one
  container per concern or one holding several; and where the temporal window
  lives. `scripts/data/scan_phase.py` is the only driver on it, so it is the
  only thing to keep working.

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
