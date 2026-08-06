# Changelog

All notable changes to this project will be documented in this file.

The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `Device`, a compute device as every library in this stack must agree to see
  it, with `DeviceLike = str | torch.device | Device` for what a caller may
  write. `Device.activate` points `cv2.cuda` and CuPy at the same GPU as torch;
  each of those keeps a process-global current device, and CuPy's was never set,
  so it stayed on device 0 while `gpumat_to_cupy` wrapped pointers cv2 had
  allocated elsewhere.
- `finite_range` in `iivs_cardio/common/range.py`, the `(min, max)` of a frame's
  finite values or `None` when it has none, so a caller holding a frame can range
  it without a second read. Non-finite values are ignored rather than propagated,
  which keeps a masked frame's NaN background from swallowing the range -- that
  tolerance is this function's, not the pipeline's, which refuses one on the way
  in.
- The pipeline model in `iivs_cardio/common/pipeline.py`. `Step` carries an
  index, the value it describes or its absence, and whatever the stream itself
  does not consume; `Stage` is an indexed source that fills each step once and
  keeps a window of them, so one filtered frame reaches every consumer that asks
  for it rather than being computed per consumer (+94% for a second read at
  median `(2,2,2)`). Hooks are the side branches, firing once per index however
  often a step is asked for, and `Stage.run` opens every hook in the graph
  together -- a stage feeding two others is reached along both paths and still
  opened once, and a failure part-way leaves no folder behind rather than some
  of them. `SequenceStage` adapts any `DataSequence`, which is what makes an
  online source and a cached folder the same thing to whatever reads them.
- `KoalaFrameWriter`, writing one frame per step into a numbered folder the
  Koala readers discover by, staged and moved into place on a clean exit. The
  push-shaped counterpart of `save_koala_frames`: the per-file format arrives as
  `save`, so one class serves phase, flow and metric frames, and a gap in the
  indices is refused rather than closed. `phase_frame_writer` binds the phase
  `.bin` format to it, taking the calibration as numbers rather than reaching
  into a reader for them.
- `preprocess` writes the filtered frames when `target.save_frames` is set.
  `target.overwrite` decides whether an output already under `target.root` may
  be replaced. `FrameTree` carries what every sequence's writer shares
  and `PhaseStageFactory` hands out a stage with its hooks already registered,
  so whatever owns a stage never wires that stage's side branches.
- `preprocess` refuses `target.save_frames` in a `--multirun`. Frames go to
  `target.root`, which carries no job number, so every job of a sweep would write
  the one tree: 1.45 TB apiece, in turn, and nothing in the tree would say which
  config finally left it there. `is_multirun` is what answers the question.
- `StageFactory` in `iivs_cardio/common/pipeline.py`, the whole of what a
  driver sees: how many items, what each is called, how to run one on a device,
  and how to bracket the run. It carries the stage's own `name`, which the job
  assigns -- the same filtered-phase run is preprocessing under one pipeline and
  postprocessing behind a hologram reconstruction, so the machinery cannot claim
  a name for itself. `run_all` reads that name instead of taking its own, which
  is what keeps a parent and its workers from filing one run under two.
- `Reporting`, a side branch's one line for whoever is logging. A branch runs
  under whichever stage registered it and has no logger of its own to use, so it
  returns the sentence and the stage writes it: `wrote 5 frames`,
  `measured [0.80, 2.52] across 5 frames` per sequence, and
  `wrote value_range.json from 4 sequences: [0.61, 2.52]` once the run is over.
  Both fire *after* the commit, so what they say is what was committed.
- Logging, under the stage's name rather than the module's. A worker opens
  `<job>/worker<id>.log` for itself in `worker_init` -- several processes
  appending to one file interleave, and on Windows tear -- and appends, since
  `compute.lifespan` retires a worker and starts a fresh one under the same id.
  One file per worker rather than per stage, so a job that filters and then
  estimates reads in order. The parent's own file keeps the configuration, the
  per-item verdicts and the summary; `report_insights` goes there too, and says
  so when a lone worker means no pool collected any.
- `IncompleteRunError`, raised once every other item has finished. `mpire`
  re-raises a task's exception in the parent and tears the pool down, so a run
  that let one through would lose every sequence still to come -- hours of
  finished work at dataset scale. A worker returns its failure as a result
  instead, and the error carries `skipped` and `total` whole rather than folded
  into a message a retry cannot read back.
- A `coverage` block in the range document -- `{covered, total, skipped}`,
  written immediately ahead of `dataset`. Bounds folded over a subset are not
  the dataset's, and a consumer setting a normalization policy from them reads a
  hole as data; the block is written even when nothing is missing, so the
  absence of a key can never be mistaken for completeness. `skipped` is derived
  from the roster against the parts on disk rather than passed in, since a
  sequence is missing whether it raised, died with its worker, or never ran.
- `mpire` and `tqdm` in the `scripts` group, and `compute.progress_bar` /
  `compute.log_insights` / `compute.lifespan` to drive the pool.
- `pin_threads`, holding each worker to `torch`'s default thread count divided
  by the worker count. Every process otherwise sizes its pool to the whole
  machine and they contend: measured on 64 cores, sixteen unpinned workers ran
  2.7x slower than no pool at all. A lone worker keeps the machine to itself.
- `iivs-lib[torch]>=0.3.1` as a dependency, for phase sequence IO. The `torch`
  extra additionally enables `iivs.dhm.analysis.pytorch` and
  `iivs.common.data.pytorch`.

### Changed

- Every layer stores a `Device` where it stored a `torch.device`; torch calls
  take `.as_torch`. A malformed spec now raises `ValueError` like every other
  rejection there, where `torch.device` let a `RuntimeError` through, and
  `Device.resolve_all` asks the driver only when a CUDA device is named.
- `cuda_utils` rejects a rank it cannot mean rather than diagnosing it three
  wrong ways -- a bare unpack error, a channel count the array never had, or a
  broadcast failure at the assignment -- and `_cv_type` names the pairs it takes
  from the table beside it.
- A filter is recorded by what it was rather than by which class did it:
  `KernelConfig.kind` is a declared `"median"` / `"gaussian"` / `"identity"`,
  where the record used to carry the `_target_` import path -- which made two
  runs that filtered identically compare unequal as soon as the code moved.
  `describe_filter_kernel` reads it off the built config rather than the
  config node, so a config `instantiate` would reject fails before a document
  records it.
- `preprocess` writes the range document again, under
  `target.save_ranges` / `target.range_file`. `RangeDocument` is a side branch
  with a lifetime: it hands each sequence a `SequenceRangeMeter`, and folds what
  the meters left once every sequence has run. Each meter writes its own
  `SequenceRange` into a `<document>.parts` folder on a clean exit, which is how
  an answer leaves a worker at all -- `shared_objects` travels one way, so a
  copy the workers filled never comes home. A traversal that died writes
  nothing, since a prefix folded into the dataset's bounds is a hole nobody
  would see; the parts that did finish stay behind, and entering clears them so
  a re-run into the same output directory cannot fold an earlier run's answers
  in with its own.
- `SideBranch` in `iivs_cardio/common/pipeline.py`, what a stage factory holds:
  something that hands a sequence a hook. Each one asks for the narrowest
  view it can work from -- a range document takes anything with a `name`, so
  it never learns what a phase folder is -- and the container's declared type
  is what they are checked against. Being a context manager is
  optional and is what tells the two apart -- a frame writer commits itself per
  sequence, where a range document only finishes once every sequence has, and
  `StageFactory.running` brackets the ones that need it exactly as `Stage.run`
  brackets the hooks a level down.
- Non-finite values are refused rather than carried. `FilteredSequence` checks
  each source frame where it reads it -- the one place every frame passes exactly
  once, and the last place it is still on the host -- and `phase_frame_writer`
  writes with `on_nonfinite="raise"`. The formats this project reads and writes
  store a NaN happily, so the refusal has to be its own. It also retires a crash
  rather than patching one: `MedianKernel` pads out-of-range neighbours with NaN
  and counts the valid ones, and a frame whose whole neighbourhood was NaN drove
  that count to zero and the gather to index `-1`. With finite input the centre
  is always valid, so the case cannot arise.
- A sequence's range part is filed at `<document>.parts/<name>.json` with the
  name's own nesting, mirroring the frame tree written beside it, where it was
  percent-encoded into one flat file name. Entering prunes the folders its own
  parts emptied rather than the folder itself, so a path a caller pointed the
  document at keeps whatever else is in it.
- `build_filter_config` instantiates with `_convert_="all"`. Hydra hands its own
  containers through otherwise, so a `radius` arrived as an omegaconf list and
  the config was no longer the plain frozen record `asdict` and `json.dumps`
  take it for -- which surfaced only when a real run wrote a document, since
  tests build the config directly.
- `plan_devices` refuses the knob that belongs to the other device -- `workers`
  under CUDA, `gpu_ids` under CPU -- rather than dropping it silently. A worker
  count a CUDA run cannot honour is a wall clock several times what the caller
  planned for, with nothing saying why; `gpu_ids` defaults to `None` so that a
  value the caller wrote can be told from one they did not.
- `preprocess` runs its workers through `mpire`, which retires
  `_WORKER_DEVICE`, the `SimpleQueue` that filled it and `_adopt_device`: a
  worker picks its device out of `shared_objects` by worker id, and binds the
  process to it per task rather than once, which `activate` is cheap enough for.
- `source.unit` is gone rather than renamed: a run always reads radians. Every
  metric downstream reads optical path difference out of phase, so a cache in
  degrees or nanometres is one every consumer would have to convert back, and
  the header still carries `height_scale` for whoever wants metres.
- `target.range_file` defaults to `value_range` rather than `phase_range`. The
  document holds whatever a stage measured, and (2) and (3) will write one too.
- `search_sources` tells its two failures apart -- finding nothing under `root`,
  and filtering everything out -- since they are fixed differently.
- The script helpers split by what they know: `scripts/_hydra.py` holds the
  hydra boundary (`apply_schema`, `output_directory`, `is_multirun`),
  `scripts/_compute.py` the machine's division (`ComputeConfig`, `plan_devices`,
  `pin_threads`, `report_insights`). `scripts/_config.py` is gone into the
  first.
- The pipeline pieces sit under `iivs_cardio/data/pipeline/`, leaving
  `scripts/data/` with the two hydra bridges (`_filtering.py` for the kernel,
  `_process.py` for the stages) and one entry point. The range types moved with
  them: they were kept beside the scripts because `source` is a path relative to
  something only the document fixed, and `PhaseFilteredSequence.name` fixes it
  in code now. `preprocess_phase` is `preprocess`, since what it preprocesses is
  the config's to say.
- `FilteredSequence` absorbs `FrameSequence`, taking its frame `step` and its
  typed view of the source; `iivs_cardio/data/sequence.py` is gone with it.
  `step` is applied before filtering, as it was, and `origin` returns the source
  as the type it was given -- so a caller needing a phase folder's header reaches
  it without carrying a second reference. The wrapper was generic in name only:
  it composed a filter, so no flow or kinematics sequence could ever have used
  it. `FrameSequence.value_range` goes too, left without a caller once the scan
  took to ranging in its single traversal.

- `kaparoo-python` minimum raised to `0.12.0`; `0.11.1` is what added
  `DataSequence._normalize_index`. `FilteredSequence` calls it instead of
  carrying its own copy, so a negative index now reports
  `index -7 out of range for length 6` rather than `... for 6 frames`.
- `backward_warp` / `BackwardWarp` sample at `grid + offset` instead of
  `grid - transform`, and the second parameter is renamed accordingly. A forward
  optical flow is now passed unchanged rather than negated at every call site,
  which removes the bare sign flip that this project has already been bitten by;
  `offset` rather than `flow` keeps `common/` free of optical-flow vocabulary
  while stating the sign in the name. To displace an image *by* a field, negate
  it.
- `warp_consistency` / `WarpConsistency` now reconstruct `frame1` by sampling
  `frame2` at `grid + flow`, instead of reconstructing `frame2` from `frame1` at
  `grid - flow`. The forward flow is defined on `frame1`'s grid, so this
  direction needs no inverse and is exact; the previous one approximated the
  inverse with an error growing as `|flow| * |grad flow|`. Scores shift only in
  the 5th decimal at sub-pixel motion, and estimator rankings are unchanged.
