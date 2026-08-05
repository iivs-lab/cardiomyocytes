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
  which keeps a masked frame's NaN background from swallowing the range.
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
- `preprocess_phase` writes the filtered frames when `target.save_frames` is set.
  `target.overwrite` decides whether an output already under `target.root` may
  be replaced. `FrameDestination` carries what every sequence's writer shares
  and `PhaseStageFactory` hands out a stage with its hooks already registered,
  so whatever owns a stage never wires that stage's side branches.
- `preprocess_phase` refuses `target.save_frames` in a `--multirun`. Frames go to
  `target.root`, which carries no job number, so every job of a sweep would write
  the one tree: 1.45 TB apiece, in turn, and nothing in the tree would say which
  config finally left it there. `is_multirun` is what answers the question.
- `mpire` and `tqdm` in the `scripts` group, and `compute.progress_bar` /
  `compute.insights` / `compute.worker_lifespan` to drive the pool.
- `pin_threads`, holding each worker to `torch`'s default thread count divided
  by the worker count. Every process otherwise sizes its pool to the whole
  machine and they contend: measured on 64 cores, sixteen unpinned workers ran
  2.7x slower than no pool at all. A lone worker keeps the machine to itself.
- `iivs-lib[torch]>=0.2.0` as a dependency, for phase sequence IO. The `torch`
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
- `preprocess_phase` writes the range document again, under
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
- `plan_devices` refuses the knob that belongs to the other device -- `workers`
  under CUDA, `gpu_ids` under CPU -- rather than dropping it silently. A worker
  count a CUDA run cannot honour is a wall clock several times what the caller
  planned for, with nothing saying why; `gpu_ids` defaults to `None` so that a
  value the caller wrote can be told from one they did not.
- `preprocess_phase` runs its workers through `mpire`, which retires
  `_WORKER_DEVICE`, the `SimpleQueue` that filled it and `_adopt_device`: a
  worker picks its device out of `shared_objects` by worker id, and binds the
  process to it per task rather than once, which `activate` is cheap enough for.
- `source.unit` is `source.phase_unit`, beside `frame_step`: both say how to read
  a sequence, where the fields above them say which sequences to read.
- `search_sources` tells its two failures apart -- finding nothing under `root`,
  and filtering everything out -- since they are fixed differently.
- The script helpers split by what they know: `scripts/_hydra.py` holds the
  hydra boundary (`apply_schema`, `output_directory`, `is_multirun`),
  `scripts/_compute.py` the machine's division (`ComputeConfig`, `plan_devices`,
  `pin_threads`, `report_insights`), and `scripts/data/_range.py` the shape a
  run reports its value ranges in (`ValueRange`, `CompositeRange`, `FrameRange`,
  `SequenceRange`, `DatasetRange`, `as_dict`). `scripts/_config.py` is gone into
  the first. The range types are a document schema rather than library code --
  each `source` is a path relative to something only the document fixes -- which
  is why they sit beside the scripts that write them and not under
  `iivs_cardio/`, whose own currency for a range is a `tuple[float, float]`.
- `FilteredSequence` absorbs `FrameSequence`, taking its frame `step` and its
  typed view of the source; `iivs_cardio/data/sequence.py` is gone with it.
  `step` is applied before filtering, as it was, and `origin` returns the source
  as the type it was given -- so a caller needing a phase folder's header reaches
  it without carrying a second reference. The wrapper was generic in name only:
  it composed a filter, so no flow or kinematics sequence could ever have used
  it. `FrameSequence.value_range` goes too, left without a caller once the scan
  took to ranging in its single traversal.

- `kaparoo-python` minimum raised to `0.11.1`, which adds
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
