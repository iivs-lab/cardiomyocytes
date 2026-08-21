# Changelog

All notable changes to this project will be documented in this file.

The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `configs/experiment/`, where a sweep is named once rather than typed out per
  invocation. `filters.yaml` holds every option of the filter group, so
  `--multirun +experiment=filters` runs the search that was 470 characters of
  comma-separated names before. An experiment is a `@package _global_` overlay
  that fills no key of its own: it writes `hydra.sweeper.params` and leaves the
  rest of the config standing, which is why it does not sit under the group it
  varies -- a sweep can name two keys at once, and one that varied an estimator
  and a cache would have no group folder to live in.
- `sweep_parameters`, the settings a composed experiment would vary, and a
  refusal in the preprocess script for the one shape that fails quietly:
  `+experiment=...` written without `--multirun` runs once on the defaults,
  sweeping nothing and saying nothing. The sweeper is what reads those params,
  and a lone run never looks at them.

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
- `preprocess` writes the filtered frames when `target.frames.save` is set.
  `target.frames.if_present` decides whether an output already under the job's
  own directory may be replaced, and `target.frames.subpath` where a written
  sequence keeps its frames, defaulting to wherever the source keeps its own.
  Naming one is what lets a run write beside the frames it read --
  `Phase/Float/FilteredBin` next to `Phase/Float/Bin` -- rather than over them,
  and a run whose frames would land on the source is refused before the search:
  a sequence is committed by replacing its folder whole, so writing back to
  where it was read from destroyed the acquisition under a run that logged `N of
  N done` and exited 0. Either subpath is refused outright if it could reach
  outside a sequence's own folder, since a `..` would walk past that comparison
  and land wherever it pointed. `FrameTree` carries what every sequence's writer
  shares and `SequenceStageFactory` hands out a stage with its hooks already
  registered, so whatever owns a stage never wires that stage's side branches.
- `preprocess` refuses `target.frames.save` in a `--multirun`. Each job writes a
  tree of its own at 1.45 TB apiece, which is not what a sweep is for: the
  filter is what it varies and only the winning one is worth keeping, so the run
  says so rather than filling the disk in turn. `is_multirun` is what answers
  the question.
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
  `compute.tasks_per_worker` retires a worker and starts a fresh one under the
  same id.
  The files are named for the run, as `preprocess.worker0.log` beside the
  parent's own `preprocess.log`, so two runs pointed at one folder keep their
  own: a worker id says nothing about which run opened it, and a stage that
  filters and one that estimates hold different configurations. `run_all`
  refuses a folder named for another run, since the two names coincide only
  because a script passes one constant to both. It says which worker logs an
  earlier run left rather than deleting them, since only the job knows whether
  another stage wrote them. The level is padded in that format and in hydra's
  own, so the message column holds where a `WARNING` follows an `INFO`. The
  parent's own file keeps
  the configuration, the per-item verdicts and the summary; `log_insights` goes
  there too, and says so when a lone worker means no pool collected any.
- `IncompleteRunError`, raised once every other item has finished. `mpire`
  re-raises a task's exception in the parent and tears the pool down, so a run
  that let one through would lose every sequence still to come -- hours of
  finished work at dataset scale. A worker returns its failure as a result
  instead, naming the item it finished whether it failed or not, so nothing is
  inferred from the order results come back in. The error carries `failed` and
  `total` whole rather than folded into a message a retry cannot read back, and
  a side branch raising on its way out no longer replaces that verdict: once
  every item has been seen, the closing failure is logged and the verdict still
  rises.
- A `coverage` block in the range document -- `{found, selected, covered,
  skipped}`, written immediately ahead of `dataset`. Bounds folded over a subset
  are not the dataset's, and a consumer setting a normalization policy from them
  reads a hole as data; the block is written even when nothing is missing, so
  the absence of a key can never be mistaken for completeness. The three numbers
  narrow in turn -- found by the search, selected by `include` / `exclude`,
  covered by the parts that arrived -- which is what tells a document describing
  part of a dataset from one describing a smaller dataset. `skipped` is derived
  from the contents against the parts on disk rather than passed in, since a
  sequence is missing whether it raised, died with its worker, or never ran, and
  `Coverage` refuses a set of numbers no run could have had.
- `mpire` and `tqdm` in the `scripts` group, and `compute.show_progress` /
  `compute.measure_workers` / `compute.tasks_per_worker` to drive the pool. The
  pool starts its workers with `spawn` rather than the platform's default, which
  is `fork` on Linux: `run_all` plans its devices before the pool starts,
  and a worker forked after this process has touched CUDA inherits a context it
  cannot use, so every item of a multi-GPU run would have failed. `Device`
  counts the GPUs without initializing CUDA here for the same reason.
- `pin_threads`, holding each worker to `torch`'s default thread count divided
  by the worker count. Every process otherwise sizes its pool to the whole
  machine and they contend: measured on 64 cores, sixteen unpinned workers ran
  2.7x slower than no pool at all. A lone worker keeps the machine to itself.
- `iivs-lib[torch]>=0.3.1` as a dependency, for phase sequence IO. The `torch`
  extra additionally enables `iivs.dhm.analysis.pytorch` and
  `iivs.common.data.pytorch`.
- `all_finite` in `iivs_cardio/common/range.py`, beside `finite_range` and
  reading a frame the same way: `aminmax` in one fused pass, where NaN reaches
  both bounds and an infinity lands on the one it belongs to. `isfinite().all()`
  answers the same question through a mask the size of the frame, which on the
  path every source frame takes is an allocation worth not making (672 -> 247 us
  per 900x900 float32 frame, against ~3000 us of filtering).

- `check_compute_env` speaks for CuPy, the third thing in this stack that
  reaches CUDA and the only one with no CPU build to fall back on. Two
  arithmetic checks for different reasons -- CuPy builds its kernels at first
  use rather than shipping them, so a toolkit the driver disagrees with fails
  there and nowhere above, and the matmul goes through cuBLAS and back to the
  host -- and two for what this project actually asks of it: `cuda_utils` wraps
  a tensor as an array and back with no copy, which holds only while torch and
  CuPy read one `__cuda_array_interface__`, so the device pointer is what is
  compared rather than the values, which would agree either way. The linkage
  line reports the CUDA runtime CuPy found, which is not obliged to be the one
  torch was built against.
- `check_compute_env` names the cuDNN trees a Linux run mapped, read off
  `/proc/self/maps`. One soname is loaded once per process, so the sub-libraries
  torch asks for can come from two trees at once -- a system copy that arrived
  first and the wheel's own for everything it did not carry -- which is the
  mismatch it dies on, and is visible as nothing more than two parent
  directories among the mapped files. The section runs last, since it can only
  report what the ones above loaded, and off Windows alone, which is where
  `/proc` is.
- `iivs_cardio/beating_profile/`, a sketch of the field graph a profile is
  folded from: `base.py` is where a field declares what it needs of its
  neighbours and how far either side it reads, `graph.py` folds those into an
  order and into the frames each end of a sequence gives up, and `fields.py`
  holds the metrics. Asking for force is asking for five computations rather
  than one. A scaffold rather than a design: nothing re-exports it and nothing
  tests it.
### Changed

- The OpenCV estimators are three layers where they were an inheritance chain: a
  config holding the parameters as a value, a `Backend` holding the cv2 algorithm
  and the `Device` it was made on and making the calls, and `OpenCVEstimator`, the
  one streaming interface over any of them. `Farneback`, `DualTVL1` and `DeepFlow`
  were estimator subclasses whose whole content was a factory method and an
  `__init__` that had to set `self.config` before `super().__init__`, the base
  constructor calling the abstract factory; they are `FarnebackConfig`,
  `DualTVL1Config` and `DeepFlowConfig` now, and `build` is written once rather
  than three times. `SUPPORTED_DEVICES` moved to the config, DeepFlow having no
  CUDA implementation being a fact about the algorithm rather than about the
  machinery that streams frames through it. hydra configs are untouched:
  `_target_: ...FarnebackConfig` already named the config.
- `Backend` is `CPUBackend` and `CUDABackend`, which leaves every device branch in
  one layer. The estimator held six `is_cuda` branches, five attributes only one
  device ever set, and two `cast`s over the union of cv2's two algorithm types;
  it now has none of them, each backend taking the concrete cv2 type it calls.

- Every layer stores a `Device` where it stored a `torch.device`; torch calls
  take `.as_torch`. A malformed spec now raises `ValueError` like every other
  rejection there, where `torch.device` let a `RuntimeError` through, and
  `Device.resolve_all` asks the driver only when a CUDA device is named.
- `cuda_utils` rejects a rank it cannot mean rather than diagnosing it three
  wrong ways -- a bare unpack error, a channel count the array never had, or a
  broadcast failure at the assignment -- and `_cv_type` names the pairs it takes
  from the table beside it.
- The filter group is `configs/filter/`, so an override selects from it the way
  `compute=cuda` does: `filter=median_cuboid_2x2x2`, and `identity` for none.
  It sat under `configs/data/transforms/filtering/` mirroring the package, and
  the short form a reader infers from `compute` was a *value* override there --
  hydra accepted it because `filter` was a declared key, put the string in, and
  `instantiate` refused it several frames later. A group's folder is named for
  the key it fills now, which is what makes the two forms one. `filter=null` is
  gone with it, since a group override takes no null: `filter=identity` builds
  the same `IdentityConfig` and can sit in a sweep list, and `~filter` drops the
  default outright.
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
  is always valid, so the pipeline cannot reach it; the index is clamped anyway,
  since `apply` is a public function of its arguments and on CUDA that gather is
  a device-side assert that takes the worker's context with it.
- A run reuses what an earlier one left, through a block per branch:
  `if_present` (`error | overwrite | reuse`) and `if_unsourced` (`keep |
  delete`) sit on `target.frames` and `target.ranges` alike, the second for an
  output whose sequence the dataset no longer holds. The two axes are separate
  because they answer different questions -- what to do with the output of a
  sequence being written, and what to do with one nothing will be written for --
  and a run may want either without the other.
- A range document counts against the whole dataset rather than against the run
  that wrote it. `RangeDocument` takes a contents of every sequence the source
  holds against the frames each would be measured over, and `selected` names
  which of them this run was given; `Coverage` gains `reused` and `unselected`
  beside `skipped`, and every sequence the source holds is in exactly one of
  the three. Counting against the selection said `31 of 31` for a document
  whose bounds came from 121 -- and folding parts an earlier run left is what
  reuse does, so the two had to be told apart. The fold takes only parts the
  contents names whose settings match this run, which is what lets one left by a
  larger dataset or a different filter stay on disk without reaching the
  document. Parts carry their own `settings` for that comparison, since a part
  outliving its document is the case that needs it.
- A run that gave up still writes its document. `RangeDocument.__exit__` used to
  return without saving when the run raised, so a worker dying at sequence 90 of
  121 left the healthy 90 on disk with nothing to read them by; the document now
  says what it covers and names what it does not. A run that covered nothing
  writes `coverage` with no `dataset` block, since a range over nothing has no
  bounds to invent -- `to_range` returns `None` there rather than refusing.
- `FrameTree` has a lifetime, not only writers. Closing it walks the output tree
  for what a killed worker staged and the empty shells above it, which the
  writer only collects while it is alive, and for folders whose sequence the
  source has lost. Both were previously nobody's to collect.
- A side branch may decline an item: `SideBranch.get_hook` returns `Hook | None`,
  and a sequence every branch declines is not read at all. `StageFactory.
  run_stage` says whether it read one, and the run's summary splits three ways
  (`121 of 121 ready in 4.2s (90 reused, 31 computed)`) -- `done` counted only
  what was computed, which read as a smaller run than it was.
- Outputs with no sequence behind them are always named, whatever the policy
  then does with them: a dataset that shrank and a share that came up half make
  the same absence, and only whoever started the run can tell them apart.

- Branches that could not all commit take back what did. Closing them
  in turn is not one commit, so the range document's part reached disk and the
  frame tree failed to move a moment later, leaving a part standing for a
  sequence with no frames -- and a part is what `coverage` counts as covered,
  so no number in the finished document showed it. `Reverting` is the hook that
  can undo a clean close, `close_together` calls it on everything that closed
  before the failure, and `SequenceRangeMeter` removes the part it wrote. Only
  worth having where undoing is possible: a frame tree that replaced a folder
  already there cannot put that one back. The reverse direction needs nothing,
  since a sequence whose part is missing is named in `coverage.skipped`.
- `KoalaFrameWriter` numbers frames from the first one that arrives rather than
  from the step index, so a stream with nothing to say until its second step
  writes a folder numbered from zero. That is the ordinary shape of a stage
  needing two frames to make one, and it used to be refused as
  `non-contiguous frame 1: expected 0`. A gap between frames is still refused,
  since renumbering would close it -- the same reason `search_sources` refuses
  one at the source.
- A range document refuses what a range cannot be, wherever it reads one back.
  `isinstance(True, int)` is true, so `{"min_value": true, "max_value": false}`
  read as `[1.0, 0.0]`, a range running backwards; a pair the wrong way round is
  now refused however it was built. Non-finite bounds go too, and that is what
  makes the fold order-independent: `min` and `max` carry a NaN through or drop
  it depending on which part holds it, so `[2.0, nan, 1.0]` folded to `1.0` and
  `[nan, 2.0, 1.0]` to `nan`. Documents are written with `allow_nan=False`,
  since `json.dumps` writes `NaN` and `Infinity` by default and neither is JSON.
  A part that cannot be read names itself, the fold reading them sorted rather
  than in the order they were written.
- Filtering hands back memory of its own. A float32 frame cast with
  `.to(torch.float32)` is a no-op, so the window buffered a view of the source's
  own storage -- a phase folder returns a slice of an array it keeps -- and
  `IdentityKernel` then returned that same tensor to the caller; `FilterKernel`
  now states that what comes back owns its memory. The non-finite check follows
  the cast rather than preceding it, since a float64 `1e39` passes `isfinite`
  and reaches the kernel as `inf`.
- The default worker count follows this process's own affinity
  (`os.process_cpu_count`), so a run under `taskset`, a cpuset, or a scheduler's
  allocation sizes its pool to what it may use rather than to the host's cores.
  A cgroup cpu quota is still not visible. The progress bar is left undrawn when
  stderr is not a terminal, and the run says so: `tqdm` renders with a carriage
  return, so a redirected pool writes one long line of fragments at the timer's
  rate rather than the run's.
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
- `plan_devices` refuses a `workers` whose shape the device cannot read -- a
  count under CUDA, gpu ids under CPU -- rather than dropping it silently. A
  worker count a CUDA run cannot honour is a wall clock several times what the
  caller planned for, with nothing saying why. One field rather than two, so the
  pair can no longer contradict each other; `null` still means the machine's own
  answer, told apart from a value the caller wrote.
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
  and filtering everything out -- since they are fixed differently. It also
  refuses a sequence whose frames are not numbered from zero without a gap,
  naming the sequence, before any of them runs. Discovery is a name pattern and
  a sort, so a folder missing a frame opened as an ordinary shorter sequence:
  the filter joined the frames either side of the gap as neighbours, and the
  tree written back out was numbered densely, which took the gap out of the
  data entirely and left the run reporting success over numbers that were wrong.
- What more than one stage shares sits in `scripts/_common/`, mirroring
  `iivs_cardio/common/`: `hydra.py` holds the hydra boundary (`apply_schema`,
  `output_directory`, `is_multirun`), `compute.py` the machine's division
  (`ComputeConfig`, `plan_devices`, `pin_threads`, `log_insights`),
  `dataset.py` where the data is and which of it a run takes
  (`SourceConfig`, `SequenceSelectConfig`, `FrameSelectConfig`), and
  `phase.py` how a phase tree is searched. Elsewhere in
  `scripts/` a leading underscore marks a module that is not an entry point;
  the folder carries it once here, since nothing inside is one. A second
  modality would join as `hologram.py` beside the first. `scripts/_config.py`
  is gone into `hydra.py`.
- `build_phase_stages` is `build_preprocess_stages`. It hardcodes the range
  document and the frame tree and `preprocess.py` is its only caller, so the
  stage is what it is specific to; the phase it reads is already said by the
  module it lives in and by the sequences it hands back. (2) and (3) name
  theirs the same way, beside the target config each takes.
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

- `common/pipeline.py` is now the package `common/pipeline/`, holding `base.py`
  (what the file was) and `branch.py` (moved out of `data/pipeline/`). Nothing
  importing `iivs_cardio.common.pipeline` changes.
  The branch helpers moved because the policies a side branch reads are the
  pipeline's, not the phase data's: `_process.py` now takes
  `EXISTING_OUTPUT_POLICIES` and `read_policy` from `common.pipeline` and only
  `FrameTree` / `RangeDocument` from `data.pipeline`. `common/__init__` and
  `common/pipeline/__init__` re-export every public name their submodules carry;
  `data/pipeline/__init__` sheds the four policy names that left with the move.
- Neither the source search nor `FrameTree.list_sequences` descends into a
  time-lapse. Both looked for `Phase/Float/Bin` by walking to it, so every frame
  folder was listed to check whether another time-lapse was nested inside one,
  which cannot happen. `search_dirs` recognises a sequence by a single
  `dir_exists` on the parent's listing instead, and `exclude` prunes what sits
  under one: the walk of a 20-sequence x 2000-frame tree drops from 52 to 7 ms
  at the source and 39 to 7 ms at the output. The source pays the listing once
  more when `PhaseBinFolder` opens each folder for its frames, which is the one
  that is needed.
- `target.frames.if_present` takes `reuse`. A tree opening under it reads each
  written sequence's `source.json` and keeps the folder when three things still
  hold: the settings match, the source names match, and the folder holds as many
  frames as its record says. The third has no counterpart in a range part, which
  is one file and so is either there or not; a folder can be half removed, and
  reusing that would leave a short sequence reading as a whole one. Judging
  happens when the tree opens, in one process with the whole dataset in view,
  for the same reason the document judges there.
- `if_present: error` refuses when the tree opens rather than when the writer
  meets the folder. The writer meets them one at a time, so a run over 500
  sequences whose 300th was already written paid for 299 of them first.
  `FrameTree` takes `selected` for this, so a retry of one sequence is not
  refused by the ones that already succeeded.
- `source.frame_start` and `source.frame_count` beside `frame_step`, with
  `source.if_frames_short` (`take | error`) for a sequence that cannot supply
  the count. `take` takes what there is and names the short sequences in the
  log, since sequences differ in length and that is the dataset's ordinary
  shape; `error` refuses for the whole run before a frame is read, the way a
  missing frame already does, for the run whose premise is that every sequence
  gives the same number. The name says the count and the flag says the policy,
  so neither contradicts the other.
- `frame_indices(total, start=, step=, limit=)` is the one place those three
  become positions. A run reads its frames through `FilteredSequence` and names
  them again when `search_sources` lists what the source holds, and the two
  have to agree exactly: a document counts what it covers by matching one
  against the other, so a difference between them would show as every output
  being stale rather than as a wrong index. They were two independent
  expressions of the same slice, which one more setting each would have made
  three ways to disagree.
- `kaparoo-python` moves to `>=0.13.1` and `iivs-lib` to `>=0.4.0`. Between them
  they answer every request the two handoff documents made, and several helpers
  written here are deleted for the upstream equivalent (see Removed). `descend`,
  new in kaparoo 0.13.1 and reaching the phase searches through iivs-lib 0.4.0,
  replaces the rule that pruned a sequence's subtree by testing each directory's
  parent: the walk now says what it means, which is not to look inside a folder
  that holds frames.
- `read_policy` is `ensure_policy`, and hands its check to
  `kaparoo.utils.ensure_one_of`, which 0.13.0 taught to return the `Literal` it
  validated rather than `str`. The `cast` goes with it. The refusal keeps its own
  shape (`unsupported <key> '<value>': expected ...`), which every other refusal
  in this package matches, and `from None` keeps a config typo to one traceback.
- A selection is recognised as a spec file by `is_spec_file` rather than by
  `str.endswith`, so `.JSON` counts and a file named `.json` does not. That is
  the test `select` itself applies, so the log now agrees with the selection it
  describes.
- The preprocess log helpers say which half of the selection each is about:
  `_log_frame_selection` and `_log_sequence_selection`, where one of the two
  was `_log_selection` and frames are selected too. `log_filter_config` sits
  beside `describe_filter_kernel` in `_filtering.py`, `_range_file` and
  `_log_short_sequences` under the logging banner they belong to, and every
  helper takes its logger last, as the ones taking a config already did.
- The configuration log names what each line is about. A selection that is too
  long to list says `excluding 6 sequences` where it said `excluding 6`, and one
  given as a file says `the sequences listed in <path>` where the verb stood
  alone. A policy is set in under the output it belongs to, and says
  `overwriting the frames it finds` / `reusing the ranges that match this run`
  in the verb the setting itself uses. `log_target_config` asks whether each
  output is written once rather than twice, so what it writes and what it does
  with what is there arrive together.
- A stage that finds outputs with no source says `1 output with no source`
  rather than `1 output(s) have no source`, matching the line `FrameTree`
  already writes beside it.
- The progress bar counts what the parent has collected rather than what the
  workers reported finishing. Both run paths draw it and `mpire` is no longer
  asked to draw one of its own: the two counted different things, and on a run
  that computed nothing the pool's bar reached the end and closed while four
  more results were still being logged beneath it. The bar now trails a slow
  item, since `imap` returns in order, and catches up by the end.
- The selection splits by what it survives. `SequenceSelectConfig` (`include` /
  `exclude`) is the run's, since a sequence keeps its name wherever it is
  written and the same two settings pick the same sequences from every tree;
  `FrameSelectConfig` (`start` / `step` / `count` / `if_short`) is each tree's,
  since frame numbers do not survive a stage boundary and two trees a run reads
  may be at different rates -- a 20 Hz source asked for 10 Hz takes every second
  frame where a 10 Hz flow cache takes every one, and the same rate is what they
  arrive at. The preprocess config carries them as `select:` and
  `source.frames:`.
- `frame_indices` takes `count` rather than `limit`, matching the
  `select.frame_count` key that feeds it and the refusal it already wrote
  (`invalid frame count`). `FilteredSequence` and `PhaseFilteredSequence` follow.
- The range document, its parts, and a written folder's record are all written
  without `indent=2`. The whitespace was a large share of a file that carries an
  entry per frame, and nothing reads these by eye.

- `run_root` places the job's directory, where `target.root` did. That field was
  the one nothing read: it fed `hydra.run.dir` while the folder the branches
  write under came from `runtime.output_dir`, and in a sweep the two differ by
  the job number. `PreprocessTargetConfig` keeps only the two branch blocks,
  both of which are read.
- A config settles its own layout. `SequenceLayout` holds `subpath` and reserves
  `DEFAULT_SUBPATH` without a value, resolving as a method; the stage's
  `PreprocessSourceConfig` supplies Koala's and `FrameBranchConfig` supplies
  `frames`. Eight call sites had passed the default by hand from a constant in
  the phase reader, which is how `_process.py` came to know across six lines
  that this stage reads phase.
- `PhaseStageFactory` is `SequenceStageFactory`, and `TargetConfig` is
  `PreprocessTargetConfig`. The first mentions phase in two annotations and
  nowhere else -- it holds sequences, asks each branch for a hook, runs them and
  logs -- and the second describes what one stage writes, so it takes that
  stage's prefix and a second stage declares its own beside it.
- `compute.device` is a `DeviceKind` rather than a `str`. `plan_devices` asks it
  only whether it is cuda, taking the gpu ids from `workers`, so a spec naming
  an index was accepted with the index dropped and the run went to whichever
  card `workers` named. omegaconf refuses it at composition instead.
- `main` reads the filter where it reads the rest, handing the builder a
  `KernelConfig` rather than a `DictConfig`: four config nodes arrived as
  objects and the fifth as a container, and omegaconf now stops at the entry
  point.
- The builders take their arguments to one rule: the configs and the logger
  positional, and what the run discovers or is told at runtime keyword-only,
  ordered destination, dataset, this run's share, reporting. `output_root` had
  been positional in two of the three entry points and keyword-only in the
  third.
- `WorkerLogFolder` takes its name off the run it was built for, where the stage
  name was written twice with nothing checking that the two agreed.
- The five parses in `main`, and the builders below it, follow the order the run
  consumes them: read, select, filter, write.
- `check_compute_env` hands its checks back as values rather than printing them
  where they are found. Every section computed, judged, printed and accumulated
  in one breath, over thirty-six lines that each rebuilt the layout by hand:
  nine copies of one PASS/FAIL ternary, and twenty-two strings carrying their
  own indentation. `Check` and `Section` hold what a section found and `_render`
  is the single place a line's shape is decided, so a section's verdict falls
  out of `all()` and the accumulator whose two operands could be written the
  wrong way round is gone. The script gains a `main()` behind the usual guard
  with it, where it had run on import with the summary and the exit at module
  level.
- `generate_dotenv --help` says what the file is for. The parser took
  `description=__doc__` and the module has no docstring to take, the convention
  here being that it should not have one, so `--help` led with a blank line and
  the one thing a reader opens it for was the one thing it did not say.
- `ruff` reads a `dataclass`'s annotations at runtime. `OmegaConf.structured`
  builds a node from a config class's annotations, so a policy alias left behind
  `TYPE_CHECKING` is a `NameError` rather than a check.
### Removed

- `counted` and `prune_above` from `iivs_cardio.common.pipeline`, and
  `SELECTION_SPECS` from the preprocess script, for `kaparoo.utils.quantify`,
  `kaparoo.filesystem.prune_upward` and `kaparoo.filesystem.is_spec_file`.
  `quantify` also takes a `plural`, which the local one could not, and it
  reaches `writer.py` and `stage.py` without either importing a pipeline module
  for a text helper.
- `get_args(X.__value__)` at five sites, for `kaparoo.utils.literal_values(X)`,
  which raises on anything that does not resolve to a `Literal` rather than
  reporting it as empty. `get_args` on a PEP 695 alias returns an empty tuple,
  which every membership check downstream then accepts.
- `FRAME_POLICIES`, an alias for `EXISTING_OUTPUT_POLICIES`. `build_branches`
  reached for both in one function, which read as two vocabularies where there
  is one: both branches take `if_present`, and the same three values.

- The two standalone optical flow probes. Both questions they answered -- that
  cuda and cpu agree, and what the speedup is -- are answered, and the
  acquisition they ran against is being replaced. Neither was ever imported, and
  every `ruff` and `ty` finding left in the repository was one of theirs, so a
  green CI said nothing while they stayed.
- The per-key comments in the preprocess config, a second copy of the
  `Attributes:` block that had drifted from it. The `frames` ones had drifted
  furthest, describing a phase tree where the block is the general one every
  stage reads its own trees by.
### Fixed

- A CPU `push` copies the frame it retains, as the CUDA path always did. It kept
  the caller's tensor, so a caller streaming into one reusable buffer -- the
  ordinary shape of frame IO -- had the retained frame overwritten by the next
  read: `prev` and `curr` became one picture and the motion went silently to
  zero. A frame taken from a chunk was also a view pinning the whole batch, where
  the contract is one frame however long the chunk.
- A config whose `_create` answers for a device other than the one it was handed
  is refused where it is built, rather than reaching cv2 as a crash or a wrong
  answer several frames later. A CUDA algorithm is allocated on whichever device
  was current when cv2 was asked for it and nothing on the object says which, so
  the pairing is checked at the one place that knows both halves.

- A frame folder half removed no longer reads as complete. `_count_frames`
  counted directories as well as files, so a folder holding one frame and one
  directory answered two, which is exactly the number a two-frame record
  expects.
- A sequence passed over says why. `nothing to compute: every branch already
  holds this sequence` named a cause that is not there when the run writes
  nothing at all, which now says so instead.
- A log line no longer tears the progress bar it lands on. `hydra` attaches a
  `StreamHandler` to the root logger and `tqdm` draws on that same stderr with a
  carriage return, so every line overwrote the bar's own and a fresh bar
  appeared below it, once per line. Console handlers route through `tqdm.write`
  while a bar is drawn; file handlers are untouched, so the log on disk is
  unchanged.
- `scripts/env/` is visible to git again. `.gitignore`'s `env/` matches at every
  depth, and on a case-insensitive filesystem `ENV/` matches the same folder a
  second time, so anything added there read as committed locally and was absent
  on the server. The five virtualenv directory patterns are anchored to the root.

- `setup-opencv-cuda.ps1` picks the newest cuDNN by version rather than by how
  it spells. `Sort-Object FullName` compares paths as text, so `v9.9` came after
  both `v9.10` and the `v9.23` in use: a machine holding 9.9 beside a later
  release relinked 9.9, and nothing said so -- the links are made, the script
  reports success, and the compute check passes, since what is linked is a
  perfectly good cuDNN of the wrong age. The upgrade the script asks for
  afterwards is what walks into it, the installer keeping each release in a
  folder of its own. Ranking is on the version the DLL carries, with the folder
  as a fallback, and a build for another CUDA major ranks below one for this
  toolkit rather than being refused.
- A cuDNN fault points at the repair the running platform has. Both pieces of
  advice named Windows and the PowerShell script whatever the host was, so a
  Linux run was handed a file it cannot execute. The fault is one thing either
  way -- a soname is loaded once per process, and torch gets whichever cuDNN
  arrived first -- but on Linux there is nothing to repair, only a system copy
  or an `LD_LIBRARY_PATH` to find, so the lines name the two commands that show
  them.
- `target.ranges.if_present: error` refuses the parts an earlier run left rather
  than clearing them, which is what the frame tree already did with a folder.
  The two are only ever apart in one case and it is the expensive one: a run
  that finishes writes its document whatever went wrong, so parts without a
  document belong to a run killed outright, and the retry walked in, dropped
  everything it had committed, and started over without a line to say what it
  had just spent.
- The range document is written from the parts that read. One that could not be
  read took the whole document with it, since the fold refuses such a part
  rather than passing it over, so the sequences that finished cleanly were left
  as parts with nothing folding them, stating their bounds, or saying what they
  cover. A sequence whose part did not read is not among the folded ones, so
  `coverage` counts it in `skipped`, which is where a retry looks; the refusal
  is raised once the document is on disk rather than instead of it.
- A selection that came out empty is refused rather than read as no selection at
  all. An empty container is no filter to the reader, so a list built and left
  empty took the whole dataset instead of none of it, and said nothing either
  way. That is the one mistake here that runs quietly in the expensive
  direction.
- A selection naming a listing that is not there says which setting carried it.
  The reader opened `select.include` only once the walk was done, so a mistyped
  path spent the whole search before failing, and failed as a bare
  `FileNotFoundError` from inside the library.
- Staging a killed run left is cleared at the end that runs next. It was cleared
  on the way out, which is the one end a run killed outright never reaches, so
  what it staged sat in the output tree for as long as the tree did.
- Three settings that only failed once the run had started are refused before
  the walk. A `start` past the end of every sequence took none of their frames
  and nothing said so, `if_short` being guarded on `count`; and an empty layout,
  which is how a flat tree says its frames sit in the sequence folder itself,
  left `if_present` guarding nothing and `if_unsourced` offering to delete
  `.hydra`, both plates and the range parts on a nested one.
- The two paths in one log block are written in the separators the source was
  written in. `source:` printed the string the run was given and `target:` a
  `Path`, so on Windows one line came back with slashes and the other with
  backslashes.
- The short-sequence warning stops at five names, the bound its sibling already
  had. It joined every name into one line, so a dataset most of whose sequences
  fall short of the count would have put all of them in a single warning.
### Performance

- `push_chunk` and `calc_batch` fill one batch as the flows come, where they
  collected a chunk's flows in a list and `torch.stack`ed it. The list is still
  alive when `stack` allocates the result, so a chunk was held twice over: 120
  frames of 900x900 peaked at 1471 MiB for a 735 MiB result, and `push_chunk` is
  the call whose contract tells a caller to bound the chunk to bound the memory.
  Peak is 736 MiB, and 685 ms is 637, the intermediate `contiguous` being spared
  as well. `Backend.push` and `Backend.calc` take an `out` for it, and
  `Backend.retained` says whether a chunk's first frame is spent retaining, which
  is what sizes the batch exactly. Allocating for the frames and returning a
  slice would keep the whole storage alive, and `torch.save` writes out the
  storage, not the view.

- A sweep searches the dataset once rather than once per filter. Every job of a
  `--multirun` runs in one process and differs only in the filter, yet each
  rebuilt the same answer: on 440 sequences and 448,800 frames the search, the
  frame-number check and the contents together come to about 16 s, and 16 chunks
  by 11 filters would have paid it 176 times. `search_sources` holds its newest
  answer, keyed on the whole of `SourceConfig` and on the working directory,
  which a relative `root` or `include` needs since `hydra.job.chdir` can move it.
  Only the newest is held, so a caller asking for anything else pays what it paid
  before.
- `validate(level="names")` costs about 0.5 us a file rather than 8.8, from
  iivs-lib 0.4.0, measured here over 40,000 frames.
