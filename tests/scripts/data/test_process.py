from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from iivs.dhm.data.koala import PHASE_FLOAT_BIN
from iivs.dhm.data.phase import (
    PhaseBinFolder,
    PhaseUnit,
    read_phase_bin_header,
    save_phase_bin,
)

from iivs_cardio.data.pipeline import FrameTree, PhaseStageFactory, RangeDocument
from scripts._compute import ComputeConfig, IncompleteRunError, run_all
from scripts.data._filtering import parse_filter_config
from scripts.data._process import (
    SELECTION_LIMIT,
    SourceConfig,
    TargetConfig,
    build_branches,
    build_phase_stages,
    build_sequences,
    log_configs,
    search_sources,
)
from scripts.data.preprocess import CONFIG_NAME, CONFIG_PATH
from tests.scripts.conftest import (
    FRAMES,
    HEIGHT_SCALE,
    PIXEL_SIZE,
    SEQUENCES,
)

STAGE = "preprocess"


def _scan(
    phase_tree: Path,
    dest: Path,
    workers: int,
    *,
    save_frames: bool = True,
    save_ranges: bool = False,
) -> None:
    source = SourceConfig(root=str(phase_tree))
    compute = ComputeConfig(device="cpu", workers=workers, show_progress=False)
    config = TargetConfig(
        root=str(dest), save_frames=save_frames, save_ranges=save_ranges
    )

    run_all(build_phase_stages(source, config, name=STAGE, output_root=dest), compute)


def _document(dest: Path) -> dict:
    return json.loads((dest / "value_range.json").read_text(encoding="utf-8"))


def _written(dest: Path) -> dict[str, list[float]]:
    """Every frame written under `dest`, keyed by the sequence it belongs to."""
    out: dict[str, list[float]] = {}
    for folder in sorted(dest.rglob(PHASE_FLOAT_BIN)):
        read = PhaseBinFolder(folder)
        key = folder.relative_to(dest).as_posix()
        out[key] = [float(np.asarray(read[index]).sum()) for index in range(len(read))]

    return out


def test_sources_are_found_under_the_root(phase_tree):
    # The one that fails silently: a search that finds nothing leaves every other
    # check green, since there is then no sequence to get anything wrong with.
    sources = search_sources(SourceConfig(root=str(phase_tree)))

    assert len(sources) == SEQUENCES


def test_a_root_holding_nothing_and_an_empty_selection_are_told_apart(
    phase_tree, tmp_path
):
    # The two are fixed differently, so they must not arrive as one message. A
    # root that does not exist at all fails earlier still, inside the search.
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(ValueError, match=r"no time-lapse holds"):
        search_sources(SourceConfig(root=str(empty)))

    with pytest.raises(ValueError, match=r"include/exclude left none"):
        search_sources(
            SourceConfig(
                root=str(phase_tree),
                exclude=[f"TL_{s:02d}" for s in range(SEQUENCES)],
            )
        )


def test_a_sequence_knows_what_it_is_called_in_its_dataset(phase_tree):
    # Derived from the folder it was opened over, so a side branch reading the
    # name cannot land somewhere the frames did not come from.
    sequences = build_sequences(
        SourceConfig(root=str(phase_tree)), parse_filter_config(None)
    )

    assert [sequence.name for sequence in sequences] == [
        f"TL_{index:02d}" for index in range(SEQUENCES)
    ]


def test_a_target_asking_for_nothing_is_refused(phase_tree, tmp_path):
    # Both branches off is a config that configures nothing, which is a mistake
    # rather than a way to say "just read" -- that is `target_config=None`.
    source = SourceConfig(root=str(phase_tree))
    target = TargetConfig(root=str(tmp_path), save_frames=False, save_ranges=False)

    with pytest.raises(ValueError, match=r"nothing to do"):
        build_phase_stages(source, target, name=STAGE, output_root=tmp_path)


def test_a_target_asking_for_nothing_is_refused_before_the_source_is_read(tmp_path):
    # Reading the source costs a header per frame -- about 4.3s for 121x1200 --
    # and a sweep pays it per job, since it does not stop at the first refusal.
    # Nothing about this verdict needs the frames, so a root that cannot be read
    # at all still has to come back with the configuration's own complaint.
    source = SourceConfig(root=str(tmp_path / "not-here"))
    target = TargetConfig(root=str(tmp_path), save_frames=False, save_ranges=False)

    with pytest.raises(ValueError, match=r"nothing to do"):
        build_phase_stages(source, target, name=STAGE, output_root=tmp_path)


def test_a_factory_offers_one_stage_per_sequence(phase_tree, tmp_path):
    source = SourceConfig(root=str(phase_tree))
    stages = build_phase_stages(source, name=STAGE, output_root=tmp_path)

    assert len(stages) == SEQUENCES


def test_a_run_without_a_target_writes_nothing(phase_tree, tmp_path):
    # `target_config=None` is how a run that only reads -- a cached source, or a
    # timing pass -- says it wants no side branch. The root is still named, so
    # what keeps it empty is the missing target rather than a missing place.
    dest = tmp_path / "out"
    source = SourceConfig(root=str(phase_tree))
    compute = ComputeConfig(device="cpu", workers=0, show_progress=False)

    run_all(build_phase_stages(source, name=STAGE, output_root=dest), compute)

    assert not dest.exists()


def test_the_pool_reports_what_the_lone_path_does(phase_tree, tmp_path):
    # The range branch gathers through files a worker leaves behind, so the two
    # paths agree only if what crosses the process boundary is complete.
    _scan(phase_tree, tmp_path / "lone", 0, save_frames=False, save_ranges=True)
    _scan(phase_tree, tmp_path / "pooled", 2, save_frames=False, save_ranges=True)

    lone = _document(tmp_path / "lone")
    assert [s["source"] for s in lone["dataset"]["sequences"]] == [
        f"TL_{index:02d}" for index in range(SEQUENCES)
    ]
    assert all(len(s["frames"]) == FRAMES for s in lone["dataset"]["sequences"])
    assert _document(tmp_path / "pooled")["dataset"] == lone["dataset"]


def test_the_parts_a_run_gathered_stay_beside_the_document(phase_tree, tmp_path):
    dest = tmp_path / "out"

    _scan(phase_tree, dest, 2, save_frames=False, save_ranges=True)

    assert sorted(p.name for p in (dest / "value_range.parts").iterdir()) == [
        f"TL_{index:02d}.json" for index in range(SEQUENCES)
    ]


def test_the_pool_writes_what_the_lone_path_does(phase_tree, tmp_path):
    # Covers the whole worker path in one assertion. `mpire` hands its workers a
    # positional signature that no type checker sees, so the pool can go wrong
    # while every other check stays green.
    _scan(phase_tree, tmp_path / "lone", 0)
    _scan(phase_tree, tmp_path / "pooled", 2)

    lone = _written(tmp_path / "lone")
    assert len(lone) == SEQUENCES
    assert all(len(frames) == FRAMES for frames in lone.values())
    assert _written(tmp_path / "pooled") == lone


def test_the_written_tree_mirrors_the_source(phase_tree, tmp_path):
    # A frame tree crosses to the workers as a recipe, so the folders have to
    # land under its root -- with nothing coming back to say they did.
    dest = tmp_path / "out"

    _scan(phase_tree, dest, 2)

    for sequence in range(SEQUENCES):
        written = PhaseBinFolder(dest / f"TL_{sequence:02d}" / PHASE_FLOAT_BIN)
        assert len(written) == FRAMES


def test_the_written_frames_are_always_in_radians(phase_tree, tmp_path):
    # A metric reads optical path difference out of phase, so the cache a run
    # leaves carries one unit -- and the header still holds `height_scale` for
    # whoever wants metres back.
    dest = tmp_path / "out"

    _scan(phase_tree, dest, 0)

    header = read_phase_bin_header(dest / "TL_00" / PHASE_FLOAT_BIN / "00000_phase.bin")
    assert header.unit is PhaseUnit.RADIANS


def test_the_job_names_the_stage_every_line_is_filed_under(
    phase_tree, tmp_path, caplog
):
    # Nothing here is preprocessing by nature -- the same filtered-phase run is
    # postprocessing behind a hologram reconstruction -- so the name is the job's
    # to give, and every line of the run has to follow it: the configuration, the
    # driver's own summary, the per-sequence block, and the side branches.
    stage = "reconstruct"
    source = SourceConfig(root=str(phase_tree))
    target = TargetConfig(root=str(tmp_path), save_frames=True, save_ranges=True)
    compute = ComputeConfig(device="cpu", workers=0, show_progress=False)

    with caplog.at_level(logging.INFO):
        stages = build_phase_stages(source, target, name=stage, output_root=tmp_path)
        run_all(stages, compute)

    assert stages.name == stage
    assert {record.name for record in caplog.records} == {stage}

    messages = [record.getMessage().strip() for record in caplog.records]
    assert any(message.startswith("source: ") for message in messages)
    assert messages.count(f"wrote {FRAMES} frames") == SEQUENCES
    assert sum(message.startswith("measured [") for message in messages) == SEQUENCES
    assert any(
        message.startswith(f"wrote value_range.json from {SEQUENCES} sequences")
        for message in messages
    )


def test_a_sequence_leads_its_own_block(phase_tree, tmp_path, caplog):
    # The name is the head and its lines hang under it, so a reader skimming the
    # left margin sees one entry per sequence rather than one flat list. Pinned
    # because the nesting is a default a call site can lose without failing.
    source = SourceConfig(root=str(phase_tree))
    target = TargetConfig(root=str(tmp_path), save_frames=False, save_ranges=True)
    compute = ComputeConfig(device="cpu", workers=0, show_progress=False)

    with caplog.at_level(logging.INFO):
        stages = build_phase_stages(source, target, name=STAGE, output_root=tmp_path)
        run_all(stages, compute)

    logged = [record.getMessage() for record in caplog.records]
    head = logged.index("TL_00")

    assert logged[head + 1].startswith("  filtering ")
    assert logged[head + 2].startswith("  measured ")
    assert logged[head + 3].startswith("  done in ")


def test_a_sequence_holding_a_non_finite_frame_costs_only_that_sequence(
    phase_tree, tmp_path
):
    # The formats this project reads store a NaN happily, so a bad acquisition
    # arrives looking like any other. Refusing it must not cost the run: at a
    # dataset's scale the sequences already finished are hours of work.
    save_phase_bin(
        phase_tree / "TL_01" / PHASE_FLOAT_BIN / "00002_phase.bin",
        np.full((4, 5), np.nan, dtype=np.float32),
        pixel_size=PIXEL_SIZE,
        height_scale=HEIGHT_SCALE,
        overwrite=True,
        on_nonfinite="ignore",  # the whole point: nothing upstream objects
    )
    dest = tmp_path / "out"

    with pytest.raises(IncompleteRunError, match=r"1 of 3 failed") as failure:
        _scan(phase_tree, dest, 0, save_frames=True, save_ranges=True)

    ((name, reason),) = failure.value.failed.items()
    assert name == "TL_01"
    assert "non-finite value in" in reason

    # The other two are whole -- frames committed, ranges folded -- and the one
    # that failed left no half-written folder behind.
    subpath = Path(PHASE_FLOAT_BIN).as_posix()
    assert sorted(_written(dest)) == [f"TL_{s:02d}/{subpath}" for s in (0, 2)]
    assert [s["source"] for s in _document(dest)["dataset"]["sequences"]] == [
        "TL_00",
        "TL_02",
    ]

    # And the document says so: bounds folded over two of three sequences are
    # not the dataset's, and the consumer that sets a policy from them is the
    # one who would never find out.
    assert _document(dest)["coverage"] == {
        "covered": 2,
        "total": 3,
        "skipped": ["TL_01"],
    }


class _Hook:
    """A side branch's hook that may have nothing to say when it is over."""

    def __init__(self, line: str | None) -> None:
        self._line = line

    def __call__(self, step) -> None:
        return None

    def report(self) -> str | None:
        return self._line


class _Branch:
    """A side branch that reports the same thing its hooks do."""

    def __init__(self, line: str | None) -> None:
        self._line = line

    def get_hook(self, source) -> _Hook:
        return _Hook(self._line)

    def report(self) -> str | None:
        return self._line


def test_a_branch_with_nothing_to_say_adds_no_line(phase_tree, caplog):
    # `report()` answering `None` is the contract for a branch that committed
    # nothing, and the block has to leave it out rather than log an empty line.
    # Both scopes are covered here: a hook reports per sequence, a branch once.
    def lines(said: str | None) -> list[str]:
        stages = PhaseStageFactory(
            build_sequences(
                SourceConfig(root=str(phase_tree)), parse_filter_config(None)
            ),
            _Branch(said),
            name=STAGE,
        )
        caplog.clear()
        with caplog.at_level(logging.INFO):
            run_all(stages, ComputeConfig(device="cpu", workers=0, show_progress=False))

        return [record.getMessage() for record in caplog.records]

    spoken = lines("spoke")
    assert sum(message.strip() == "spoke" for message in spoken) == SEQUENCES + 1

    quiet = lines(None)
    assert all(message.strip() for message in quiet)
    assert not any("None" in message for message in quiet)


# ------------------------------ the side branches ------------------------- #


def _branches(tmp_path, *, sequence_names=("TL_00",), subpath=None, step=1, **target):
    source = SourceConfig(root="/dataset", subpath=subpath, frame_step=step)
    config = TargetConfig(root=str(tmp_path), **target)

    return build_branches(
        source,
        config,
        parse_filter_config(None),
        tmp_path,
        list(sequence_names),
    )


@pytest.mark.parametrize(
    ("target", "kinds"),
    (
        ({"save_ranges": True}, [RangeDocument]),
        ({"save_frames": True, "save_ranges": False}, [FrameTree]),
        ({"save_frames": True, "save_ranges": True}, [FrameTree, RangeDocument]),
    ),
)
def test_a_target_asks_for_the_branches_it_names(tmp_path, target, kinds):
    branches = _branches(tmp_path, **target)

    assert [type(branch) for branch in branches] == kinds


def test_the_settings_carry_what_would_change_the_numbers(tmp_path):
    # What a later run compares to decide whether this document still describes
    # it, so it has to hold everything that changes the frames a sequence reads.
    (document,) = _branches(tmp_path, subpath="Phase/Float/Other", step=3)

    assert document.settings["source"] == {
        "subpath": "Phase/Float/Other",
        "frame_step": 3,
    }
    assert document.settings["filter"]["kind"] == "identity"


def test_an_unset_subpath_is_recorded_as_the_one_that_was_read(tmp_path):
    # `None` means the Koala default rather than "no subpath", and a document
    # saying `null` could not be compared against a run that named it outright.
    (document,) = _branches(tmp_path, subpath=None)

    assert document.settings["source"]["subpath"] == PHASE_FLOAT_BIN


def test_the_selection_stays_out_of_the_settings(tmp_path):
    # `include` / `exclude` change which sequences a run covers, not what any
    # sequence's numbers mean -- and `coverage` already reports that. Recording
    # them here would refuse reuse to a run that narrowed its selection.
    source = SourceConfig(root="/dataset", include=["TL_00"], exclude=["TL_09"])
    config = TargetConfig(root=str(tmp_path), save_ranges=True)

    (document,) = build_branches(
        source, config, parse_filter_config(None), tmp_path, ["TL_00"]
    )

    assert set(document.settings["source"]) == {"subpath", "frame_step"}


def test_the_roster_reaches_the_document_that_reports_on_it(tmp_path):
    (document,) = _branches(tmp_path, sequence_names=("TL_00", "TL_01", "TL_02"))

    assert document.sequence_names == ("TL_00", "TL_01", "TL_02")


def test_branches_are_refused_before_any_of_them_is_built(tmp_path):
    with pytest.raises(ValueError, match=r"nothing to do"):
        _branches(tmp_path, save_frames=False, save_ranges=False)


# ---------------------------- the configuration log ----------------------- #


def _logged(caplog, source, target=None, kernel=None):
    with caplog.at_level(logging.INFO):
        log_configs(source, target, kernel or parse_filter_config(None), name=STAGE)

    return [record.getMessage() for record in caplog.records]


def test_each_block_is_tagged_by_what_it_configures(caplog):
    # A tag apiece rather than a verb, so a reader looking for one of the three
    # finds it by name -- and the run's own lines below are never mistaken for
    # configuration.
    logged = _logged(caplog, SourceConfig(root="/dataset"), TargetConfig(root="/out"))
    heads = [line for line in logged if not line.startswith("  ")]

    assert heads == [
        "source: /dataset",
        "filter: identity kernel",
        "target: /out",
    ]


def test_the_subpath_is_shown_as_the_shape_it_is(caplog):
    # Shown rather than described: prose about which of the two nests inside the
    # other kept being read backwards, where the path template cannot be.
    logged = _logged(caplog, SourceConfig(root="/dataset", subpath="Phase/Other"))

    assert "  reading <sequence>/Phase/Other" in logged


@pytest.mark.parametrize(
    ("step", "said"),
    (
        (1, None),
        (2, "  reading frames 0, 2, 4, ..."),
        (3, "  reading frames 0, 3, 6, ..."),
        (21, "  reading frames 0, 21, 42, ..."),
    ),
)
def test_the_stride_is_said_only_when_it_drops_frames(caplog, step, said):
    # The indices carry the stride and settle where the count starts, which the
    # stride on its own leaves open.
    logged = _logged(caplog, SourceConfig(root="/dataset", frame_step=step))
    stride = [line for line in logged if line.startswith("  reading frames")]

    assert stride == ([] if said is None else [said])


def test_a_selection_naming_a_file_points_at_the_file(caplog):
    # A path is a promise about a file, where a name is the answer itself, so
    # the two cannot read the same.
    logged = _logged(caplog, SourceConfig(root="/d", include="/cfg/keep.json"))

    assert "  including as listed in /cfg/keep.json" in logged


def test_a_selection_naming_one_thing_says_it_outright(caplog):
    logged = _logged(caplog, SourceConfig(root="/d", exclude="TL_09"))

    assert "  excluding TL_09" in logged


def test_a_short_selection_is_listed_one_to_a_line(caplog):
    logged = _logged(caplog, SourceConfig(root="/d", exclude=["TL_07", "TL_09"]))
    head = logged.index("  excluding:")

    assert logged[head + 1 : head + 3] == ["    TL_07", "    TL_09"]


def test_a_long_selection_points_at_the_config_instead(caplog):
    # Listing it would bury the lines around it, and the job's own `.hydra` holds
    # it exactly: `config.yaml` always, as part of the composed config, and
    # `overrides.yaml` whenever the command line is what set it -- which is the
    # usual way, since both selections default to null. The path stays relative
    # because the log file naming it already sits in that directory, which is
    # what keeps a sweep's jobs pointing at their own copies.
    names = [f"TL_{index:02d}" for index in range(SELECTION_LIMIT + 1)]

    logged = _logged(caplog, SourceConfig(root="/d", exclude=names))

    assert f"  excluding {len(names)}, listed in .hydra/{{config,overrides}}.yaml" in (
        logged
    )
    assert "    TL_00" not in logged


@pytest.mark.parametrize("selection", (None, []))
def test_a_selection_that_narrows_nothing_is_left_out(caplog, selection):
    # An absent line reads as "all of it", which is what an unset selection is.
    logged = _logged(caplog, SourceConfig(root="/dataset", include=selection))

    assert not [line for line in logged if "including" in line]


def test_the_filter_fits_on_one_line(caplog):
    # `kind` leads the line, so repeating it among the settings said it twice.
    with initialize_config_dir(config_dir=CONFIG_PATH, version_base=None):
        composed = compose(
            config_name=CONFIG_NAME,
            overrides=["data/transforms/filtering@filter=median_cuboid_2x2x1"],
        )

    logged = _logged(
        caplog,
        SourceConfig(root="/dataset"),
        kernel=parse_filter_config(composed.filter),
    )

    assert "filter: median kernel (radius=[2, 2, 1], shape=cuboid)" in logged
    assert not [line for line in logged if "kind=" in line]


def test_a_kernel_with_nothing_to_set_says_only_what_it_is(caplog):
    logged = _logged(caplog, SourceConfig(root="/dataset"))

    assert "filter: identity kernel" in logged


@pytest.mark.parametrize(
    ("target", "written"),
    (
        ({}, ["  writing the value ranges to value_range.json"]),
        (
            {"save_frames": True},
            [
                "  writing the filtered frames to <sequence>/Phase/Float/Bin",
                "  writing the value ranges to value_range.json",
            ],
        ),
        (
            {"save_frames": True, "save_ranges": False},
            ["  writing the filtered frames to <sequence>/Phase/Float/Bin"],
        ),
        ({"save_ranges": False}, ["  writing nothing"]),
        (
            {"range_file": "phase_range"},
            ["  writing the value ranges to phase_range.json"],
        ),
    ),
)
def test_a_target_says_what_it_writes_and_where(caplog, target, written):
    # A line apiece, since one naming both would read as though the frames went
    # into the document too. The extension belongs to the document, so the log
    # names the file that will be there rather than the stem a run configured.
    logged = _logged(
        caplog, SourceConfig(root="/d"), TargetConfig(root="/out", **target)
    )

    assert [line for line in logged if line.startswith("  writing")] == written


def test_the_written_frames_take_the_shape_the_source_was_read_in(caplog):
    # The mirroring shown rather than claimed: the two lines carry the same
    # template, close enough to read against each other, and a target that
    # stopped following the source's subpath would show up as a mismatch.
    logged = _logged(
        caplog,
        SourceConfig(root="/dataset", subpath="Phase/Other"),
        TargetConfig(root="/out", save_frames=True),
    )

    assert "  reading <sequence>/Phase/Other" in logged
    assert "  writing the filtered frames to <sequence>/Phase/Other" in logged


def test_a_run_allowed_to_replace_says_so(caplog):
    # Only the permissive state is said, refusing being the default. Worth
    # revisiting when a third answer lands -- reuse whatever is still valid --
    # since silence can separate two states but not three.
    allowed = _logged(
        caplog, SourceConfig(root="/d"), TargetConfig(root="/o", overwrite=True)
    )
    assert "  replacing what is already there" in allowed

    caplog.clear()
    refused = _logged(caplog, SourceConfig(root="/d"), TargetConfig(root="/o"))
    assert not [line for line in refused if "replace" in line]


def test_a_run_without_a_target_says_nothing_about_writing(caplog):
    logged = _logged(caplog, SourceConfig(root="/dataset"))

    assert not any(line.startswith("target") for line in logged)
