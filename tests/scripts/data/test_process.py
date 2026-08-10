from __future__ import annotations

import contextlib
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
    search_phase_bin_folders,
)
from omegaconf import OmegaConf

from iivs_cardio.common.device import Device
from iivs_cardio.data.pipeline import FrameTree, PhaseStageFactory, RangeDocument
from iivs_cardio.data.transforms.filtering.kernel import MedianConfig
from iivs_cardio.data.writer import RECORD_FILE
from scripts._compute import ComputeConfig, IncompleteRunError, run_all
from scripts._phase import build_sequences, search_sources
from scripts._trees import SELECTION_LIMIT, SourceConfig
from scripts.data._filtering import parse_filter_config
from scripts.data._process import (
    TargetConfig,
    build_branches,
    build_phase_stages,
    log_configs,
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
    subpath: str | None = None,
    overwrite: bool = False,
) -> None:
    source = SourceConfig(root=str(phase_tree))
    compute = ComputeConfig(device="cpu", workers=workers, show_progress=False)
    policy = "overwrite" if overwrite else "error"
    config = TargetConfig(
        root=str(dest),
        subpath=subpath,
        save_frames=save_frames,
        save_ranges=save_ranges,
        if_frames_exist=policy,
        if_ranges_exist=policy,
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


def _rewrite_unit(phase_tree: Path, name: str, unit: PhaseUnit) -> None:
    """Rewrite one sequence's frames under `unit`, in a single statement."""
    for frame in range(FRAMES):
        save_phase_bin(
            phase_tree / name / PHASE_FLOAT_BIN / f"{frame:05d}_phase.bin",
            np.zeros((4, 5), dtype=np.float32),
            pixel_size=PIXEL_SIZE,
            height_scale=HEIGHT_SCALE,
            unit=unit,
            overwrite=True,
        )


def test_sources_are_found_under_the_root(phase_tree):
    # The one that fails silently: a search that finds nothing leaves every other
    # check green, since there is then no sequence to get anything wrong with.
    sources, contents = search_sources(SourceConfig(root=str(phase_tree)))

    assert len(sources) == SEQUENCES
    assert sorted(contents) == [f"TL_{index:02d}" for index in range(SEQUENCES)]
    assert all(len(frames) == FRAMES for frames in contents.values())


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


def test_a_sequence_missing_a_frame_is_refused_by_name(phase_tree, tmp_path):
    # The one that would be believed: discovery is a name pattern and a sort, so
    # a gap opens as an ordinary shorter sequence. The filter would then treat
    # the frames either side of it as neighbours, and the tree written back out
    # is numbered from zero without a gap, leaving the run reporting success
    # over numbers that are wrong by the width of the gap. Refused for the whole
    # dataset rather than per item, and the name says which sequence to fix.
    (phase_tree / "TL_01" / PHASE_FLOAT_BIN / "00002_phase.bin").unlink()
    dest = tmp_path / "out"

    with pytest.raises(ValueError, match=r"TL_01: non-contiguous"):
        search_sources(SourceConfig(root=str(phase_tree)))

    with pytest.raises(ValueError, match=r"TL_01: non-contiguous"):
        _scan(phase_tree, dest, 0)

    assert not dest.exists()


def test_a_sequence_that_cannot_be_read_in_radians_is_named(phase_tree):
    # A header may say `UNKNOWN`, which `save_phase_bin` only warns about, and
    # setting the unit runs the conversion check again. One such acquisition
    # took the whole run down saying only that it could not convert: with 121
    # sequences and no name, there was nothing to `exclude` on.
    with pytest.warns(UserWarning, match="unit=UNKNOWN"):
        _rewrite_unit(phase_tree, "TL_01", PhaseUnit.UNKNOWN)

    with pytest.raises(ValueError, match=r"TL_01: cannot convert phase"):
        search_sources(SourceConfig(root=str(phase_tree)))

    sources, _ = search_sources(SourceConfig(root=str(phase_tree), exclude=["TL_01"]))
    assert len(sources) == SEQUENCES - 1


def test_a_sequence_missing_a_frame_can_be_excluded_rather_than_fixed(phase_tree):
    # The check sits after the selection, so one bad acquisition does not put a
    # whole dataset out of reach, and `exclude` is the way round it while the
    # frames are recovered. Ahead of the selection it would close that off, and
    # every other test would stay green.
    (phase_tree / "TL_01" / PHASE_FLOAT_BIN / "00002_phase.bin").unlink()

    sources, contents = search_sources(
        SourceConfig(root=str(phase_tree), exclude=["TL_01"])
    )

    assert len(sources) == SEQUENCES - 1
    assert len(contents) == SEQUENCES  # the excluded one is still in the dataset


def test_the_root_is_walked_once_however_many_runs_ask(phase_tree, monkeypatch):
    # A sweep runs every job in one process and only the filter differs between
    # them, so the walk that opens every sequence would otherwise be paid once
    # per filter. Counted with a spy rather than compared: a cache that never
    # hit would hand back the same answer and pass on the result alone.
    walks = []

    def spy(*args, **kwargs):
        walks.append(args)
        return search_phase_bin_folders(*args, **kwargs)

    monkeypatch.setattr("scripts._phase.search_phase_bin_folders", spy)

    first, first_contents = search_sources(SourceConfig(root=str(phase_tree)))
    second, second_contents = search_sources(SourceConfig(root=str(phase_tree)))

    assert len(walks) == 1
    assert [source.root for source in second] == [source.root for source in first]
    assert second_contents == first_contents


def test_a_source_asking_for_something_else_is_walked_again(phase_tree, monkeypatch):
    # `include` is part of what makes one search differ from another, and only
    # the newest is held. Reusing across the change would hand a run the very
    # sequences it asked to leave out.
    walks = []

    def spy(*args, **kwargs):
        walks.append(args)
        return search_phase_bin_folders(*args, **kwargs)

    monkeypatch.setattr("scripts._phase.search_phase_bin_folders", spy)

    search_sources(SourceConfig(root=str(phase_tree)))
    taken, contents = search_sources(
        SourceConfig(root=str(phase_tree), include=["TL_00"])
    )

    assert len(walks) == 2
    assert [source.root.parents[2].name for source in taken] == ["TL_00"]
    assert len(contents) == SEQUENCES  # the ones left out are still the dataset


def test_what_a_run_does_with_its_own_answer_stays_its_own(phase_tree):
    # The held answer is handed out as a fresh list and dict, so a job that
    # reorders or adds to what it was given cannot leave the next one short.
    sources, contents = search_sources(SourceConfig(root=str(phase_tree)))
    sources.clear()
    contents.clear()

    again, again_contents = search_sources(SourceConfig(root=str(phase_tree)))

    assert len(again) == SEQUENCES
    assert sorted(again_contents) == [f"TL_{index:02d}" for index in range(SEQUENCES)]


def _short(phase_tree: Path, name: str, keep: int) -> None:
    """Leave `name` holding `keep` frames, so a count can fall short of it."""
    for frame in range(keep, FRAMES):
        (phase_tree / name / PHASE_FLOAT_BIN / f"{frame:05d}_phase.bin").unlink()


def test_a_count_takes_that_many_from_where_the_start_says(phase_tree):
    # The three settings become positions in one place, and the contents is
    # what a document counts itself against, so this is where they agree.
    source = SourceConfig(
        root=str(phase_tree), frame_start=1, frame_step=2, frame_count=2
    )

    sequences, contents = search_sources(source)

    assert contents["TL_00"] == ("00001_phase.bin", "00003_phase.bin")
    assert all(len(frames) == 2 for frames in contents.values())
    assert len(sequences) == SEQUENCES


def test_a_sequence_that_cannot_supply_the_count_is_taken_as_it_is(phase_tree, caplog):
    # Sequences differ in length, so falling short is the dataset's ordinary
    # shape rather than a fault. Named, since a reader comparing them has to
    # know which ones are not on the same footing.
    _short(phase_tree, "TL_01", FRAMES - 1)
    source = SourceConfig(root=str(phase_tree), frame_count=FRAMES)

    with caplog.at_level(logging.INFO):
        stages = build_phase_stages(source, name=STAGE, output_root="/out")

    assert len(stages) == SEQUENCES
    said = " ".join(record.getMessage() for record in caplog.records)
    assert f"1 sequence gave fewer than {FRAMES}: TL_01 ({FRAMES - 1})" in said


def test_a_sequence_that_cannot_supply_the_count_can_be_refused_instead(phase_tree):
    # `error` is for the run whose premise is that every sequence gives the
    # same number, and it refuses for the dataset rather than per item: the
    # premise is broken before a single frame has been read.
    _short(phase_tree, "TL_01", 2)
    source = SourceConfig(
        root=str(phase_tree), frame_count=FRAMES, if_frames_short="error"
    )

    with pytest.raises(ValueError, match=r"TL_01: 2 frames after the stride"):
        search_sources(source)


def test_a_count_nobody_falls_short_of_says_nothing(phase_tree, caplog):
    source = SourceConfig(root=str(phase_tree), frame_count=FRAMES)

    with caplog.at_level(logging.INFO):
        build_phase_stages(source, name=STAGE, output_root="/out")

    assert not [line for line in caplog.messages if "gave fewer" in line]


def test_a_narrowed_run_says_how_much_of_the_dataset_it_took(phase_tree, tmp_path):
    # The retry that produced a document reading as complete: `include` takes one
    # of three, and the file said "covered 1, skipped none" over a `source` that
    # names the whole root. The search already counts what the root held, and
    # that count is what the document was missing.
    dest = tmp_path / "out"
    source = SourceConfig(root=str(phase_tree), include=["TL_01"])
    target = TargetConfig(root=str(dest), save_frames=False, save_ranges=True)
    compute = ComputeConfig(device="cpu", workers=0, show_progress=False)

    run_all(build_phase_stages(source, target, name=STAGE, output_root=dest), compute)

    assert _document(dest)["coverage"] == {
        "found": SEQUENCES,
        "selected": 1,
        "covered": 1,
        "reused": 0,
        "skipped": [],
        "unselected": ["TL_00", "TL_02"],
    }


def test_a_stride_leaves_the_two_outputs_naming_frames_differently(
    phase_tree, tmp_path, caplog
):
    # The ranges are filed under the source and the cache numbers its own from
    # zero, so at a stride the same name means different frames in the two. It
    # is not a missing key but a wrong one: a reader joining by name gets a real
    # entry holding another frame's bounds. Position is the key, and the run
    # says so out loud rather than leaving it to whoever reads the pair.
    dest = tmp_path / "out"
    source = SourceConfig(root=str(phase_tree), frame_step=2)
    target = TargetConfig(root=str(dest), save_frames=True, save_ranges=True)
    compute = ComputeConfig(device="cpu", workers=0, show_progress=False)

    with caplog.at_level(logging.INFO):
        stages = build_phase_stages(source, target, name=STAGE, output_root=dest)
        run_all(stages, compute)

    (sequence,) = [
        s for s in _document(dest)["dataset"]["sequences"] if s["source"] == "TL_00"
    ]
    written = PhaseBinFolder(dest / "TL_00" / PHASE_FLOAT_BIN)

    assert [frame["source"] for frame in sequence["frames"]] == [
        "00000_phase.bin",
        "00002_phase.bin",
    ]
    assert [path.name for path in written.files] == [
        "00000_phase.bin",
        "00001_phase.bin",
    ]

    # Joined by position, every pair agrees; by name, the second would not.
    for index, frame in enumerate(sequence["frames"]):
        held = np.asarray(written[index])
        assert (frame["min_value"], frame["max_value"]) == (held.min(), held.max())

    warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("renumbers the frames" in message for message in warned)


def test_a_sequence_knows_what_it_is_called_in_its_dataset(phase_tree):
    # Derived from the folder it was opened over, so a side branch reading the
    # name cannot land somewhere the frames did not come from.
    sequences, _ = build_sequences(
        SourceConfig(root=str(phase_tree)), parse_filter_config(None)
    )

    assert [sequence.name for sequence in sequences] == [
        f"TL_{index:02d}" for index in range(SEQUENCES)
    ]


def test_a_target_asking_for_nothing_is_refused(phase_tree, tmp_path):
    # Both branches off is a config that configures nothing, which is a mistake
    # rather than a way to say "just read": that is `target_config=None`.
    source = SourceConfig(root=str(phase_tree))
    target = TargetConfig(root=str(tmp_path), save_frames=False, save_ranges=False)

    with pytest.raises(ValueError, match=r"nothing to do"):
        build_phase_stages(source, target, name=STAGE, output_root=tmp_path)


def test_a_target_asking_for_nothing_is_refused_before_the_source_is_read(tmp_path):
    # Reading the source costs a header per frame, about 4.3s for 121x1200,
    # and a sweep pays it per job, since it does not stop at the first refusal.
    # Nothing about this verdict needs the frames, so a root that cannot be read
    # at all still has to come back with the configuration's own complaint.
    source = SourceConfig(root=str(tmp_path / "not-here"))
    target = TargetConfig(root=str(tmp_path), save_frames=False, save_ranges=False)

    with pytest.raises(ValueError, match=r"nothing to do"):
        build_phase_stages(source, target, name=STAGE, output_root=tmp_path)


# ------------------------------ the destination --------------------------- #

FILTERED = "Phase/Float/FilteredBin"


@pytest.mark.parametrize("config", (SourceConfig, TargetConfig))
@pytest.mark.parametrize("subpath", ("/elsewhere", "../raw", "Phase/../../raw"))
def test_a_subpath_that_reaches_outside_a_sequence_is_refused(config, subpath):
    # The destination check compares the two subpaths as they stand, so a `..`
    # walks straight past it and `Path(root, name, subpath)` lands wherever it
    # points: back inside the source, on a run the check just cleared. Refused
    # where a subpath is read instead, which covers whichever end named it.
    # `is_absolute()` is not the test: `/elsewhere` is False on Windows, where
    # it still resets the path to the drive root, and the anchor is what says so
    # on both platforms.
    with pytest.raises(ValueError, match=r"invalid subpath"):
        config(root="/d", subpath=subpath).resolve_subpath()


def test_a_subpath_reaching_outside_stops_the_run_before_it_starts(
    phase_tree, tmp_path
):
    dest = tmp_path / "out"

    with pytest.raises(ValueError, match=r"invalid subpath"):
        _scan(phase_tree, dest, 0, subpath="../../elsewhere")

    assert not dest.exists()


def test_writing_frames_over_the_source_is_refused(phase_tree):
    # A sequence is committed by replacing its folder whole and atomically, so
    # this ran to "3 of 3 done" and exited 0 with every acquisition it had read
    # gone. Nothing else in the run would have said so: the frames it wrote are
    # the ones it meant to write, and they are where it meant to put them.
    before = _written(phase_tree)

    with pytest.raises(ValueError, match=r"frames would land on the source"):
        _scan(phase_tree, phase_tree, 0, overwrite=True)

    assert _written(phase_tree) == before


def test_a_destination_a_later_run_would_find_again_is_refused(phase_tree):
    # Nested under the source and laid out the way the source is, this run's
    # output is more sequences to the next run, which folds a dataset boundary
    # over frames that are already filtered and doubles the tree again.
    with pytest.raises(ValueError, match=r"frames would land on the source"):
        _scan(phase_tree, phase_tree / "filtered", 0)


def test_a_destination_that_holds_the_source_is_refused(phase_tree):
    # `Phase/Float` is replaced whole too, and that takes `Bin` with it, so the
    # check cannot be an equality between the two folders.
    with pytest.raises(ValueError, match=r"frames would land on the source"):
        _scan(phase_tree, phase_tree, 0, subpath="Phase/Float", overwrite=True)


def test_frames_may_be_written_beside_the_ones_they_were_read_from(phase_tree):
    # Which is why the check is about the folders and not about the roots: a
    # filtered tree kept next to the raw one is how this layout is arranged
    # already, and it collides with neither. The source survives, and a later
    # search finds the sequences it found before rather than twice as many.
    before = _written(phase_tree)

    _scan(phase_tree, phase_tree, 0, subpath=FILTERED)

    assert _written(phase_tree) == before
    assert len(search_sources(SourceConfig(root=str(phase_tree)))[0]) == SEQUENCES
    assert len(PhaseBinFolder(phase_tree / "TL_00" / FILTERED)) == FRAMES


def test_a_run_writing_no_frames_may_share_the_source_root(phase_tree):
    # The document goes beside the dataset rather than into any sequence of it,
    # so a run that only measures touches nothing it reads.
    _scan(phase_tree, phase_tree, 0, save_frames=False, save_ranges=True)

    assert _document(phase_tree)["coverage"]["covered"] == SEQUENCES


def test_a_factory_offers_one_stage_per_sequence(phase_tree, tmp_path):
    source = SourceConfig(root=str(phase_tree))
    stages = build_phase_stages(source, name=STAGE, output_root=tmp_path)

    assert len(stages) == SEQUENCES


def test_a_run_without_a_target_writes_nothing(phase_tree, tmp_path):
    # `target_config=None` is how a run that only reads, such as one over a
    # cached source or a timing pass, says it wants no side branch. The root
    # is still named, so
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
    # land under its root, with nothing coming back to say they did.
    dest = tmp_path / "out"

    _scan(phase_tree, dest, 2)

    for sequence in range(SEQUENCES):
        written = PhaseBinFolder(dest / f"TL_{sequence:02d}" / PHASE_FLOAT_BIN)
        assert len(written) == FRAMES


def test_the_written_frames_are_always_in_radians(phase_tree, tmp_path):
    # A metric reads optical path difference out of phase, so the cache a run
    # leaves carries one unit, and the header still holds `height_scale` for
    # whoever wants metres back.
    dest = tmp_path / "out"

    _scan(phase_tree, dest, 0)

    header = read_phase_bin_header(dest / "TL_00" / PHASE_FLOAT_BIN / "00000_phase.bin")
    assert header.unit is PhaseUnit.RADIANS


def test_a_written_sequence_says_which_frames_it_was_made_from(phase_tree, tmp_path):
    # A phase header holds no time and no source name, and the folder is
    # renumbered from zero, so at a stride the cache's frame 1 is the source's
    # frame 2 and nothing in the frames says so. (2) derives its time radius
    # from the beat period, which it reads out of the source's `timestamps.txt`.
    dest = tmp_path / "out"
    source = SourceConfig(root=str(phase_tree), frame_step=2)
    target = TargetConfig(
        root=str(dest), subpath=FILTERED, save_frames=True, save_ranges=False
    )
    compute = ComputeConfig(device="cpu", workers=0, show_progress=False)

    run_all(build_phase_stages(source, target, name=STAGE, output_root=dest), compute)

    folder = dest / "TL_00" / FILTERED
    record = json.loads((folder / RECORD_FILE).read_text(encoding="utf-8"))

    assert record["source"] == "TL_00"
    assert record["settings"]["source"] == {
        "subpath": PHASE_FLOAT_BIN,
        "frame_start": 0,
        "frame_step": 2,
        "frame_count": None,
    }
    kept = range(0, FRAMES, 2)
    assert record["frames"] == [f"{index:05d}_phase.bin" for index in kept]
    assert len(PhaseBinFolder(folder)) == len(record["frames"])


def _cache(phase_tree: Path, dest: Path, **target: object) -> None:
    """Run a frames-only pass, so a second one has something to reuse."""
    source = SourceConfig(root=str(phase_tree))
    config = TargetConfig(
        root=str(dest), subpath=FILTERED, save_frames=True, save_ranges=False, **target
    )
    compute = ComputeConfig(device="cpu", workers=0, show_progress=False)

    run_all(build_phase_stages(source, config, name=STAGE, output_root=dest), compute)


def _mtimes(dest: Path) -> dict[str, float]:
    return {
        path.relative_to(dest).as_posix(): path.stat().st_mtime
        for path in sorted(dest.rglob("*.bin"))
    }


def test_a_second_run_keeps_the_frames_the_first_one_wrote(
    phase_tree, tmp_path, caplog
):
    # The whole point: 470 GB of reading and writing, and nothing about the
    # sequences changed. Compared by mtime rather than by content, since
    # rewriting the same bytes is exactly the work being avoided.
    dest = tmp_path / "out"
    _cache(phase_tree, dest)
    before = _mtimes(dest)

    with caplog.at_level(logging.INFO):
        _cache(phase_tree, dest, if_frames_exist="reuse")

    assert _mtimes(dest) == before
    said = [record.getMessage() for record in caplog.records]
    assert f"kept {SEQUENCES} sequences already written" in said
    assert f"{SEQUENCES} of {SEQUENCES} ready" in " ".join(said)


def test_a_sequence_whose_filter_changed_is_written_again(phase_tree, tmp_path):
    # The record is what tells them apart: same frames, different settings, so
    # the folder describes numbers this run would not produce.
    dest = tmp_path / "out"
    _cache(phase_tree, dest)
    before = _mtimes(dest)

    source = SourceConfig(root=str(phase_tree))
    config = TargetConfig(
        root=str(dest),
        subpath=FILTERED,
        save_frames=True,
        save_ranges=False,
        if_frames_exist="reuse",
    )
    filtered = {
        "_target_": f"{MedianConfig.__module__}.MedianConfig",
        "radius": [1, 1, 0],
    }
    compute = ComputeConfig(device="cpu", workers=0, show_progress=False)
    stages = build_phase_stages(
        source, config, OmegaConf.create(filtered), name=STAGE, output_root=dest
    )

    run_all(stages, compute)

    assert _mtimes(dest) != before


def test_a_folder_missing_a_frame_is_not_reused(phase_tree, tmp_path):
    # A range part is one file and so is there or not; a folder can be half
    # removed. Reusing that leaves a short sequence reading as a whole one.
    dest = tmp_path / "out"
    _cache(phase_tree, dest)
    (dest / "TL_01" / FILTERED / f"{FRAMES - 1:05d}_phase.bin").unlink()

    _cache(phase_tree, dest, if_frames_exist="reuse")

    assert len(PhaseBinFolder(dest / "TL_01" / FILTERED)) == FRAMES


def test_a_tree_told_no_settings_leaves_the_folder_saying_nothing(phase_tree, tmp_path):
    # The record is what a caller asks for, so a tree built without one writes
    # frames and no more. Flow and metric caches start here.
    dest = tmp_path / "out"
    _, contents = build_sequences(
        SourceConfig(root=str(phase_tree)), parse_filter_config(None)
    )
    stages = _factory(phase_tree, FrameTree(dest, PHASE_FLOAT_BIN, contents))

    with contextlib.suppress(Exception):
        stages.run_stage(0, Device("cpu"))

    assert not (dest / "TL_00" / PHASE_FLOAT_BIN / RECORD_FILE).exists()
    assert (dest / "TL_00" / PHASE_FLOAT_BIN / "00000_phase.bin").exists()


def test_a_source_in_metres_is_converted_rather_than_relabelled(phase_tree, tmp_path):
    # Every source the suite builds is already in radians, so the conversion
    # `search_sources` asks for was a no-op everywhere, so the call could be
    # deleted and nothing would fail. A source in metres is what makes it do
    # something: `height_scale` is the height one radian stands for, in m per
    # rad, so the values come back divided by it rather than restamped.
    dest = tmp_path / "out"
    height = np.linspace(0.0, 4 * HEIGHT_SCALE, 20, dtype=np.float32).reshape(4, 5)
    for frame in range(FRAMES):
        save_phase_bin(
            phase_tree / "TL_00" / PHASE_FLOAT_BIN / f"{frame:05d}_phase.bin",
            height,
            pixel_size=PIXEL_SIZE,
            height_scale=HEIGHT_SCALE,
            unit=PhaseUnit.METERS,
            overwrite=True,
        )

    _scan(phase_tree, dest, 0, save_frames=True)

    written = PhaseBinFolder(dest / "TL_00" / PHASE_FLOAT_BIN)
    radians = height / HEIGHT_SCALE

    assert read_phase_bin_header(written.get_file(0)).unit is PhaseUnit.RADIANS
    np.testing.assert_allclose(np.asarray(written[0]), radians, rtol=1e-5)


def test_the_job_names_the_stage_every_line_is_filed_under(
    phase_tree, tmp_path, caplog
):
    # Nothing here is preprocessing by nature, since the same filtered-phase run
    # is postprocessing behind a hologram reconstruction, so the name is the job's
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

    # The other two are whole, frames committed and ranges folded, and the one
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
        "found": 3,
        "selected": 3,
        "covered": 2,
        "reused": 0,
        "skipped": ["TL_01"],
        "unselected": [],
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


class _Exploding:
    """A side branch whose hook gives up part way through a sequence."""

    def get_hook(self, source):
        def hook(step: object) -> None:
            msg = "the branch gave up"
            raise RuntimeError(msg)

        return hook


@pytest.mark.parametrize("branch", (_Branch(None), _Exploding()))
def test_a_finished_sequence_lets_go_of_its_window(phase_tree, monkeypatch, branch):
    # The factory holds every sequence for the whole run, so a window kept past
    # the item it belongs to is kept to the end, once per sequence, on the
    # device, and again in each worker's own copy of them. A sequence that gave
    # up holds one too, so the release has to happen either way. Spied rather
    # than measured, since what the release costs is a device this test has no
    # reason to need.
    sequences, _ = build_sequences(
        SourceConfig(root=str(phase_tree)), parse_filter_config(None)
    )
    stages = PhaseStageFactory(sequences, branch, name=STAGE)
    released: list[str] = []
    monkeypatch.setattr(sequences[0], "release", lambda: released.append("let go"))

    with contextlib.suppress(RuntimeError):
        stages.run_stage(0, Device("cpu"))

    assert released == ["let go"]


class _Unclosable:
    """A branch that gathers across the run and cannot commit at the end."""

    def get_hook(self, source) -> _Hook:
        return _Hook(None)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            msg = "the branch could not commit"
            raise OSError(msg)


class _Watching:
    """A branch that records what it was told the run ended with."""

    def __init__(self) -> None:
        self.closed_with: list[object] = []

    def get_hook(self, source) -> _Hook:
        return _Hook(None)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.closed_with.append(exc_type)


def _factory(phase_tree, *branches):
    sequences, _ = build_sequences(
        SourceConfig(root=str(phase_tree)), parse_filter_config(None)
    )

    return PhaseStageFactory(sequences, *branches, name=STAGE)


class _Unsourced:
    """A branch holding outputs of sequences the source no longer has."""

    def __init__(self, *names: str) -> None:
        self._names = list(names)

    def get_hook(self, source):
        return None

    def list_unsourced(self) -> list[str]:
        return self._names


class _Counting:
    """A branch whose hook also says how much it has taken, as a writer might."""

    class _Hook:
        def __init__(self) -> None:
            self.seen: list[int] = []

        def __call__(self, step) -> None:
            self.seen.append(step.index)

        def __len__(self) -> int:
            return len(self.seen)

    def __init__(self) -> None:
        self.hook = self._Hook()

    def get_hook(self, source):
        return self.hook


def test_a_hook_that_is_empty_is_still_a_hook(phase_tree):
    # `get_hook` returns `Hook | None`, so the filter has to ask for `None` and
    # not for truth: a hook counting what it has taken is falsy before it takes
    # anything, and one dropped here leaves the sequence reported as done with
    # nothing written for it.
    branch = _Counting()
    stages = _factory(phase_tree, branch)

    assert stages.run_stage(0, Device("cpu"))
    assert branch.hook.seen == list(range(FRAMES))


def test_a_run_says_which_outputs_have_no_sequence_behind_them(phase_tree, caplog):
    # Said whatever the branch then does with them, and before the frames are
    # spent: a dataset that shrank and a share that came up half read the same
    # from here, and only whoever started the run can tell them apart.
    stages = _factory(phase_tree, _Unsourced("plate/TL_09"), _Unsourced("plate/TL_09"))

    with caplog.at_level(logging.INFO):
        _run_nothing(stages)

    said = [record.getMessage() for record in caplog.records]
    assert "1 output with no source: plate/TL_09" in said


def test_a_run_says_it_removed_them_and_not_only_that_they_were_due(
    phase_tree, tmp_path, caplog
):
    # The line before the run names them whatever the policy is, so on its own
    # it leaves an operator reading names with no word that anything was acted
    # on. Destructive, and until now only its failure was ever loud.
    out = tmp_path / "out"
    (out / "plate/TL_09" / PHASE_FLOAT_BIN).mkdir(parents=True)
    _, contents = build_sequences(
        SourceConfig(root=str(phase_tree)), parse_filter_config(None)
    )
    tree = FrameTree(out, PHASE_FLOAT_BIN, contents, if_sources_gone="delete")

    with caplog.at_level(logging.INFO):
        _run_nothing(_factory(phase_tree, tree))

    said = [record.getMessage() for record in caplog.records]
    assert "1 output with no source: plate/TL_09" in said
    assert "removed 1 folder with no source" in said
    assert not (out / "plate").exists()


def test_a_run_whose_outputs_all_have_sources_says_nothing_about_it(phase_tree, caplog):
    # The line is for the surprising state, so a run with nothing odd about it
    # must not carry one.
    stages = _factory(phase_tree, _Unsourced())

    with caplog.at_level(logging.INFO):
        _run_nothing(stages)

    assert not [line for line in caplog.messages if "no source" in line]


def _run_nothing(stages) -> None:
    """Open the branches and close them, in a single statement."""
    with stages.running():
        pass


def _give_up(stages) -> None:
    """Open the branches and give up inside, in a single statement."""
    with stages.running():
        msg = "the run gave up"
        raise RuntimeError(msg)


def test_one_branch_that_cannot_commit_does_not_silence_the_others(phase_tree, caplog):
    # The branches write separately, the same as the hooks of one sequence a
    # level down, and one `ExitStack` over them handed each whatever the last
    # raised.
    # What committed still says so: a branch that committed nothing reports
    # nothing anyway, so the line is only ever about work that landed.
    stages = _factory(phase_tree, _Unclosable(), _Branch("spoke"))

    with caplog.at_level(logging.INFO), pytest.raises(OSError, match="not commit"):
        _run_nothing(stages)

    assert "spoke" in [record.getMessage().strip() for record in caplog.records]


def test_a_run_that_gave_up_reaches_every_branch(phase_tree):
    # The other direction: what the run itself ended with does go to all of
    # them, since that is the outcome they bracket.
    watching = _Watching()

    with pytest.raises(RuntimeError, match="the run gave up"):
        _give_up(_factory(phase_tree, watching))

    assert watching.closed_with == [RuntimeError]


def test_a_branch_with_nothing_to_say_adds_no_line(phase_tree, caplog):
    # `report()` answering `None` is the contract for a branch that committed
    # nothing, and the block has to leave it out rather than log an empty line.
    # Both scopes are covered here: a hook reports per sequence, a branch once.
    def lines(said: str | None) -> list[str]:
        stages = PhaseStageFactory(
            build_sequences(
                SourceConfig(root=str(phase_tree)), parse_filter_config(None)
            )[0],
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


def _branches(
    tmp_path,
    *,
    sequence_names=("TL_00",),
    selected=None,
    subpath=None,
    step=1,
    **target,
):
    source = SourceConfig(root="/dataset", subpath=subpath, frame_step=step)
    config = TargetConfig(root=str(tmp_path), **target)
    contents = dict.fromkeys(sequence_names, ("00000_phase.bin",))

    return build_branches(
        source,
        config,
        parse_filter_config(None),
        tmp_path,
        contents,
        selected,
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
        "frame_start": 0,
        "frame_step": 3,
        "frame_count": None,
    }
    assert document.settings["filter"]["kind"] == "identity"


def test_an_unset_subpath_is_recorded_as_the_one_that_was_read(tmp_path):
    # `None` means the Koala default rather than "no subpath", and a document
    # saying `null` could not be compared against a run that named it outright.
    (document,) = _branches(tmp_path, subpath=None)

    assert document.settings["source"]["subpath"] == PHASE_FLOAT_BIN


def test_the_selection_stays_out_of_the_settings(tmp_path):
    # `include` / `exclude` change which sequences a run covers, not what any
    # sequence's numbers mean, and `coverage` already reports that. Recording
    # them here would refuse reuse to a run that narrowed its selection.
    source = SourceConfig(root="/dataset", include=["TL_00"], exclude=["TL_09"])
    config = TargetConfig(root=str(tmp_path), save_ranges=True)

    (document,) = build_branches(
        source, config, parse_filter_config(None), tmp_path, {"TL_00": ()}
    )

    assert set(document.settings["source"]) == {
        "subpath",
        "frame_start",
        "frame_step",
        "frame_count",
    }


def test_the_contents_reaches_the_document_that_reports_on_it(tmp_path):
    (document,) = _branches(tmp_path, sequence_names=("TL_00", "TL_01", "TL_02"))

    assert tuple(document.contents) == ("TL_00", "TL_01", "TL_02")
    assert document.selected == ("TL_00", "TL_01", "TL_02")


def test_a_narrowed_run_keeps_the_whole_dataset_in_its_contents(tmp_path):
    # What the document counts against: a run given one of three still describes
    # a dataset of three, and coverage measured against the selection alone
    # would call one part of it complete.
    (document,) = _branches(
        tmp_path, sequence_names=("TL_00", "TL_01", "TL_02"), selected=["TL_01"]
    )

    assert tuple(document.contents) == ("TL_00", "TL_01", "TL_02")
    assert document.selected == ("TL_01",)


def test_branches_are_refused_before_any_of_them_is_built(tmp_path):
    with pytest.raises(ValueError, match=r"nothing to do"):
        _branches(tmp_path, save_frames=False, save_ranges=False)


def _frame_tree(tmp_path, **target):
    return build_branches(
        SourceConfig(root="/dataset"),
        TargetConfig(root=str(tmp_path), save_frames=True, save_ranges=False, **target),
        parse_filter_config(None),
        tmp_path,
        {"TL_00": ()},
    )


def test_a_target_naming_a_subpath_is_where_the_frames_go(tmp_path):
    # Without one they follow the source's layout, which is what puts them on
    # top of it whenever the two roots are the same.
    (tree,) = _frame_tree(tmp_path, subpath=FILTERED)

    assert tree.subpath == FILTERED


def test_an_unset_target_subpath_follows_the_source(tmp_path):
    (tree,) = _frame_tree(tmp_path)

    assert tree.subpath == PHASE_FLOAT_BIN


def test_the_settings_record_the_subpath_that_was_read(tmp_path):
    # A later run compares against where the frames came from, so writing them
    # somewhere else must not change what the document says it describes.
    source = SourceConfig(root="/dataset")
    config = TargetConfig(root=str(tmp_path), subpath=FILTERED, save_ranges=True)

    (document,) = build_branches(
        source, config, parse_filter_config(None), tmp_path, {"TL_00": ()}
    )

    assert document.settings["source"]["subpath"] == PHASE_FLOAT_BIN


def test_branches_are_refused_where_the_frames_would_land_on_the_source(tmp_path):
    # The same verdict `build_phase_stages` reaches before its search, made here
    # too, since a caller may assemble the branches without going through it.
    with pytest.raises(ValueError, match=r"frames would land on the source"):
        build_branches(
            SourceConfig(root=str(tmp_path)),
            TargetConfig(root=str(tmp_path), save_frames=True),
            parse_filter_config(None),
            tmp_path,
            ["TL_00"],
        )


# ---------------------------- the configuration log ----------------------- #


def _logged(caplog, source, target=None, kernel=None, output_root="/out"):
    with caplog.at_level(logging.INFO):
        log_configs(
            source,
            target,
            kernel or parse_filter_config(None),
            output_root,
            name=STAGE,
        )

    return [record.getMessage() for record in caplog.records]


def test_the_target_line_says_where_the_run_writes_not_what_placed_it(caplog):
    # `target.root` places the job's directory and nothing reads it to write
    # with: a sweep gives each job one of its own beneath it, so both jobs
    # logged the same `target:` while writing to `<root>/0` and `<root>/1`. The
    # line follows the branches, which is the path a reader goes looking in.
    logged = _logged(
        caplog,
        SourceConfig(root="/dataset"),
        TargetConfig(root="/out"),
        output_root="/out/0",
    )

    assert "target: /out/0" in logged
    assert "target: /out" not in logged


def test_each_block_is_tagged_by_what_it_configures(caplog):
    # A tag apiece rather than a verb, so a reader looking for one of the three
    # finds it by name, and the run's own lines below are never mistaken for
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
    # `overrides.yaml` whenever the command line is what set it, which is the
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
            overrides=["filter=median_cuboid_2x2x1"],
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


@pytest.mark.parametrize("name", ("value.range", "ranges.txt"))
def test_a_range_file_carrying_another_extension_names_the_setting(caplog, name):
    # The library's refusal names the extensions it takes but not the setting
    # that holds one, and there are two `.json` names in this configuration --
    # so the reader was left with an extension and no key to go and change.
    with pytest.raises(ValueError, match=r"invalid `target.range_file`"):
        _logged(
            caplog, SourceConfig(root="/d"), TargetConfig(root="/o", range_file=name)
        )


def test_the_written_frames_take_the_shape_the_source_was_read_in(caplog):
    # The mirroring shown rather than claimed: the two lines carry the same
    # template, close enough to read against each other, so a target that named
    # no subpath of its own and stopped following the source would not match.
    logged = _logged(
        caplog,
        SourceConfig(root="/dataset", subpath="Phase/Other"),
        TargetConfig(root="/out", save_frames=True),
    )

    assert "  reading <sequence>/Phase/Other" in logged
    assert "  writing the filtered frames to <sequence>/Phase/Other" in logged


def test_a_target_naming_a_subpath_says_that_one(caplog):
    # The two lines part company here, and the one about writing has to follow
    # the target: a reader checking where the output went reads this line.
    logged = _logged(
        caplog,
        SourceConfig(root="/dataset"),
        TargetConfig(root="/out", subpath=FILTERED, save_frames=True),
    )

    assert f"  reading <sequence>/{PHASE_FLOAT_BIN}" in logged
    assert f"  writing the filtered frames to <sequence>/{FILTERED}" in logged


@pytest.mark.parametrize(
    ("policy", "said"),
    (
        ("overwrite", "  replacing the ranges already there"),
        ("reuse", "  keeping the ranges already there that still describe this run"),
    ),
)
def test_a_run_says_what_it_does_with_what_is_already_there(caplog, policy, said):
    # Refusing is the default and says nothing; each of the other two is named,
    # since silence can separate two states but not three.
    target = TargetConfig(root="/o", if_ranges_exist=policy)

    assert said in _logged(caplog, SourceConfig(root="/d"), target)

    caplog.clear()
    refused = _logged(caplog, SourceConfig(root="/d"), TargetConfig(root="/o"))
    assert not [line for line in refused if "already there" in line]


def test_a_run_that_will_delete_says_so_before_it_reads_anything(caplog):
    # The one setting whose default is the safe one and whose other value is
    # not undoable, so it is named up front rather than only in what it removed.
    target = TargetConfig(root="/o", if_sources_gone="delete")

    said = "  dropping outputs whose sequence the source has lost"
    assert said in _logged(caplog, SourceConfig(root="/d"), target)

    caplog.clear()
    kept = _logged(caplog, SourceConfig(root="/d"), TargetConfig(root="/o"))
    assert not [line for line in kept if "dropping" in line]


@pytest.mark.parametrize(
    ("source", "said"),
    (
        (SourceConfig(root="/d"), None),
        (SourceConfig(root="/d", frame_step=2), "  reading frames 0, 2, 4, ..."),
        (SourceConfig(root="/d", frame_start=5), "  reading frames 5, 6, 7, ..."),
        (
            SourceConfig(root="/d", frame_start=1, frame_step=3, frame_count=2),
            "  reading frames 1, 4 (at most 2 frames)",
        ),
        (
            SourceConfig(root="/d", frame_step=2, frame_count=9),
            "  reading frames 0, 2, 4, ... (at most 9 frames)",
        ),
    ),
)
def test_a_run_says_which_frames_it_takes_unless_it_takes_them_all(
    caplog, source, said
):
    # Shown as positions rather than as the three settings: what a reader
    # checks is whether these are the frames they meant.
    logged = _logged(caplog, source)

    if said is None:
        assert not [line for line in logged if "reading frames" in line]
    else:
        assert said in logged


def test_a_run_that_will_refuse_a_short_sequence_says_so_up_front(caplog):
    # The one value of `if_frames_short` that can end the run, so it is named
    # with the rest of what the run was told rather than only in the failure.
    source = SourceConfig(root="/d", frame_count=9, if_frames_short="error")

    assert "  refusing a sequence that cannot supply them" in _logged(caplog, source)

    caplog.clear()
    taking = SourceConfig(root="/d", frame_count=9)
    assert not [line for line in _logged(caplog, taking) if "refusing" in line]


def test_a_run_without_a_target_says_nothing_about_writing(caplog):
    logged = _logged(caplog, SourceConfig(root="/dataset"))

    assert not any(line.startswith("target") for line in logged)


def test_a_run_with_no_target_does_not_blame_a_branch_for_holding_anything(
    phase_tree, caplog
):
    # No branch was asked and none declined: there is simply nothing this run
    # wants. Naming reuse as the cause would send a reader looking for a cache.
    stages = build_phase_stages(
        SourceConfig(root=str(phase_tree)), name=STAGE, output_root="/out"
    )

    with caplog.at_level(logging.INFO):
        assert not stages.run_stage(0, Device("cpu"))

    said = " ".join(record.getMessage() for record in caplog.records)
    assert "nothing to do: this run writes nothing" in said
    assert "already holds" not in said
