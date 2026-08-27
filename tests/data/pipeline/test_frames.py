from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from iivs.dhm.data.koala import PHASE_FLOAT_BIN
from iivs.dhm.data.phase import PhaseBinFolder, PhaseUnit, read_phase_bin_header

from iivs_cardio.common.pipeline import Step
from iivs_cardio.common.pipeline.frames import RECORD_FILE
from iivs_cardio.data.pipeline import FrameTree, phase_frame_writer

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


def _tree(tmp_path: Path, *names: str, **policy) -> FrameTree:
    return FrameTree(tmp_path, PHASE_FLOAT_BIN, dict.fromkeys(names, ()), **policy)


def test_a_tree_has_to_be_told_what_the_source_holds(tmp_path):
    # Without it every folder here is unsourced, and `if_unsourced` would take
    # the whole tree: hours of filtering, over a source that is simply absent
    # from an argument nobody passed.
    with pytest.raises(TypeError, match="contents"):
        FrameTree(tmp_path, PHASE_FLOAT_BIN)  # type: ignore[call-arg]


def test_a_tree_makes_its_own_root_rather_than_leaving_it_to_a_writer(tmp_path):
    # A writer takes away the folders its own opening made, and the first one
    # through made this too where nothing else had. One sequence giving up then
    # took the whole output tree with it.
    root = tmp_path / "out"

    with _tree(root, "TL_00"):
        assert root.is_dir()

    assert root.is_dir()


def _sequence(tmp_path: Path, name: str) -> Path:
    folder = tmp_path / name / PHASE_FLOAT_BIN
    folder.mkdir(parents=True)
    (folder / "00000_phase.bin").write_bytes(b"")

    return folder


def test_a_tree_names_the_sequences_it_holds_rather_than_the_folders(tmp_path):
    # The layout is three levels deep, so stopping one short would call every
    # level of one sequence a sequence of its own.
    for name in ("plate/TL_00", "plate/TL_01"):
        _sequence(tmp_path, name)

    assert _tree(tmp_path).list_sequences() == ["plate/TL_00", "plate/TL_01"]


def test_a_tree_refuses_a_policy_nobody_offers(tmp_path):
    # Config arrives as text whatever the field says, so a typo has to be
    # caught where the setting's own name can still be named.
    with pytest.raises(ValueError, match=r"unsupported if_present 'sync'"):
        _tree(tmp_path, if_present="sync")


def test_a_selection_naming_what_the_source_lacks_is_refused(tmp_path):
    # The contents is the dataset, so a selection outside it is the caller
    # having built one of the two from somewhere else.
    with pytest.raises(ValueError, match=r"selected 'TL_99'"):
        _tree(tmp_path, "TL_00", selected=["TL_99"])


def test_a_sequence_already_written_is_refused_before_a_frame_is_read(tmp_path):
    # The writer refuses the same thing, one sequence at a time, so a run over
    # 500 whose 300th is already there paid for 299 of them first. Here the
    # whole tree is in view and nothing has been read yet.
    _sequence(tmp_path, "plate/TL_00")
    tree = _tree(tmp_path, "plate/TL_00", "plate/TL_01")

    with pytest.raises(FileExistsError, match=r"1 sequence already written"):
        tree.__enter__()


@pytest.mark.parametrize(
    ("written", "why"),
    (
        ("{ not json", "unreadable"),
        ('["a", "b"]', "not a mapping"),
        ('{"settings": null}', "no sources listed"),
        ('{"settings": null, "sources": "00000_phase.bin"}', "sources not a list"),
        (
            '{"settings": {"filter": 1}, "sources": ["00000_phase.bin"]}',
            "other settings",
        ),
        ('{"settings": null, "sources": ["00099_phase.bin"]}', "other sources"),
    ),
)
def test_a_record_that_cannot_be_believed_is_written_again(tmp_path, written, why):
    # Judging is not reading: a record that says nothing usable means the run
    # cannot tell, and the safe answer to "cannot tell" is to write it again.
    _sequence(tmp_path, "TL_00")
    (tmp_path / "TL_00" / PHASE_FLOAT_BIN / RECORD_FILE).write_text(
        written, encoding="utf-8"
    )
    tree = FrameTree(
        tmp_path,
        PHASE_FLOAT_BIN,
        {"TL_00": ("00000_phase.bin",)},
        if_present="reuse",
    )

    with tree:
        pass

    assert tree.report() is None, why


def test_a_folder_holding_fewer_frames_than_its_record_is_written_again(tmp_path):
    # A range part is one file and so is there or not; a folder can be half
    # removed, and reusing that leaves a short sequence reading as a whole one.
    folder = _sequence(tmp_path, "TL_00")
    (folder / "00001_phase.bin").write_bytes(b"")
    contents = {"TL_00": ("00000_phase.bin", "00001_phase.bin")}
    record = {"settings": None, "sources": list(contents["TL_00"])}
    (folder / RECORD_FILE).write_text(json.dumps(record), encoding="utf-8")

    with FrameTree(tmp_path, PHASE_FLOAT_BIN, contents, if_present="reuse") as kept:
        pass

    (folder / "00001_phase.bin").unlink()
    with FrameTree(tmp_path, PHASE_FLOAT_BIN, contents, if_present="reuse") as short:
        pass

    assert kept.report() == "kept 1 sequence already written"
    assert short.report() is None


def test_something_that_is_not_a_frame_cannot_stand_in_for_one(tmp_path):
    # The count is what catches a half removed folder, so anything counted
    # that is not a frame lets a missing one through: one frame beside one
    # directory is the two a two-frame record expects.
    folder = _sequence(tmp_path, "TL_00")
    (folder / "00001_phase.bin").write_bytes(b"")
    contents = {"TL_00": ("00000_phase.bin", "00001_phase.bin")}
    record = {"settings": None, "sources": list(contents["TL_00"])}
    (folder / RECORD_FILE).write_text(json.dumps(record), encoding="utf-8")

    (folder / "00001_phase.bin").unlink()
    (folder / "leftover").mkdir()

    tree = FrameTree(tmp_path, PHASE_FLOAT_BIN, contents, if_present="reuse")
    with tree:
        pass

    assert tree.report() is None


def test_a_tree_reads_its_record_back_under_the_name_it_was_given(tmp_path):
    # The name reaches the writer and the reader from one setting, so a tree
    # told one thing must not go looking under the default.
    folder = _sequence(tmp_path, "TL_00")
    contents = {"TL_00": ("00000_phase.bin",)}
    record = {"settings": None, "sources": list(contents["TL_00"])}
    (folder / "origin.json").write_text(json.dumps(record), encoding="utf-8")

    tree = FrameTree(
        tmp_path, PHASE_FLOAT_BIN, contents, record_file="origin", if_present="reuse"
    )
    with tree:
        pass

    assert tree.report() == "kept 1 sequence already written"


def test_a_record_left_under_another_name_counts_as_a_frame(tmp_path):
    # The count is what catches a half removed folder, and only this tree's own
    # record is set aside: a folder written under a different name is a folder
    # this tree cannot vouch for, so it must not read as a whole one.
    folder = _sequence(tmp_path, "TL_00")
    contents = {"TL_00": ("00000_phase.bin",)}
    record = {"settings": None, "sources": list(contents["TL_00"])}
    (folder / "origin.json").write_text(json.dumps(record), encoding="utf-8")
    (folder / RECORD_FILE).write_text(json.dumps(record), encoding="utf-8")

    tree = FrameTree(
        tmp_path, PHASE_FLOAT_BIN, contents, record_file="origin", if_present="reuse"
    )
    with tree:
        pass

    assert tree.report() is None


@pytest.mark.parametrize("named", ("../up", "sub/down", "source.bin"))
def test_a_tree_refuses_a_record_name_it_could_not_write_beside_the_frames(
    tmp_path, named
):
    # Caught here rather than at the writer, since the tree reads by the name
    # too and a run refused part way through has already spent the frames.
    with pytest.raises(ValueError, match=r"invalid file name|unsupported extension"):
        _tree(tmp_path, record_file=named)


def test_a_sequence_the_run_was_not_given_is_not_in_its_way(tmp_path):
    # `error` is about what this run would write, so a folder outside the
    # selection is not a collision: a retry of one sequence must not be
    # refused by the ones that already succeeded.
    written = _sequence(tmp_path, "plate/TL_00")
    tree = _tree(tmp_path, "plate/TL_00", "plate/TL_01", selected=["plate/TL_01"])

    with tree:
        pass

    assert tree.list_sequences() == ["plate/TL_00"]  # untouched, not claimed
    assert (written / "00000_phase.bin").exists()
    assert tree.report() is None


def test_a_folder_inside_a_sequence_is_not_a_sequence_of_its_own(tmp_path):
    # The walk stops at a sequence rather than reaching one, so whatever a
    # writer left under the frames cannot be named alongside its own sequence.
    _sequence(tmp_path, "plate/TL_00")
    (tmp_path / "plate/TL_00" / PHASE_FLOAT_BIN / "leftover" / PHASE_FLOAT_BIN).mkdir(
        parents=True
    )

    assert _tree(tmp_path).list_sequences() == ["plate/TL_00"]


def test_a_tree_with_no_root_yet_holds_nothing(tmp_path):
    # A first run closes its tree before hydra's directory has anything in it,
    # and both walks reach for a root that is not there yet.
    tree = _tree(tmp_path / "not_here")

    assert tree.list_sequences() == []
    assert tree.clear_staging() is None


def test_a_folder_the_source_has_lost_is_named(tmp_path):
    # Sorted, which the listing rather than this gives: the walk hands back
    # the folders in order and dropping the sourced ones keeps them in it.
    _sequence(tmp_path, "kept")
    _sequence(tmp_path, "went")
    _sequence(tmp_path, "gone")

    assert _tree(tmp_path, "kept").list_unsourced() == ["gone", "went"]


def test_a_folder_the_source_has_lost_stays_unless_the_policy_says_otherwise(tmp_path):
    # Hours of filtering, and the source it came from is already not there to
    # make it again, so the same absence a half mounted share makes must not
    # be enough on its own.
    _sequence(tmp_path, "kept")
    _sequence(tmp_path, "gone")

    with _tree(tmp_path, "kept", if_present="overwrite"):
        pass

    assert (tmp_path / "gone" / PHASE_FLOAT_BIN).is_dir()


def test_a_folder_the_source_has_lost_goes_when_the_policy_says_so(tmp_path):
    _sequence(tmp_path, "kept")
    _sequence(tmp_path, "gone")

    with _tree(tmp_path, "kept", if_present="overwrite", if_unsourced="delete"):
        pass

    assert not (tmp_path / "gone").exists()
    assert (tmp_path / "kept" / PHASE_FLOAT_BIN).is_dir()


def test_a_removal_is_said_to_have_happened_and_not_only_to_have_been_due(tmp_path):
    # The line before the run names what has no source whatever the policy is,
    # so without this one an operator reads the names and never learns that
    # anything was acted on. Destructive, and only its failure was ever loud.
    _sequence(tmp_path, "kept")
    _sequence(tmp_path, "gone")
    tree = _tree(tmp_path, "kept", if_present="overwrite", if_unsourced="delete")

    assert tree.report() is None

    with tree:
        pass

    assert tree.report() == "removed 1 folder with no source"


def test_a_tree_that_took_nothing_away_reports_nothing(tmp_path):
    _sequence(tmp_path, "kept")
    tree = _tree(tmp_path, "kept", if_present="overwrite", if_unsourced="delete")

    with tree:
        pass

    assert tree.report() is None


def test_the_job_s_own_directory_survives_the_clearing(tmp_path):
    # The tree is rooted at the directory hydra gave the job, which is also
    # where the run's configuration and its logs are kept. Anything that swept
    # by shape rather than by name took the record of the run with it.
    (tmp_path / ".hydra").mkdir()
    (tmp_path / ".hydra" / "config.yaml").write_text("source:\n", encoding="utf-8")
    (tmp_path / "left_by_hand").mkdir()
    _sequence(tmp_path, "kept")

    with _tree(tmp_path, "kept", if_present="overwrite"):
        pass

    assert (tmp_path / ".hydra" / "config.yaml").exists()
    assert (tmp_path / "left_by_hand").is_dir()


def test_dropping_a_nested_unsourced_folder_takes_what_it_empties(tmp_path):
    # The invariant opening a document holds to as well: a sequence dropped
    # from a nested dataset would otherwise leave the path down to it standing,
    # which reads as a plate that is still there.
    _sequence(tmp_path, "plate/2026.03.11/kept")
    _sequence(tmp_path, "plate/2026.03.12/gone")

    with _tree(
        tmp_path,
        "plate/2026.03.11/kept",
        if_present="overwrite",
        if_unsourced="delete",
    ):
        pass

    assert not (tmp_path / "plate" / "2026.03.12").exists()
    assert (tmp_path / "plate" / "2026.03.11" / "kept").is_dir()


def test_what_a_killed_worker_staged_is_collected_by_the_next_run(tmp_path):
    # The writer climbs back down its own parents only while it is alive. A
    # worker killed outright leaves the staged folder and the shells above it,
    # and nothing else was in a position to find them.
    _sequence(tmp_path, "kept")
    staged = tmp_path / "died" / "Phase" / "Float" / ".Bin.k3j2h.tmp"
    staged.mkdir(parents=True)
    (staged / "00000_phase.bin").write_bytes(b"")

    with _tree(tmp_path, "kept", "died", if_present="overwrite"):
        pass

    assert not (tmp_path / "died").exists()
    assert (tmp_path / "kept" / PHASE_FLOAT_BIN).is_dir()


@pytest.mark.parametrize(
    ("policy", "replacing"), (("error", False), ("overwrite", True))
)
def test_the_policy_reaches_the_writer_that_acts_on_it(tmp_path, policy, replacing):
    _sequence(tmp_path, "TL_00")
    tree = _tree(tmp_path, "TL_00", if_present=policy)

    if replacing:
        assert tree.if_present == "overwrite"
    else:
        assert tree.if_present == "error"


PIXEL_SIZE = 1.5e-7
HEIGHT_SCALE = 2.0e-7


def _frames(count: int = 4, height: int = 4, width: int = 5) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [rng.random((height, width), dtype=np.float32) for _ in range(count)]


def _write(dest: Path, frames: Iterable[np.ndarray], **kwargs) -> None:
    writer = phase_frame_writer(
        dest, pixel_size=PIXEL_SIZE, height_scale=HEIGHT_SCALE, **kwargs
    )
    with writer:
        for index, frame in enumerate(frames):
            writer.write(Step(index, torch.from_numpy(frame)))


def test_written_folder_reads_back_with_the_same_values(tmp_path):
    frames = _frames()
    dest = tmp_path / "Bin"

    _write(dest, frames)

    folder = PhaseBinFolder(dest)
    assert len(folder) == len(frames)
    for index, frame in enumerate(frames):
        assert np.array_equal(folder[index], frame)


def test_frames_are_numbered_from_zero(tmp_path):
    # The source may be strided or start elsewhere; the written tree is dense.
    dest = tmp_path / "Bin"

    _write(dest, _frames(3))

    assert sorted(p.name for p in dest.iterdir()) == [
        "00000_phase.bin",
        "00001_phase.bin",
        "00002_phase.bin",
    ]


def test_the_header_carries_the_scale_it_was_given(tmp_path):
    dest = tmp_path / "Bin"

    _write(dest, _frames(1))

    header = read_phase_bin_header(dest / "00000_phase.bin")
    assert header.pixel_size == pytest.approx(PIXEL_SIZE)
    assert header.height_scale == pytest.approx(HEIGHT_SCALE)
    assert header.unit == PhaseUnit.RADIANS


def test_nanometres_are_normalized_to_metres(tmp_path):
    # The `.bin` header cannot hold NANOMETERS, so the writer converts values and
    # records METERS. A caller therefore states the unit its frames are in and
    # does not pre-convert, and doing both would scale twice.
    dest = tmp_path / "Bin"
    frame = np.array([[1000.0]], dtype=np.float32)  # nanometres

    _write(dest, [frame], unit=PhaseUnit.NANOMETERS)

    assert read_phase_bin_header(dest / "00000_phase.bin").unit == PhaseUnit.METERS
    assert PhaseBinFolder(dest)[0] == pytest.approx(1e-6)  # metres


def test_an_existing_folder_is_kept_unless_overwrite(tmp_path):
    dest = tmp_path / "Bin"
    _write(dest, _frames(2))

    with pytest.raises(FileExistsError):
        _write(dest, _frames(3))

    assert len(PhaseBinFolder(dest)) == 2  # the first write is still intact


def test_overwrite_replaces_the_folder(tmp_path):
    dest = tmp_path / "Bin"
    _write(dest, _frames(2))

    _write(dest, _frames(3), overwrite=True)

    assert len(PhaseBinFolder(dest)) == 3


def test_a_failure_part_way_leaves_the_previous_folder_untouched(tmp_path):
    dest = tmp_path / "Bin"
    _write(dest, _frames(2))

    def stream():
        yield from _frames(1)
        msg = "boom"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="boom"):
        _write(dest, stream(), overwrite=True)

    assert len(PhaseBinFolder(dest)) == 2  # staged: no half-written tree


def test_an_empty_sequence_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="no frame was written"):
        _write(tmp_path / "Bin", [])


@pytest.mark.parametrize("bad", (np.nan, np.inf, -np.inf))
def test_a_nonfinite_frame_is_refused_rather_than_cached(tmp_path, bad):
    # What this writes is the cache the next stage reads, and the format itself
    # stores a NaN happily, so a value let through here is one the run that
    # meets it has no way to trace back. The folder is left uncommitted.
    dest = tmp_path / "Bin"
    frame = _frames(1)[0]
    frame[0, 0] = bad

    with pytest.raises(ValueError, match="finite"):
        _write(dest, [frame])

    assert not dest.exists()


def test_an_absent_step_writes_nothing(tmp_path):
    # The tail of a stage spanning several steps: the folder ends earlier.
    dest = tmp_path / "Bin"
    writer = phase_frame_writer(dest, pixel_size=PIXEL_SIZE, height_scale=HEIGHT_SCALE)

    with writer:
        writer.write(Step(0, torch.from_numpy(_frames(1)[0])))
        writer.write(Step[torch.Tensor, None](1, None))

    assert len(PhaseBinFolder(dest)) == 1
