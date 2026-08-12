from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from iivs.dhm.data.koala import PHASE_FLOAT_BIN

from iivs_cardio.data.pipeline import FrameTree
from iivs_cardio.data.writer import RECORD_FILE

if TYPE_CHECKING:
    from pathlib import Path


def _tree(tmp_path: Path, *names: str, **policy) -> FrameTree:
    return FrameTree(tmp_path, PHASE_FLOAT_BIN, dict.fromkeys(names, ()), **policy)


def test_a_tree_has_to_be_told_what_the_source_holds(tmp_path):
    # Without it every folder here is unsourced, and `if_unsourced` would take
    # the whole tree: hours of filtering, over a source that is simply absent
    # from an argument nobody passed.
    with pytest.raises(TypeError, match="contents"):
        FrameTree(tmp_path, PHASE_FLOAT_BIN)  # type: ignore[call-arg]


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
        ('{"settings": null}', "no frames listed"),
        ('{"settings": null, "frames": "00000_phase.bin"}', "frames not a list"),
        (
            '{"settings": {"filter": 1}, "frames": ["00000_phase.bin"]}',
            "other settings",
        ),
        ('{"settings": null, "frames": ["00099_phase.bin"]}', "other frames"),
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
    record = {"settings": None, "frames": list(contents["TL_00"])}
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
    record = {"settings": None, "frames": list(contents["TL_00"])}
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
    record = {"settings": None, "frames": list(contents["TL_00"])}
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
    record = {"settings": None, "frames": list(contents["TL_00"])}
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
    _sequence(tmp_path, "kept")
    _sequence(tmp_path, "gone")

    assert _tree(tmp_path, "kept").list_unsourced() == ["gone"]


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
