from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from iivs.dhm.data.koala import PHASE_FLOAT_BIN

from iivs_cardio.data.pipeline import FrameTree

if TYPE_CHECKING:
    from pathlib import Path


def _tree(tmp_path: Path, *names: str, **policy) -> FrameTree:
    return FrameTree(tmp_path, PHASE_FLOAT_BIN, dict.fromkeys(names, ()), **policy)


def test_a_tree_has_to_be_told_what_the_source_holds(tmp_path):
    # Without it every folder here is unsourced, and `if_sources_gone` would take
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


def test_a_tree_refuses_the_policy_it_cannot_yet_carry_out(tmp_path):
    # `ExistingOutputPolicy` offers three and a tree can do two: without this
    # the third arrives as `overwrite=False` and quietly behaves as 'error'.
    with pytest.raises(ValueError, match="expected 'error', 'overwrite'"):
        _tree(tmp_path, if_frames_exist="reuse")


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

    with _tree(tmp_path, "kept"):
        pass

    assert (tmp_path / "gone" / PHASE_FLOAT_BIN).is_dir()


def test_a_folder_the_source_has_lost_goes_when_the_policy_says_so(tmp_path):
    _sequence(tmp_path, "kept")
    _sequence(tmp_path, "gone")

    with _tree(tmp_path, "kept", if_sources_gone="delete"):
        pass

    assert not (tmp_path / "gone").exists()
    assert (tmp_path / "kept" / PHASE_FLOAT_BIN).is_dir()


def test_a_removal_is_said_to_have_happened_and_not_only_to_have_been_due(tmp_path):
    # The line before the run names what has no source whatever the policy is,
    # so without this one an operator reads the names and never learns that
    # anything was acted on. Destructive, and only its failure was ever loud.
    _sequence(tmp_path, "kept")
    _sequence(tmp_path, "gone")
    tree = _tree(tmp_path, "kept", if_sources_gone="delete")

    assert tree.report() is None

    with tree:
        pass

    assert tree.report() == "removed 1 folder with no source"


def test_a_tree_that_took_nothing_away_reports_nothing(tmp_path):
    _sequence(tmp_path, "kept")
    tree = _tree(tmp_path, "kept", if_sources_gone="delete")

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

    with _tree(tmp_path, "kept"):
        pass

    assert (tmp_path / ".hydra" / "config.yaml").exists()
    assert (tmp_path / "left_by_hand").is_dir()


def test_dropping_a_nested_unsourced_folder_takes_what_it_empties(tmp_path):
    # The invariant opening a document holds to as well: a sequence dropped
    # from a nested dataset would otherwise leave the path down to it standing,
    # which reads as a plate that is still there.
    _sequence(tmp_path, "plate/2026.03.11/kept")
    _sequence(tmp_path, "plate/2026.03.12/gone")

    with _tree(tmp_path, "plate/2026.03.11/kept", if_sources_gone="delete"):
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

    with _tree(tmp_path, "kept", "died"):
        pass

    assert not (tmp_path / "died").exists()
    assert (tmp_path / "kept" / PHASE_FLOAT_BIN).is_dir()


@pytest.mark.parametrize(
    ("policy", "replacing"), (("error", False), ("overwrite", True))
)
def test_the_policy_reaches_the_writer_that_acts_on_it(tmp_path, policy, replacing):
    _sequence(tmp_path, "TL_00")
    tree = _tree(tmp_path, "TL_00", if_frames_exist=policy)

    if replacing:
        assert tree.if_frames_exist == "overwrite"
    else:
        assert tree.if_frames_exist == "error"
