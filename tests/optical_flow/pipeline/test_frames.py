from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from iivs_cardio.common.pipeline import Step
from iivs_cardio.common.pipeline.frames import RECORD_FILE
from iivs_cardio.optical_flow.data import OpticalFlowFolder
from iivs_cardio.optical_flow.pipeline import FlowTree

SUBPATH = "flow"


class _Sequence:
    """The whole of what a flow tree asks of the sequence it writes."""

    def __init__(self, name: str) -> None:
        self.name = name


def _phase_names(count: int) -> tuple[str, ...]:
    return tuple(f"{i:05d}_phase.bin" for i in range(count))


def _tree(tmp_path: Path, name: str, frames: int, **policy) -> FlowTree:
    return FlowTree(tmp_path, SUBPATH, {name: _phase_names(frames)}, **policy)


def _flow() -> torch.Tensor:
    return torch.zeros((2, 8, 8), dtype=torch.float32)


def _write(tree: FlowTree, source: _Sequence, frames: tuple[str, ...]) -> None:
    """Run a whole sequence through the tree, as a stage's hooks would."""
    with tree:
        writer = tree.get_hook(source)
        assert writer is not None
        with writer:
            for index, name in enumerate(frames):
                writer(Step(index, _flow(), Path(name)))


def test_a_flow_tree_writes_one_field_per_pair(tmp_path):
    # Five frames make four flows, and the folder is numbered from zero.
    source = _Sequence("plate/TL_00")
    tree = _tree(tmp_path, source.name, 5)

    _write(tree, source, _phase_names(5)[:-1])

    folder = tmp_path / source.name / SUBPATH
    written = sorted(p.name for p in folder.glob("*.npy"))
    assert written == [f"{i:05d}_flow.npy" for i in range(4)]
    assert len(OpticalFlowFolder(folder)) == 4


def test_a_written_flow_reads_back_as_a_flow_folder(tmp_path):
    source = _Sequence("plate/TL_00")
    _write(_tree(tmp_path, source.name, 3), source, _phase_names(3)[:-1])

    folder = OpticalFlowFolder(tmp_path / source.name / SUBPATH, validate="data")
    assert folder.frame_shape == (8, 8)
    assert folder.load_file(folder.get_file(0)).shape == (2, 8, 8)


def test_a_non_finite_flow_is_refused_rather_than_written(tmp_path):
    # What is written here is what a later run reads, so one that got through
    # could not be traced back to the run that made it.
    source = _Sequence("plate/TL_00")
    tree = _tree(tmp_path, source.name, 3)

    broken = _flow()
    broken[0, 0, 0] = float("nan")

    with tree:
        writer = tree.get_hook(source)
        assert writer is not None
        with pytest.raises(ValueError, match="finite"), writer:
            writer(Step(0, broken, Path("00000_phase.bin")))


def test_a_folder_this_tree_wrote_is_reused_rather_than_written_again(tmp_path):
    # The record holds one name per flow, which is one fewer than the source
    # holds. Comparing against the source's own frames would find a name too
    # many and refuse every folder the tree ever wrote.
    source = _Sequence("plate/TL_00")
    settings = {"estimator": "farneback"}

    first = _tree(tmp_path, source.name, 5, settings=settings)
    _write(first, source, _phase_names(5)[:-1])

    again = _tree(tmp_path, source.name, 5, settings=settings, if_present="reuse")
    with again:
        assert again.get_hook(source) is None
        assert again.report() == "kept 1 sequence already written"


def test_the_record_names_the_frame_each_flow_starts_from(tmp_path):
    # Start labelling: `flow[i]` runs from frame `i`, so the one left out is
    # the last rather than the first.
    source = _Sequence("plate/TL_00")
    tree = _tree(tmp_path, source.name, 4, settings={"estimator": "farneback"})

    _write(tree, source, _phase_names(4)[:-1])

    read = (tmp_path / source.name / SUBPATH / RECORD_FILE).read_text("utf-8")
    assert json.loads(read)["frames"] == list(_phase_names(4)[:-1])


def test_a_folder_written_from_another_source_is_not_reused(tmp_path):
    # The tree was told the source holds six frames, so it owes five flows and
    # the four already there describe a sequence that has since grown.
    source = _Sequence("plate/TL_00")
    settings = {"estimator": "farneback"}

    _write(
        _tree(tmp_path, source.name, 5, settings=settings), source, _phase_names(5)[:-1]
    )

    grown = _tree(tmp_path, source.name, 6, settings=settings, if_present="reuse")
    with grown:
        assert grown.get_hook(source) is not None
