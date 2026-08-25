from __future__ import annotations

import json
import logging

import pytest

from iivs_cardio.common.device import Device
from iivs_cardio.optical_flow.data import FLOW_FLOAT_NPY, OpticalFlowFolder
from iivs_cardio.optical_flow.estimators import FarnebackConfig
from scripts._common.dataset import FrameSelectConfig, SequenceSelectConfig
from scripts._common.phase import LAST_SEARCH
from scripts.optical_flow._normalizing import NormalizeConfig
from scripts.optical_flow._process import (
    FlowSourceConfig,
    FlowTargetConfig,
    build_flow_stages,
    log_target_config,
)
from tests.scripts.optical_flow.helpers import (
    FRAMES,
    SEQUENCES,
    phase_tree,
    range_document,
)

NAMES = tuple(f"TL_{index:02d}" for index in range(SEQUENCES))
SPANS = dict.fromkeys(NAMES, (0.0, 4.0))


@pytest.fixture(autouse=True)
def _forget_the_last_search() -> None:
    LAST_SEARCH.clear()


@pytest.fixture()
def tree(tmp_path):
    return phase_tree(tmp_path / "src")


def _configs(tree, tmp_path, **frames):
    source = FlowSourceConfig(root=str(tree), frames=FrameSelectConfig(**frames))
    normalize = NormalizeConfig(
        range_file=str(range_document(tmp_path / "value_range", SPANS))
    )

    return source, SequenceSelectConfig(), normalize


def _run(tree, tmp_path, target=None, **frames):
    """Assemble a job and walk every sequence through it, as a driver would."""
    source, select, normalize = _configs(tree, tmp_path, **frames)
    output = tmp_path / "out"
    output.mkdir(exist_ok=True)

    stages = build_flow_stages(
        source,
        select,
        FarnebackConfig(),
        normalize,
        None,
        target,
        output_root=output,
        name="optical_flow",
    )

    with stages.running():
        for index in range(len(stages)):
            stages.run_stage(index, Device("cpu"))

    return output


def _target(**flows) -> FlowTargetConfig:
    config = FlowTargetConfig()
    for key, value in flows.items():
        setattr(config.flows, key, value)

    return config


# ========================== #
#          Running           #
# ========================== #


def test_a_run_scores_every_sequence_into_one_document(tree, tmp_path):
    output = _run(tree, tmp_path, _target())

    written = json.loads((output / "flow_evaluation.json").read_text("utf-8"))

    assert written["coverage"]["covered"] == SEQUENCES
    assert written["dataset"]["pairs"] == SEQUENCES * (FRAMES - 1)
    assert written["dataset"]["metrics"]["ssim"]["scored"] > 0


def test_the_flows_are_written_where_koala_would_keep_them(tree, tmp_path):
    output = _run(tree, tmp_path, _target(save=True))

    folder = output / NAMES[0] / FLOW_FLOAT_NPY

    assert sorted(p.name for p in folder.glob("*.npy")) == [
        "00000_flow.npy",
        "00001_flow.npy",
        "00002_flow.npy",
    ]


def test_a_written_flow_reads_back_as_the_field_it_is(tree, tmp_path):
    output = _run(tree, tmp_path, _target(save=True))

    flows = OpticalFlowFolder(output / NAMES[0] / FLOW_FLOAT_NPY)

    assert len(flows) == FRAMES - 1
    assert flows[0].shape == (2, 48, 48)


def test_n_frames_answer_one_flow_fewer(tree, tmp_path):
    # The frame selection counts source frames, as it does in preprocessing, so
    # asking for three phase frames answers two flows.
    output = _run(tree, tmp_path, _target(save=True), count=3)

    written = json.loads((output / "flow_evaluation.json").read_text("utf-8"))

    assert written["dataset"]["pairs"] == SEQUENCES * 2


def test_what_the_run_scaled_by_goes_on_record(tree, tmp_path):
    # What a later run compares to decide whether it may reuse this one, and
    # the ranges rather than the document they were read from.
    output = _run(tree, tmp_path, _target(save=True))

    written = json.loads((output / "flow_evaluation.json").read_text("utf-8"))
    settings = written["settings"]

    assert settings["normalize"] == {
        "level": "dataset",
        "range": [0.0, 4.0],
        "target": [0.0, 255.0],
    }
    assert settings["estimator"]["kind"] == "farneback"
    assert settings["filter"]["kind"] == "identity"


def test_both_branches_record_the_same_settings(tree, tmp_path):
    # A flow folder read on its own has to say what made it, and it must not
    # disagree with the document that scored it.
    output = _run(tree, tmp_path, _target(save=True))

    document = json.loads((output / "flow_evaluation.json").read_text("utf-8"))
    record = output / NAMES[0] / FLOW_FLOAT_NPY / "source.json"
    folder = json.loads(record.read_text("utf-8"))

    assert folder["settings"] == document["settings"]


def test_a_second_run_reuses_what_the_first_left(tree, tmp_path):
    target = _target(save=True, if_present="reuse")
    target.evaluations.if_present = "reuse"

    _run(tree, tmp_path, target)
    output = _run(tree, tmp_path, target)

    written = json.loads((output / "flow_evaluation.json").read_text("utf-8"))

    assert written["coverage"]["reused"] == SEQUENCES
    assert written["coverage"]["covered"] == SEQUENCES


# ========================== #
#          Refusals          #
# ========================== #


def test_a_target_that_writes_nothing_is_refused(tree, tmp_path):
    target = FlowTargetConfig()
    target.evaluations.save = False

    with pytest.raises(ValueError, match="nothing to do"):
        _run(tree, tmp_path, target)


def test_flows_that_would_land_on_the_phase_they_read_are_refused(tree, tmp_path):
    source, select, normalize = _configs(tree, tmp_path)
    target = _target(save=True, subpath="Phase/Float/Bin")

    with pytest.raises(ValueError, match="flows would land on the source"):
        build_flow_stages(
            source,
            select,
            FarnebackConfig(),
            normalize,
            None,
            target,
            output_root=tree,
            name="optical_flow",
        )


def test_a_sequence_the_document_has_no_range_for_is_refused_by_name(tree, tmp_path):
    source = FlowSourceConfig(root=str(tree))
    partial = {NAMES[0]: (0.0, 4.0)}
    normalize = NormalizeConfig(
        level="sequence",
        range_file=str(range_document(tmp_path / "value_range", partial)),
    )

    with pytest.raises(ValueError, match=f"no normalizer for '{NAMES[1]}'"):
        build_flow_stages(
            source,
            SequenceSelectConfig(),
            FarnebackConfig(),
            normalize,
            None,
            _target(),
            output_root=tmp_path / "out",
            name="optical_flow",
        )


def test_a_sequence_too_short_to_make_a_pair_is_refused(tree, tmp_path):
    source, select, normalize = _configs(tree, tmp_path, count=1)

    with pytest.raises(ValueError, match="a flow needs two"):
        build_flow_stages(
            source,
            select,
            FarnebackConfig(),
            normalize,
            None,
            _target(),
            output_root=tmp_path / "out",
            name="optical_flow",
        )


# ========================== #
#          Logging           #
# ========================== #


def test_the_run_says_what_it_was_asked_to_do(tree, tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="optical_flow"):
        _run(tree, tmp_path, _target(save=True))

    said = "\n".join(caplog.messages)

    assert f"source: {tree}" in said
    assert "normalize: by the dataset range" in said
    assert "estimator: farneback" in said
    assert f"writing the flows to <sequence>/{FLOW_FLOAT_NPY}" in said
    assert "writing the evaluations to flow_evaluation.json" in said


def test_a_document_name_that_is_not_one_names_the_setting_that_holds_it(
    tree, tmp_path
):
    # The library's own refusal says what is wrong with the name but not which
    # setting carries it, which is where a reader has to go.
    target = _target()
    target.evaluations.file = "flow_evaluation.txt"

    with pytest.raises(ValueError, match=r"`target\.evaluations\.file`"):
        _run(tree, tmp_path, target)


def test_a_run_that_writes_nothing_says_so_rather_than_naming_an_output(caplog):
    target = FlowTargetConfig()
    target.evaluations.save = False
    logger = logging.getLogger("test_process")

    with caplog.at_level(logging.INFO, logger="test_process"):
        log_target_config(target, logger, output_root="out")

    assert "writing nothing" in "\n".join(caplog.messages)


def test_the_policies_a_branch_was_given_are_said_under_its_output(
    tree, tmp_path, caplog
):
    target = _target(save=True, if_present="overwrite", if_unsourced="delete")

    with caplog.at_level(logging.INFO, logger="optical_flow"):
        _run(tree, tmp_path, target)

    said = "\n".join(caplog.messages)

    assert "overwriting the flows it finds" in said
    assert "dropping the flows a source no longer has" in said
