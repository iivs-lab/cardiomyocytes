from __future__ import annotations

import logging
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import InstantiationException
from omegaconf import OmegaConf

from iivs_cardio.optical_flow.estimators import DeepFlowConfig, FarnebackConfig
from scripts.optical_flow._estimating import (
    describe_estimator_config,
    log_estimator_config,
    parse_estimator_config,
)
from scripts.optical_flow.estimate import CONFIG_NAME, CONFIG_PATH

FARNEBACK = "iivs_cardio.optical_flow.estimators.opencv.farneback.FarnebackConfig"


def _estimator_options() -> tuple[str, ...]:
    return tuple(sorted(p.stem for p in Path(CONFIG_PATH, "estimator").glob("*.yaml")))


def _node(target: str, **settings: object):
    return OmegaConf.create({"_target_": target, **settings})


def test_a_node_becomes_the_estimator_it_names():
    config = parse_estimator_config(_node(FARNEBACK, num_levels=1))

    assert isinstance(config, FarnebackConfig)
    assert config.num_levels == 1


def test_an_absent_estimator_is_refused_rather_than_defaulted():
    # Unlike the filter there is no estimator that does nothing, so a run with
    # none has no way to produce a flow at all.
    with pytest.raises(TypeError, match="`estimator` is not set"):
        parse_estimator_config(None)


def test_a_group_name_written_as_the_value_is_refused():
    # What `estimator=farneback` looks like when the group prefix was left off,
    # which otherwise reaches instantiate as an unreadable target.
    with pytest.raises(TypeError, match="holds the name 'farneback'"):
        parse_estimator_config("farneback")  # type: ignore[invalid-argument-type]


def test_a_node_that_names_no_estimator_points_at_the_group():
    # Reached by dropping `_target_`, or by adding a node by hand: `instantiate`
    # then hands back a plain mapping, and the failure would surface later as a
    # missing attribute on a dict.
    with pytest.raises(TypeError, match=r"select from `estimator`"):
        parse_estimator_config(OmegaConf.create({"num_levels": 3}))


def test_a_target_outside_the_estimators_is_refused_by_the_whitelist():
    # A config file names what to import, so the whitelist is what keeps a
    # document from running arbitrary code.
    node = _node("iivs_cardio.data.transforms.filtering.kernel.identity.IdentityConfig")

    with pytest.raises(InstantiationException, match="whitelist"):
        parse_estimator_config(node)


@pytest.mark.parametrize("name", _estimator_options())
def test_every_option_of_the_group_composes_into_an_estimator_config(name):
    with initialize_config_dir(config_dir=CONFIG_PATH, version_base=None):
        composed = compose(
            config_name=CONFIG_NAME,
            overrides=[f"estimator={name}", "run_root=.", "source.root=."],
        )

    config = parse_estimator_config(composed.estimator)

    assert describe_estimator_config(config)["kind"] == name


def test_a_description_names_which_estimator_and_what_shapes_it():
    described = describe_estimator_config(FarnebackConfig(num_levels=2, win_size=7))

    assert described["kind"] == "farneback"
    assert described["num_levels"] == 2
    assert described["win_size"] == 7


def test_two_estimators_are_told_apart_by_what_is_recorded():
    # What a later run compares to decide whether a document still describes
    # it: two runs under different estimators must not read as one.
    assert describe_estimator_config(FarnebackConfig()) != describe_estimator_config(
        DeepFlowConfig()
    )


def test_a_description_is_a_fresh_mapping_each_time():
    config = FarnebackConfig()
    described = describe_estimator_config(config)
    described["kind"] = "edited"

    assert describe_estimator_config(config)["kind"] == "farneback"


def test_the_log_line_names_the_estimator_and_its_settings(caplog):
    logger = logging.getLogger("test_estimating")

    with caplog.at_level(logging.INFO, logger="test_estimating"):
        log_estimator_config(FarnebackConfig(num_levels=2), logger)

    (line,) = caplog.messages

    assert "estimator: farneback" in line
    assert "num_levels=2" in line
