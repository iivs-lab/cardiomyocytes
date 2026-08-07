from __future__ import annotations

from dataclasses import fields

import pytest
from hydra import compose, initialize_config_dir

from scripts._compute import ComputeConfig
from scripts.data._process import SourceConfig, TargetConfig
from scripts.data.preprocess import CONFIG_NAME, CONFIG_PATH


def _composed(*overrides: str):
    with initialize_config_dir(config_dir=CONFIG_PATH, version_base=None):
        return compose(config_name=CONFIG_NAME, overrides=list(overrides))


@pytest.mark.parametrize(
    ("group", "schema"), (("source", SourceConfig), ("target", TargetConfig))
)
def test_every_schema_field_is_reachable_from_the_command_line(group, schema):
    # hydra only lets an override touch a key the composed config already holds, so
    # a field added to the schema alone is settable in code and refused on the CLI.
    # Derived from the schema, so a new field has to reach the YAML as well.
    composed = _composed()

    missing = {field.name for field in fields(schema)} - set(composed[group])
    assert not missing


@pytest.mark.parametrize(("group", "workers"), (("cpu", 0), ("cuda", [0])))
def test_a_compute_group_reaches_every_field_and_shapes_workers_for_its_device(
    group, workers
):
    # One knob whose shape follows the device -- a count on cpu, gpu ids on cuda
    # -- so the group has to carry the shape `plan_devices` will accept.
    composed = _composed(f"compute={group}")["compute"]

    missing = {field.name for field in fields(ComputeConfig)} - set(composed)
    assert not missing
    assert composed["workers"] == workers


def test_a_key_no_schema_declares_is_refused():
    # Guards the test above: it only means something while composition is strict.
    with pytest.raises(Exception, match=r"(?i)could not override|not in struct"):
        _composed("compute.no_such_field=1")
