from __future__ import annotations

from contextlib import contextmanager
from dataclasses import fields
from typing import TYPE_CHECKING

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from hydra.core.singleton import Singleton
from omegaconf import OmegaConf

from scripts._common.compute import ComputeConfig
from scripts._common.dataset import SequenceSelectConfig, SourceConfig
from scripts.data._process import TargetConfig
from scripts.data.preprocess import CONFIG_NAME, CONFIG_PATH, main

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from omegaconf import DictConfig


def _composed(*overrides: str):
    with initialize_config_dir(config_dir=CONFIG_PATH, version_base=None):
        return compose(config_name=CONFIG_NAME, overrides=list(overrides))


@pytest.mark.parametrize(
    ("group", "schema"),
    (
        ("source", SourceConfig),
        ("select", SequenceSelectConfig),
        ("target", TargetConfig),
    ),
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
    # One knob whose shape follows the device, a count on cpu and gpu ids on
    # cuda, so the group has to carry the shape `plan_devices` will accept.
    composed = _composed(f"compute={group}")["compute"]

    missing = {field.name for field in fields(ComputeConfig)} - set(composed)
    assert not missing
    assert composed["workers"] == workers


def test_a_key_no_schema_declares_is_refused():
    # Guards the test above: it only means something while composition is strict.
    with pytest.raises(Exception, match=r"(?i)could not override|not in struct"):
        _composed("compute.no_such_field=1")


# ------------------------------- the job itself --------------------------- #


@contextmanager
def _job(output_dir: Path, *overrides: str) -> Iterator[DictConfig]:
    """Compose and install this job's hydra node, the way `hydra.main` does.

    Composing does not fill `runtime.output_dir`, nor make the directory. The
    launcher does both, once it knows where the job goes, so this does, and
    the singleton is put back afterwards since a job left in it would answer for
    whatever ran next.
    """
    before = Singleton.get_state()
    output_dir.mkdir(parents=True, exist_ok=True)

    with initialize_config_dir(config_dir=CONFIG_PATH, version_base=None):
        composed = compose(
            config_name=CONFIG_NAME,
            overrides=["compute.show_progress=false", *overrides],
            return_hydra_config=True,
        )

    OmegaConf.update(composed, "hydra.runtime.output_dir", str(output_dir))
    HydraConfig.instance().set_config(composed)

    try:
        yield composed
    finally:
        Singleton.set_state(before)


def test_a_job_writes_where_hydra_put_it_rather_than_where_target_names(
    phase_tree, tmp_path
):
    # `main` was never run by the suite, only the config it composes, and it is
    # what wires the whole of stage (1) together: the three schemas, the sweep
    # guard, the log folder, and the run. Nothing reads `target.root` to write
    # with, so a job pointed elsewhere still lands in hydra's own directory.
    named = tmp_path / "named_but_unused"
    landing = tmp_path / "job"

    with _job(landing, f"source.root={phase_tree}", f"target.root={named}") as composed:
        main.__wrapped__(composed)

    assert (landing / "value_range.json").exists()
    assert not named.exists()


def test_a_sweep_is_refused_before_it_writes_the_one_tree(phase_tree, tmp_path):
    # Frames go to the job directory, and a sweep gives each job one of its own,
    # so every job would write a tree of its own and the last would be all that
    # was left: 1.45 TB apiece, in turn, for one of them.
    with (
        _job(
            tmp_path,
            f"source.root={phase_tree}",
            f"target.root={tmp_path}",
            "target.save_frames=true",
            "hydra.mode=MULTIRUN",
        ) as composed,
        pytest.raises(ValueError, match=r"cannot write frames in a sweep"),
    ):
        main.__wrapped__(composed)

    assert not (tmp_path / "TL_00").exists()
