from __future__ import annotations

from contextlib import contextmanager
from dataclasses import fields
from typing import TYPE_CHECKING

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from hydra.core.singleton import Singleton
from omegaconf import OmegaConf

from iivs_cardio.data.transforms.filtering.kernel import IdentityConfig
from iivs_cardio.optical_flow.estimators import DeepFlowConfig
from scripts._common.dataset import SequenceSelectConfig
from scripts._common.phase import LAST_SEARCH
from scripts.optical_flow._normalizing import NormalizeConfig
from scripts.optical_flow._process import (
    FlowInputs,
    FlowSourceConfig,
    FlowTargetConfig,
)
from scripts.optical_flow.estimate import CONFIG_NAME, CONFIG_PATH, main
from tests.scripts.optical_flow.helpers import SEQUENCES, phase_tree, range_document

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from omegaconf import DictConfig

SPANS = dict.fromkeys((f"TL_{index:02d}" for index in range(SEQUENCES)), (0.0, 4.0))


@pytest.fixture(autouse=True)
def _forget_the_last_search() -> None:
    LAST_SEARCH.clear()


def _composed(*overrides: str):
    with initialize_config_dir(config_dir=CONFIG_PATH, version_base=None):
        return compose(config_name=CONFIG_NAME, overrides=list(overrides))


@pytest.mark.parametrize(
    ("group", "schema"),
    (
        ("source", FlowSourceConfig),
        ("select", SequenceSelectConfig),
        ("normalize", NormalizeConfig),
        ("target", FlowTargetConfig),
    ),
)
def test_every_schema_field_is_reachable_from_the_command_line(group, schema):
    # hydra only lets an override touch a key the composed config already holds,
    # so a field added to the schema alone is settable in code and refused on
    # the CLI. Derived from the schema, so a new field has to reach the YAML.
    composed = _composed()

    missing = {field.name for field in fields(schema)} - set(composed[group])
    assert not missing


def test_a_key_no_schema_declares_is_refused():
    # Guards the test above: it only means something while composition is strict.
    with pytest.raises(Exception, match=r"(?i)could not override|not in struct"):
        _composed("normalize.no_such_field=1")


def test_the_stage_reads_phase_and_writes_flows_by_default():
    # The two ends of the stage, which the defaults have to name for a run that
    # overrides only the roots to work at all.
    composed = _composed()

    assert composed.source.subpath is None
    assert composed.target.evaluations.save
    assert not composed.target.flows.save


# ------------------------------- the job itself --------------------------- #


@contextmanager
def _job(output_dir: Path, *overrides: str) -> Iterator[DictConfig]:
    """Compose and install this job's hydra node, the way `hydra.main` does."""
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


def _tree(tmp_path: Path) -> tuple[Path, Path]:
    return phase_tree(tmp_path / "src"), range_document(tmp_path / "value_range", SPANS)


def test_a_job_writes_where_hydra_put_it_rather_than_where_run_root_names(tmp_path):
    # `main` is what wires the whole of stage (2) together: the five schemas,
    # the sweep guard, the log folder, and the run. Nothing reads `run_root` to
    # write with, so a job pointed elsewhere still lands in hydra's directory.
    tree, ranges = _tree(tmp_path)
    named = tmp_path / "named_but_unused"
    landing = tmp_path / "job"

    with _job(
        landing,
        f"source.root={tree}",
        f"normalize.range_file={ranges}",
        f"run_root={named}",
    ) as composed:
        main.__wrapped__(composed)

    assert (landing / "flow_evaluation.json").exists()
    assert not named.exists()


def test_a_sweep_is_refused_before_it_writes_the_one_tree(tmp_path):
    # Flows go to the job directory, and a sweep gives each job one of its own,
    # so every job would write a tree of its own and the last would be all that
    # was left.
    tree, ranges = _tree(tmp_path)

    with (
        _job(
            tmp_path / "job",
            f"source.root={tree}",
            f"normalize.range_file={ranges}",
            f"run_root={tmp_path}",
            "target.flows.save=true",
            "hydra.mode=MULTIRUN",
        ) as composed,
        pytest.raises(ValueError, match=r"cannot write flows in a sweep"),
    ):
        main.__wrapped__(composed)

    assert not (tmp_path / "job" / "TL_00").exists()


def test_a_composed_job_is_read_into_one_value():
    composed = _composed(
        "source.root=/data",
        "estimator=deepflow",
        "normalize.level=given",
        "normalize.source=[0,1]",
        "select.exclude=[TL_00,TL_01]",
    )

    inputs = FlowInputs.read(composed)

    assert inputs.source.root == "/data"
    assert isinstance(inputs.estimator, DeepFlowConfig)
    assert isinstance(inputs.kernel, IdentityConfig)
    assert inputs.normalize.level == "given"
    assert inputs.normalize.source == [0.0, 1.0]
    assert inputs.target.evaluations.save


def test_what_was_read_holds_plain_python_rather_than_configuration():
    # Nothing past `main` should have to know where its settings came from, so
    # a container reaching the run would carry hydra's own rules into it.
    inputs = FlowInputs.read(_composed("source.root=/data", "select.exclude=[TL_00]"))

    assert type(inputs.select.exclude) is list
    assert type(inputs.source.frames.start) is int


def test_a_job_with_no_estimator_is_refused_where_it_is_read():
    with pytest.raises(TypeError, match="`estimator` is not set"):
        FlowInputs.read(_composed("source.root=/data", "~estimator"))
