from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import OmegaConf

from scripts._common.hydra import ensure_sweep_runs
from scripts.data._filtering import parse_filter_config
from scripts.data.preprocess import CONFIG_NAME, CONFIG_PATH

EXPERIMENTS = tuple(
    sorted(p.stem for p in Path(CONFIG_PATH, "experiment").glob("*.yaml"))
)


def _composed(name: str):
    with initialize_config_dir(config_dir=CONFIG_PATH, version_base=None):
        return compose(
            config_name=CONFIG_NAME,
            overrides=[f"+experiment={name}", "run_root=/runs", "source.root=/data"],
            return_hydra_config=True,
        )


def _filter_options(name: str) -> list[str]:
    params = _composed(name).hydra.sweeper.params
    return [option.strip() for option in str(params.filter).split(",")]


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_an_experiment_names_only_options_the_group_holds(name):
    # The sweeper turns each of these into a `filter=` override, so a name the
    # group does not hold fails partway through a sweep, on whichever job
    # carries it, after the ones before it have already run.
    held = {p.stem for p in Path(CONFIG_PATH, "filter").glob("*.yaml")}

    assert set(_filter_options(name)) <= held


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_an_experiment_sweeps_more_than_one_job(name):
    # A YAML list here composes without complaint and sweeps its last entry
    # alone, which is a sweep of one that looks like a sweep of many.
    assert len(_filter_options(name)) > 1


def test_the_experiment_named_for_the_group_holds_every_option():
    held = {p.stem for p in Path(CONFIG_PATH, "filter").glob("*.yaml")}

    assert set(_filter_options("filters")) == held


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_an_experiment_leaves_the_rest_of_the_config_alone(name):
    # An overlay, not a replacement: it names jobs and touches nothing a job
    # reads, so the defaults still stand under it.
    config = _composed(name)

    assert config.compute.device == "cpu"
    assert parse_filter_config(config.filter).build().radius == (2, 2, 2)
    assert config.target.ranges.save is True


def _entered(name: str | None, mode: RunMode):
    """Put a composed config in place as a run of `mode` would leave it."""
    overrides = ["run_root=/runs", "source.root=/data"]
    if name is not None:
        overrides.insert(0, f"+experiment={name}")

    with initialize_config_dir(config_dir=CONFIG_PATH, version_base=None):
        config = compose(
            config_name=CONFIG_NAME, overrides=overrides, return_hydra_config=True
        )

    config.hydra.mode = mode
    HydraConfig.instance().set_config(config)


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_an_experiment_without_multirun_is_refused(name):
    # The jobs are named and nobody runs them: the run takes the defaults,
    # sweeps nothing, and reports success, which is a whole run to notice.
    _entered(name, RunMode.RUN)

    with pytest.raises(ValueError, match=r"sweeps filter.*only --multirun runs"):
        ensure_sweep_runs()


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_an_experiment_under_multirun_passes(name):
    _entered(name, RunMode.MULTIRUN)

    ensure_sweep_runs()


@pytest.mark.parametrize("mode", (RunMode.RUN, RunMode.MULTIRUN))
def test_a_run_with_no_experiment_passes(mode):
    # A `--multirun` over something else is not an experiment, and a lone run
    # of the defaults is the ordinary case.
    _entered(None, mode)

    ensure_sweep_runs()


def test_every_experiment_declares_the_global_package():
    # Without the directive the file lands under `experiment:`, where the
    # sweeper never looks and nothing says so.
    for name in EXPERIMENTS:
        text = Path(CONFIG_PATH, "experiment", f"{name}.yaml").read_text("utf-8")
        assert text.startswith("# @package _global_"), name


def test_an_experiment_is_plain_python_once_composed():
    # As the filter group is: `to_object` is what the schemas are read through.
    params = _composed("filters").hydra.sweeper.params

    assert isinstance(OmegaConf.to_object(params), dict)
