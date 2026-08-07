from __future__ import annotations

from dataclasses import dataclass

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from hydra.core.singleton import Singleton
from omegaconf import DictConfig, OmegaConf, ValidationError

from scripts._hydra import apply_schema, is_multirun, output_directory
from scripts.data.preprocess import CONFIG_NAME, CONFIG_PATH


@dataclass
class _Settings:
    root: str = "???"
    workers: int = 1
    names: list[str] | None = None


def _hydra_node(output_dir: str, *overrides: str) -> DictConfig:
    """Compose the real config with hydra's own node, as a job would see it.

    `runtime.output_dir` is filled in here because composing does not set it:
    the launcher does, once it knows where the job goes.
    """
    with initialize_config_dir(config_dir=CONFIG_PATH, version_base=None):
        composed = compose(
            config_name=CONFIG_NAME,
            overrides=["source.root=/dataset", "target.root=/out", *overrides],
            return_hydra_config=True,
        )

    OmegaConf.update(composed, "hydra.runtime.output_dir", output_dir)

    return composed


@pytest.fixture()
def job(request, tmp_path):
    """Put this job's hydra node where the runtime reads it, as `hydra.main` does.

    Through the singleton's own state, since that is the only handle on it and
    a job left set would answer for whatever ran next.
    """
    before = Singleton.get_state()
    composed = _hydra_node(str(tmp_path), *getattr(request, "param", ()))
    HydraConfig.instance().set_config(composed)

    yield composed

    Singleton.set_state(before)


def test_a_node_comes_back_as_the_schema_holding_plain_values():
    # The one place a composed node becomes a dataclass. What it hands back has
    # to be free of configuration containers, since the rest of the code passes
    # it to workers and writes it into records without knowing where it came
    # from -- a `ListConfig` reaching either would be a different value there.
    node = OmegaConf.create({"root": "/dataset", "workers": 3, "names": ["a", "b"]})

    settings = apply_schema(_Settings, node)

    assert isinstance(settings, _Settings)
    assert settings.root == "/dataset"
    assert settings.workers == 3
    assert settings.names == ["a", "b"]
    assert type(settings.names) is list  # not a `ListConfig` wearing the same face


def test_a_value_that_does_not_fit_its_field_is_refused():
    # The check the schema is applied for: hydra takes any override the key
    # exists for, so the type is only tested here.
    node = OmegaConf.create({"root": "/dataset", "workers": "three"})

    with pytest.raises(ValidationError):
        apply_schema(_Settings, node)


def test_a_field_the_node_leaves_out_keeps_its_default():
    settings = apply_schema(_Settings, OmegaConf.create({"root": "/dataset"}))

    assert settings.workers == 1
    assert settings.names is None


def test_the_output_directory_is_the_one_hydra_made_for_this_job(job):
    # Everything a run writes goes here, and `target.root` only places it -- so
    # this is the value the outputs actually follow.
    assert output_directory() == job.hydra.runtime.output_dir


def test_a_lone_run_is_not_a_sweep(job):
    assert not is_multirun()


@pytest.mark.parametrize("job", (["hydra.mode=MULTIRUN"],), indirect=True)
def test_a_sweep_says_so(job):
    # `hydra.mode` is what the flag sets, and a step writing somewhere every job
    # shares has to refuse the sweep rather than let the jobs race for it.
    assert is_multirun()
