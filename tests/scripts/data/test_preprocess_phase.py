from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from iivs.dhm.data.koala import PHASE_FLOAT_BIN
from iivs.dhm.data.phase import PhaseBinFolder

from scripts._compute import ComputeConfig, run_all
from scripts.data.preprocess_phase import (
    CONFIG_NAME,
    CONFIG_PATH,
    SourceConfig,
    TargetConfig,
    build_phase_stages,
    search_sources,
)
from tests.scripts.conftest import FRAMES, SEQUENCES

if TYPE_CHECKING:
    from pathlib import Path


def _composed(*overrides: str):
    with initialize_config_dir(config_dir=CONFIG_PATH, version_base=None):
        return compose(config_name=CONFIG_NAME, overrides=list(overrides))


def _scan(phase_tree: Path, dest: Path, workers: int) -> None:
    source = SourceConfig(root=str(phase_tree))
    target = TargetConfig(root=str(dest), save_frames=True)
    compute = ComputeConfig(device="cpu", workers=workers, progress_bar=False)

    run_all(build_phase_stages(source, target), compute)


def _written(dest: Path) -> dict[str, list[float]]:
    """Every frame written under `dest`, keyed by the sequence it belongs to."""
    out: dict[str, list[float]] = {}
    for folder in sorted(dest.rglob(PHASE_FLOAT_BIN)):
        read = PhaseBinFolder(folder)
        key = folder.relative_to(dest).as_posix()
        out[key] = [float(np.asarray(read[index]).sum()) for index in range(len(read))]

    return out


def test_sources_are_found_under_the_root(phase_tree):
    # The one that fails silently: a search that finds nothing leaves every other
    # check green, since there is then no sequence to get anything wrong with.
    sources = search_sources(SourceConfig(root=str(phase_tree)))

    assert len(sources) == SEQUENCES


def test_a_root_holding_nothing_and_an_empty_selection_are_told_apart(
    phase_tree, tmp_path
):
    # The two are fixed differently, so they must not arrive as one message. A
    # root that does not exist at all fails earlier still, inside the search.
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(ValueError, match=r"no time-lapse holds"):
        search_sources(SourceConfig(root=str(empty)))

    with pytest.raises(ValueError, match=r"include/exclude left none"):
        search_sources(
            SourceConfig(
                root=str(phase_tree),
                exclude=[f"TL_{s:02d}" for s in range(SEQUENCES)],
            )
        )


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


@pytest.mark.parametrize("group", ("cpu", "cuda"))
def test_a_compute_group_holds_the_fields_its_path_uses(group):
    # `workers` drives the CPU path and `gpu_ids` the CUDA one, so each group
    # carries only its own -- and `plan_devices` now refuses the other's.
    own = {"cpu": "workers", "cuda": "gpu_ids"}
    composed = set(_composed(f"compute={group}")["compute"])

    shared = {field.name for field in fields(ComputeConfig)} - set(own.values())
    assert shared <= composed
    assert own[group] in composed
    assert own["cuda" if group == "cpu" else "cpu"] not in composed


def test_a_key_no_schema_declares_is_refused():
    # Guards the test above: it only means something while composition is strict.
    with pytest.raises(Exception, match=r"(?i)could not override|not in struct"):
        _composed("compute.no_such_field=1")


def test_a_factory_offers_one_stage_per_sequence(phase_tree):
    stages = build_phase_stages(SourceConfig(root=str(phase_tree)))

    assert len(stages) == SEQUENCES


def test_a_run_without_a_target_writes_nothing(phase_tree, tmp_path):
    # `target_config=None` is how a run that only reads -- a cached source, or a
    # timing pass -- says it wants no side branch.
    dest = tmp_path / "out"
    compute = ComputeConfig(device="cpu", workers=0, progress_bar=False)

    run_all(build_phase_stages(SourceConfig(root=str(phase_tree))), compute)

    assert not dest.exists()


def test_the_pool_writes_what_the_lone_path_does(phase_tree, tmp_path):
    # Covers the whole worker path in one assertion. `mpire` hands its workers a
    # positional signature that no type checker sees, so the pool can go wrong
    # while every other check stays green.
    _scan(phase_tree, tmp_path / "lone", workers=0)
    _scan(phase_tree, tmp_path / "pooled", workers=2)

    lone = _written(tmp_path / "lone")
    assert len(lone) == SEQUENCES
    assert all(len(frames) == FRAMES for frames in lone.values())
    assert _written(tmp_path / "pooled") == lone


def test_the_written_tree_mirrors_the_source(phase_tree, tmp_path):
    # A destination crosses to the workers as a recipe, so the folders have to
    # land under its root -- with nothing coming back to say they did.
    dest = tmp_path / "out"

    _scan(phase_tree, dest, workers=2)

    for sequence in range(SEQUENCES):
        written = PhaseBinFolder(dest / f"TL_{sequence:02d}" / PHASE_FLOAT_BIN)
        assert len(written) == FRAMES
