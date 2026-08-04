from __future__ import annotations

from dataclasses import fields

import pytest
from hydra import compose, initialize_config_dir
from iivs.dhm.data.koala import PHASE_FLOAT_BIN
from iivs.dhm.data.phase import PhaseBinFolder

from scripts._compute import ComputeConfig
from scripts._range import DatasetRangeCollector, as_dict
from scripts.data.scan_phase import (
    CONFIG_NAME,
    CONFIG_PATH,
    DatasetFieldWriter,
    SourceConfig,
    TargetConfig,
    build_sequences,
    scan_sequences,
    search_sources,
)
from tests.scripts.conftest import SEQUENCES


def _composed(*overrides: str):
    with initialize_config_dir(config_dir=CONFIG_PATH, version_base=None):
        return compose(config_name=CONFIG_NAME, overrides=list(overrides))


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
    # carries only its own. Everything else applies whichever device is chosen,
    # and so has to appear in both.
    own = {"cpu": "workers", "cuda": "gpu_ids"}
    composed = set(_composed(f"compute={group}")["compute"])

    shared = {field.name for field in fields(ComputeConfig)} - set(own.values())
    assert shared <= composed
    assert own[group] in composed


def test_a_key_no_schema_declares_is_refused():
    # Guards the test above: it only means something while composition is strict.
    with pytest.raises(Exception, match=r"(?i)could not override|not in struct"):
        _composed("compute.no_such_field=1")


def test_the_pool_returns_what_the_lone_path_does(phase_tree, tmp_path):
    # Covers the whole worker path in one assertion. `mpire` hands its workers a
    # positional signature that no type checker sees, so the pool can go wrong
    # while every other check stays green.
    source_config = SourceConfig(root=str(phase_tree))
    lone = ComputeConfig(device="cpu", workers=0, progress_bar=False)
    pooled = ComputeConfig(device="cpu", workers=2, progress_bar=False)

    expected, actual = DatasetRangeCollector(), DatasetRangeCollector()
    scan_sequences(build_sequences(lone, source_config), lone, source_config, expected)
    scan_sequences(
        build_sequences(pooled, source_config), pooled, source_config, actual
    )

    assert as_dict(actual.collected()) == as_dict(expected.collected())


def test_the_pool_writes_the_folders_a_container_points_it_at(phase_tree, tmp_path):
    # A writer container crosses to the workers as a recipe, so the folders have
    # to land under its root -- with nothing coming back to say they did.
    out = tmp_path / "out"
    source_config = SourceConfig(root=str(phase_tree))
    compute = ComputeConfig(device="cpu", workers=2, progress_bar=False)
    writers = DatasetFieldWriter(str(out), PHASE_FLOAT_BIN)

    ranges = DatasetRangeCollector()
    sequences = build_sequences(compute, source_config)
    scan_sequences(sequences, compute, source_config, ranges, writers)

    for sequence in ranges.collected().sequences:
        written = PhaseBinFolder(out / sequence.source / PHASE_FLOAT_BIN)
        assert len(written) == len(sequence.frames)
