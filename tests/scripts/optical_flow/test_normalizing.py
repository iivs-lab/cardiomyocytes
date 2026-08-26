from __future__ import annotations

import json
import logging

import pytest
import torch

from scripts.optical_flow._normalizing import (
    NormalizeConfig,
    build_normalization,
    log_normalize_config,
)
from tests.scripts.optical_flow.helpers import range_document

SPANS = {"TL_00": (0.0, 4.0), "TL_01": (-1.0, 1.0)}


def _document(tmp_path, spans=None):
    return str(range_document(tmp_path / "value_range", spans or SPANS))


def test_a_given_span_covers_every_sequence_the_source_holds(tmp_path):
    config = NormalizeConfig(level="given", source=[0.0, 2.0])

    built = build_normalization(config, torch.uint8)
    normalizers = built.normalizers(["a", "b", "c"])

    assert built.shared is not None
    assert set(normalizers) == {"a", "b", "c"}
    assert normalizers["a"].source == (0.0, 2.0)
    assert normalizers["a"].target == (0.0, 255.0)


def test_the_dataset_level_scales_every_sequence_by_the_widest_range(tmp_path):
    # DatasetResult across the whole dataset, so the sequence that reached the extreme
    # sets the constants for all of them and two sequences stay comparable.
    built = build_normalization(
        NormalizeConfig(range_file=_document(tmp_path)), torch.uint8
    )

    normalizers = built.normalizers(SPANS)

    assert built.shared is not None
    assert normalizers["TL_00"].source == (-1.0, 4.0)
    assert normalizers["TL_01"].source == (-1.0, 4.0)


def test_the_sequence_level_gives_each_sequence_its_own(tmp_path):
    config = NormalizeConfig(level="sequence", range_file=_document(tmp_path))

    built = build_normalization(config, torch.uint8)
    normalizers = built.normalizers(SPANS)

    assert built.shared is None
    assert normalizers["TL_00"].source == (0.0, 4.0)
    assert normalizers["TL_01"].source == (-1.0, 1.0)


def test_a_sequence_the_document_never_covered_gets_no_scaling(tmp_path):
    # Left out rather than refused here: which sequences the run was given is
    # settled later, and the job refuses the ones it was given by name.
    config = NormalizeConfig(level="sequence", range_file=_document(tmp_path))

    built = build_normalization(config, torch.uint8)

    assert "TL_99" not in built.normalizers([*SPANS, "TL_99"])


def test_the_target_follows_the_dtype_the_estimator_reads(tmp_path):
    config = NormalizeConfig(level="given", source=[0.0, 1.0])

    assert build_normalization(config, torch.uint8).shared.target == (0.0, 255.0)
    assert build_normalization(config, torch.float32).shared.target == (0.0, 1.0)


def test_a_target_that_was_asked_for_beats_the_dtype(tmp_path):
    config = NormalizeConfig(level="given", source=[0.0, 1.0], target=[0.0, 100.0])

    assert build_normalization(config, torch.uint8).shared.target == (0.0, 100.0)


# ========================== #
#          Recording         #
# ========================== #


def test_what_is_recorded_is_the_range_and_not_the_document_it_came_from(tmp_path):
    # The same path may hold a document that was written again, so recording
    # the path would let a run that scaled by other numbers read as this one.
    built = build_normalization(
        NormalizeConfig(range_file=_document(tmp_path)), torch.uint8
    )

    assert built.described == {
        "level": "dataset",
        "range": [-1.0, 4.0],
        "target": [0.0, 255.0],
    }


def test_the_sequence_level_records_every_range_it_used(tmp_path):
    # One value per sequence, so two runs off different documents are told
    # apart even though the level and the file are the same.
    config = NormalizeConfig(level="sequence", range_file=_document(tmp_path))

    built = build_normalization(config, torch.uint8)

    assert built.described["ranges"] == {
        "TL_00": [0.0, 4.0],
        "TL_01": [-1.0, 1.0],
    }


def test_what_is_recorded_survives_the_serializer_a_document_is_written_with(tmp_path):
    config = NormalizeConfig(level="sequence", range_file=_document(tmp_path))

    described = build_normalization(config, torch.uint8).described

    assert json.loads(json.dumps(described, allow_nan=False)) == described


# ========================== #
#          Refusals          #
# ========================== #


def test_a_measured_level_with_no_document_names_the_setting_to_fill():
    with pytest.raises(ValueError, match=r"set `normalize\.range_file`"):
        build_normalization(NormalizeConfig(), torch.uint8)


def test_a_document_that_is_not_there_names_the_setting_that_points_at_it(tmp_path):
    config = NormalizeConfig(range_file=str(tmp_path / "nowhere"))

    with pytest.raises(ValueError, match=r"no such `normalize\.range_file`"):
        build_normalization(config, torch.uint8)


def test_a_file_that_is_not_a_range_document_is_refused(tmp_path):
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"coverage": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="holds no dataset range"):
        build_normalization(NormalizeConfig(range_file=str(path)), torch.uint8)


def test_the_given_level_needs_the_span_it_scales_from():
    with pytest.raises(ValueError, match="'given' scales from `source`"):
        build_normalization(NormalizeConfig(level="given"), torch.uint8)


def test_a_measured_level_refuses_a_span_of_its_own(tmp_path):
    config = NormalizeConfig(range_file=_document(tmp_path), source=[0.0, 1.0])

    with pytest.raises(ValueError, match="'dataset' is measured"):
        build_normalization(config, torch.uint8)


def test_a_span_that_is_not_two_numbers_is_refused_by_name():
    config = NormalizeConfig(level="given", source=[0.0, 1.0, 2.0])

    with pytest.raises(ValueError, match=r"`normalize.source` takes two numbers"):
        build_normalization(config, torch.uint8)


# ========================== #
#          Logging           #
# ========================== #


def test_the_log_line_names_the_level_and_the_span(tmp_path, caplog):
    logger = logging.getLogger("test_normalizing")
    built = build_normalization(
        NormalizeConfig(range_file=_document(tmp_path)), torch.uint8
    )

    with caplog.at_level(logging.INFO, logger="test_normalizing"):
        log_normalize_config(built, logger)

    said = "\n".join(caplog.messages)

    assert "by the dataset range" in said
    assert "scaling from [-1, 4]" in said
    assert "onto [0, 255]" in said


def test_a_per_sequence_run_says_so_rather_than_naming_one_span(tmp_path, caplog):
    logger = logging.getLogger("test_normalizing")
    config = NormalizeConfig(level="sequence", range_file=_document(tmp_path))
    built = build_normalization(config, torch.uint8)

    with caplog.at_level(logging.INFO, logger="test_normalizing"):
        log_normalize_config(built, logger)

    said = "\n".join(caplog.messages)

    assert "scaling each of 2 sequences from its own" in said
