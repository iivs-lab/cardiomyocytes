from __future__ import annotations

import json

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from iivs_cardio.data.transforms.filtering.kernel import IdentityKernel, MedianKernel
from scripts.data._filtering import describe_filter_kernel, parse_filter_config
from scripts.data.preprocess import CONFIG_NAME, CONFIG_PATH


def _filter_node(name: str):
    with initialize_config_dir(config_dir=CONFIG_PATH, version_base=None):
        composed = compose(
            config_name=CONFIG_NAME,
            overrides=[f"data/transforms/filtering@filter={name}"],
        )

    return composed.filter


@pytest.mark.parametrize("node", (None, {}))
def test_a_run_that_filters_nothing_still_names_a_kernel(node):
    # The absence arrives as `null`, a missing key, or an empty node, and a
    # record saying any of those leaves a reader guessing what actually ran.
    assert isinstance(parse_filter_config(node).build(), IdentityKernel)
    assert describe_filter_kernel(parse_filter_config(node)) == {"kind": "identity"}


def test_a_filter_named_the_way_compute_is_says_the_form_it_wanted():
    # `compute=cpu` works because that group's path matches its key, and the
    # filter's does not -- so the shape a reader infers from the one they have
    # already typed leaves a bare name where a node belongs, and hydra puts the
    # string there. Inferring it is what makes this reachable, so the refusal
    # has to carry the form that does work.
    with pytest.raises(TypeError, match=r"use `data/transforms/filtering@filter="):
        parse_filter_config("identity")


def test_a_node_that_names_no_kernel_lists_the_ones_there_are():
    # Reached by dropping `_target_`, or by adding a node by hand: `instantiate`
    # then hands back a plain mapping, and the failure surfaced later still, as
    # a missing attribute on a dict. The list is each kind once -- every config
    # is a `slots=True` dataclass, so the class the decorator replaced is still
    # a subclass until the collector reaches it, and reading them as a list put
    # whichever had not been collected yet in the message twice.
    with pytest.raises(TypeError, match=r"pick one of gaussian, identity, median,"):
        parse_filter_config(OmegaConf.create({"radius": [1, 1, 1]}))


def test_a_composed_node_builds_the_kernel_it_names():
    kernel = parse_filter_config(_filter_node("median_ellipsoid_2x2x2")).build()

    assert isinstance(kernel, MedianKernel)
    assert kernel.radius == (2, 2, 2)
    assert kernel.shape == "ellipsoid"


def test_a_composed_config_holds_plain_python():
    # `instantiate` hands omegaconf's own containers through unless told not to,
    # and the record is taken for a plain one by `asdict` and every reader after.
    config = parse_filter_config(_filter_node("median_cuboid_3x3x1"))

    assert type(config.radius) is list  # ty: ignore[unresolved-attribute]
    assert config.radius == [3, 3, 1]  # ty: ignore[unresolved-attribute]


def test_a_description_survives_the_serializer_a_document_is_written_with():
    described = describe_filter_kernel(
        parse_filter_config(_filter_node("median_ellipsoid_2x2x2"))
    )

    assert described["kind"] == "median"
    assert json.loads(json.dumps(described)) == described
