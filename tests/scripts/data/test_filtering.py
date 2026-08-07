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
            overrides=[f"filter={name}"],
        )

    return composed.filter


@pytest.mark.parametrize("node", (None, {}))
def test_a_run_that_filters_nothing_still_names_a_kernel(node):
    # The absence arrives as `null`, a missing key, or an empty node, and a
    # record saying any of those leaves a reader guessing what actually ran.
    assert isinstance(parse_filter_config(node).build(), IdentityKernel)
    assert describe_filter_kernel(parse_filter_config(node)) == {"kind": "identity"}


def test_a_name_where_a_kernel_goes_says_it_has_to_select():
    # No override reaches this any more: the group is named for the key it
    # fills, so `filter=identity` selects from it exactly as `compute=cuda`
    # does, and a name hydra cannot find is refused by hydra. It stands for the
    # entry config that declares `filter` as a plain key without the group,
    # where the same override would put the string back.
    with pytest.raises(TypeError, match=r"has to select from `filter`"):
        parse_filter_config("identity")


def test_a_node_that_names_no_kernel_points_at_the_group():
    # Reached by dropping `_target_`, or by adding a node by hand: `instantiate`
    # then hands back a plain mapping, and the failure surfaced later still, as
    # a missing attribute on a dict. The group rather than the kernel kinds,
    # since a kind is not what anyone types -- `gaussian` has no option to pick
    # at all, and the default `median_ellipsoid_2x2x2` is not a kind.
    with pytest.raises(TypeError, match=r"select from `filter`"):
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
