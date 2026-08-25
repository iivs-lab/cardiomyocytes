from __future__ import annotations

__all__ = (
    "EvaluationBranchConfig",
    "FlowBranchConfig",
    "FlowSourceConfig",
    "FlowTargetConfig",
    "build_branches",
    "build_flow_stages",
    "log_configs",
    "log_target_config",
)

import logging
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, ClassVar

from kaparoo.utils import unwrap_or_factory

from iivs_cardio.common.logging import log_indented
from iivs_cardio.common.pipeline.branch import ensure_json_name
from iivs_cardio.data.transforms.filtering.kernel import IdentityConfig
from iivs_cardio.optical_flow.data import FLOW_FLOAT_NPY
from iivs_cardio.optical_flow.pipeline import (
    EvaluationDocument,
    FlowStageFactory,
    FlowTree,
)
from scripts._common.dataset import (
    BranchConfig,
    TreeBranchConfig,
    ensure_output_clear,
    log_branch_policies,
    log_source_config,
)
from scripts._common.phase import (
    PhaseSourceConfig,
    build_sequences,
    log_short_sequences,
)
from scripts.data._filtering import describe_filter_kernel, log_filter_config
from scripts.optical_flow._estimating import (
    describe_estimator_config,
    log_estimator_config,
)
from scripts.optical_flow._normalizing import (
    NormalizeConfig,
    build_normalization,
    log_normalize_config,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from logging import Logger

    from kaparoo.filesystem.types import StrPath
    from torch import Tensor

    from iivs_cardio.common.pipeline import SideBranch
    from iivs_cardio.data.transforms.filtering.kernel import KernelConfig
    from iivs_cardio.optical_flow.estimators import EstimatorConfig
    from iivs_cardio.optical_flow.pipeline import FlowSource
    from scripts._common.dataset import SequenceSelectConfig
    from scripts.optical_flow._normalizing import Normalization


# ========================== #
#          Settings          #
# ========================== #


@dataclass
class FlowSourceConfig(PhaseSourceConfig):
    """The tree this stage reads, which is phase however it was written.

    The same shape whether it reads an acquisition or the cache preprocessing
    left: which of the two it is shows in `root` and in the filter, not here.

    Attributes:
        DEFAULT_SUBPATH: As `PhaseSourceConfig`.
        subpath: As `PhaseSourceConfig`.
        root: As `PhaseSourceConfig`.
        frames: As `PhaseSourceConfig`.
    """


@dataclass
class FlowBranchConfig(TreeBranchConfig):
    """The branch that writes each sequence's flows back out as a tree.

    Attributes:
        DEFAULT_SUBPATH: Koala's own `<modality>/<precision>/<format>` form for
            a flow tree. Never followed off the source the way a frame tree's
            is: the source holds phase and this holds flows, so the two are
            different kinds and land in different folders by default.
        save: As `TreeBranchConfig`, defaulting to `False` since a run that
            only scores does not need them.
        subpath: As `TreeBranchConfig`.
        record_file: As `TreeBranchConfig`.
        if_present: As `TreeBranchConfig`.
        if_unsourced: As `TreeBranchConfig`.
    """

    DEFAULT_SUBPATH: ClassVar[str] = FLOW_FLOAT_NPY


@dataclass
class EvaluationBranchConfig(BranchConfig):
    """The branch that gathers what every sequence scored into one document.

    Attributes:
        save: Whether to write the document. Defaults to `True`, since it is
            what the stage exists to produce and the flows are the optional
            half.
        file: The name the document is given, given `.json` if it has no
            extension. Defaults to `"flow_evaluation"`, the prefix leaving room
            for a later stage writing its own document to the same root.
        if_present: As `BranchConfig`.
        if_unsourced: As `BranchConfig`.
    """

    save: bool = True
    file: str = "flow_evaluation"


@dataclass
class FlowTargetConfig:
    """What a run writes, one block per branch.

    Where they land is not here: `run_root` places the job's directory, and the
    folder a branch actually writes under is the one hydra made for the job.

    Attributes:
        flows: The branch writing the flow fields.
        evaluations: The branch writing what they scored.
    """

    flows: FlowBranchConfig = field(default_factory=FlowBranchConfig)
    evaluations: EvaluationBranchConfig = field(default_factory=EvaluationBranchConfig)


def _evaluation_file(target_config: FlowTargetConfig) -> str:
    """Return what the evaluation document is called, given `.json` if it has none.

    Raises:
        ValueError: If the name is a path or carries some other extension. The
            library's own refusal says what is wrong with the name but not
            which setting holds it, which is where a reader has to go.
    """
    try:
        return ensure_json_name(target_config.evaluations.file)
    except ValueError as error:
        msg = f"`target.evaluations.file`: {error}"
        raise ValueError(msg) from error


def _validate_output(
    source_config: FlowSourceConfig,
    target_config: FlowTargetConfig,
    output_root: StrPath,
) -> None:
    """Raise unless the target names an output this run can safely write.

    Raises:
        ValueError: If the target writes nothing, or the flows it writes would
            land on the phase they are read from.
    """
    if not target_config.flows.save:
        if not target_config.evaluations.save:
            fix = "set `target.evaluations.save` or `target.flows.save`"
            msg = f"nothing to do: {fix}"
            raise ValueError(msg)
        return

    ensure_output_clear(
        source_config.root,
        output_root,
        what="flows",
        read=source_config.resolve_subpath(),
        written=target_config.flows.resolve_subpath(),
        fix="`target.flows.subpath` beside it, or `run_root` outside the source",
    )


# ========================== #
#          Logging           #
# ========================== #


def log_target_config(
    target_config: FlowTargetConfig,
    logger: Logger,
    *,
    output_root: StrPath,
) -> None:
    """Log what a run writes and where, naming each output it will produce.

    Args:
        target_config: The settings saying what the run was told to write.
        logger: The logger the lines go to.
        output_root: The folder the branches write under, which hydra made for
            this job. A sweep gives each of its jobs one of its own beneath
            `run_root`, so the two are the same path only in a lone run.
    """
    log_indented(logger, "target: %s", PurePath(output_root).as_posix(), depth=0)

    if not (target_config.flows.save or target_config.evaluations.save):
        log_indented(logger, "writing nothing")
        return

    if target_config.flows.save:
        written = target_config.flows.resolve_subpath()
        layout = f"<sequence>/{written}" if written else "<sequence>/*"
        log_indented(logger, "writing the flows to %s", layout)
        log_branch_policies("flows", target_config.flows, logger)

    if target_config.evaluations.save:
        name = _evaluation_file(target_config)
        log_indented(logger, "writing the evaluations to %s", name)
        log_branch_policies("evaluations", target_config.evaluations, logger)


def log_configs(
    source_config: FlowSourceConfig,
    sequence_config: SequenceSelectConfig,
    kernel_config: KernelConfig,
    estimator_config: EstimatorConfig,
    normalization: Normalization,
    target_config: FlowTargetConfig | None,
    *,
    output_root: StrPath,
    name: str,
) -> None:
    """Log the whole configuration of a run, as one block per part.

    A run that writes nothing has no target to describe, which is what an
    absent `target_config` means.
    """
    logger = logging.getLogger(name)

    log_source_config(source_config, sequence_config, logger)
    log_filter_config(kernel_config, logger)
    log_normalize_config(normalization, logger)
    log_estimator_config(estimator_config, logger)

    if target_config is not None:
        log_target_config(target_config, logger, output_root=output_root)


# ========================== #
#          Building          #
# ========================== #


def build_branches(
    source_config: FlowSourceConfig,
    kernel_config: KernelConfig,
    estimator_config: EstimatorConfig,
    normalization: Normalization,
    target_config: FlowTargetConfig,
    *,
    output_root: StrPath,
    contents: Mapping[str, Sequence[str]],
    selected: Sequence[str] | None = None,
) -> list[SideBranch[FlowSource, Tensor, Path]]:
    """Build the branches a target describes, in the order they will watch.

    Which sequences a run took is not recorded, since it changes what the run
    covers rather than what any sequence's numbers mean, and `coverage` reports
    it already.

    The ranges the frames were scaled by go into the settings, not the path of
    the document they were read from. The same path may hold a document that was
    written again, and a run that scaled by other numbers must not read as this
    one when a later run comes to decide what it may reuse.

    Args:
        source_config: The tree the run reads, recorded in what the branches
            write.
        kernel_config: The filter, recorded for a later run to compare against.
        estimator_config: The estimator, recorded for the same reason.
        normalization: The scaling, whose account is recorded with the rest.
        target_config: The settings saying what the run writes.
        output_root: The folder the branches write under.
        contents: Every sequence the source holds, against the frames each
            would be read over.
        selected: The sequences of those this run was given. Defaults to `None`,
            which takes all of them.

    Returns:
        The branches, empty of neither output when the target asks for both.

    Raises:
        ValueError: If the target writes nothing, if the flows it writes would
            land on the phase they are read from, or if a policy names
            something no branch offers.
    """
    _validate_output(source_config, target_config, output_root)

    branches: list[SideBranch[FlowSource, Tensor, Path]] = []

    frames = source_config.frames

    settings = {
        "source": {
            "subpath": source_config.resolve_subpath(),
            "frames": {
                "start": frames.start,
                "step": frames.step,
                "count": frames.count,
            },
        },
        "filter": describe_filter_kernel(kernel_config),
        "normalize": dict(normalization.described),
        "estimator": describe_estimator_config(estimator_config),
    }

    if (branch := target_config.flows).save:
        branches.append(
            FlowTree(
                output_root,
                branch.resolve_subpath(),
                contents,
                settings,
                selected=selected,
                record_file=branch.record_file,
                if_present=branch.if_present,
                if_unsourced=branch.if_unsourced,
            )
        )

    if (branch := target_config.evaluations).save:
        branches.append(
            EvaluationDocument(
                Path(output_root, _evaluation_file(target_config)),
                source_config.root,
                contents,
                settings,
                selected=selected,
                if_present=branch.if_present,
                if_unsourced=branch.if_unsourced,
            )
        )

    return branches


def build_flow_stages(
    source_config: FlowSourceConfig,
    sequence_config: SequenceSelectConfig,
    estimator_config: EstimatorConfig,
    normalize_config: NormalizeConfig | None = None,
    kernel_config: KernelConfig | None = None,
    target_config: FlowTargetConfig | None = None,
    *,
    output_root: StrPath,
    name: str,
) -> FlowStageFactory:
    """Assemble everything a run needs from the configuration it was given.

    The configuration is logged before the sources are searched, so a run says
    what it was asked to do even when it cannot do it. The scaling is settled
    first for that reason: a measured level says which numbers it landed on
    rather than which document it was pointed at, and reading that document is
    what makes the line sayable at all. A target that writes nothing, or that
    would write over the source, is refused at the same point.

    Args:
        source_config: The tree the sequences are read from.
        sequence_config: Which of its sequences to read.
        estimator_config: The estimator the flows are computed with.
        normalize_config: Where the range the frames are scaled from comes from.
            Defaults to `None`, which takes the dataset range of a document that
            has to be named.
        kernel_config: The filter to apply. Defaults to `None`, which leaves the
            frames as they are, and is what a run reading a filtered cache takes.
        target_config: The settings saying what to write. Defaults to `None`,
            for a run that only reads.
        output_root: The folder the branches write under.
        name: The name the run is called by.

    Returns:
        The factory a driver runs the sequences through.

    Raises:
        ValueError: If the target writes nothing, if it would write over the
            source, if the source search finds nothing to run, or if a sequence
            it found has no range to be scaled by.
    """
    kernel_config = unwrap_or_factory(kernel_config, IdentityConfig)
    normalize_config = unwrap_or_factory(normalize_config, NormalizeConfig)

    normalization = build_normalization(normalize_config, estimator_config.FRAME_DTYPE)

    log_configs(
        source_config,
        sequence_config,
        kernel_config,
        estimator_config,
        normalization,
        target_config,
        output_root=output_root,
        name=name,
    )

    if target_config is not None:
        _validate_output(source_config, target_config, output_root)

    sequences, contents = build_sequences(source_config, sequence_config, kernel_config)

    log_short_sequences(
        source_config.frames,
        sequences=sequences,
        contents=contents,
        name=name,
    )

    branches: list[SideBranch[FlowSource, Tensor, Path]] = []

    if target_config is not None:
        selected = [sequence.name for sequence in sequences]

        branches = build_branches(
            source_config,
            kernel_config,
            estimator_config,
            normalization,
            target_config,
            output_root=output_root,
            contents=contents,
            selected=selected,
        )

    return FlowStageFactory(
        sequences,
        normalization.normalizers(contents),
        estimator_config,
        *branches,
        name=name,
    )
