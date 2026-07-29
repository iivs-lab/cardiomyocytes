from __future__ import annotations

__all__ = (
    "FLOW_CHANNELS",
    "FLOW_NDIM",
    "OpticalFlowFolder",
    "load_flow_npy",
    "read_flow_npy_header",
    "save_flow_folder",
    "save_flow_npy",
)

from functools import cached_property, partial
from typing import TYPE_CHECKING, ClassVar, override

import numpy as np
from iivs.common.data import ArrayFileList, validate_float32_array, write_npy
from iivs.dhm.data.koala import KoalaFrameFolder, save_koala_frames
from kaparoo.filesystem import ensure_file_exists, ensure_file_extension
from numpy.typing import NDArray  # runtime: subscripted in the class bases below

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path
    from typing import Any

    from iivs.common.data import OnNonFinite
    from iivs.dhm.data.koala import ValidationLevel
    from kaparoo.filesystem.types import StrPath

FLOW_NDIM = 3
"""Axis count of a stored flow field: `(2, H, W)`."""

FLOW_CHANNELS = 2
"""Size of the leading (channel) axis: `0` = dx, `1` = dy."""

# `read_magic` reports the `.npy` format version; only 1.0 and 2.0 have a public
# header reader. Version 3.0 exists solely for structured dtypes with non-latin1
# field names, which a flow field never is.
_HEADER_READERS = {
    (1, 0): np.lib.format.read_array_header_1_0,
    (2, 0): np.lib.format.read_array_header_2_0,
}


def _validate_flow_shape(shape: tuple[int, ...], path: Path) -> tuple[int, int, int]:
    """Narrow `shape` to `(2, H, W)`, naming `path` if it is anything else.

    Raises:
        ValueError: If `shape` is not `FLOW_NDIM` axes leading with `FLOW_CHANNELS`.
    """
    if len(shape) != FLOW_NDIM or shape[0] != FLOW_CHANNELS:
        msg = f"expected a ({FLOW_CHANNELS}, H, W) flow field, got {shape}: {path}"
        raise ValueError(msg)
    return shape[0], shape[1], shape[2]


def read_flow_npy_header(path: StrPath) -> tuple[tuple[int, int, int], np.dtype[Any]]:
    """Read a flow `.npy`'s shape and dtype without decoding its pixels.

    Reads the `.npy` header block alone, so the cost does not scale with the frame:
    about 5x cheaper than `np.load(mmap_mode="r")` and 30x cheaper than a full decode
    at this project's frame size. Pickle is never reached, since only the header is
    parsed.

    Args:
        path: The `.npy` file to inspect.

    Returns:
        The `(2, H, W)` shape and the stored dtype, which the caller checks -- a flow
        folder wants float32, but reading a foreign dtype is not itself an error.

    Raises:
        FileNotFoundError: If `path` does not exist.
        NotAFileError: If `path` exists but is not a regular file.
        ValueError: If the file is not a `.npy` of format 1.0 or 2.0, or its shape is
            not `(2, H, W)`.
    """
    path = ensure_file_exists(path)
    with path.open("rb") as file:
        version = np.lib.format.read_magic(file)
        reader = _HEADER_READERS.get(version)
        if reader is None:
            msg = f"expected .npy format 1.0 or 2.0, got {version}: {path}"
            raise ValueError(msg)
        shape, _fortran_order, dtype = reader(file)
    return _validate_flow_shape(shape, path), dtype


def load_flow_npy(
    path: StrPath, *, on_nonfinite: OnNonFinite = "ignore"
) -> NDArray[np.float32]:
    """Load a header-less `.npy` file as one `(2, H, W)` float32 flow field.

    Pickle is disabled, so an object array is refused rather than unpickled.

    Args:
        path: The `.npy` file to read.
        on_nonfinite: What to do about NaN / inf in the loaded array: `"ignore"`
            (default), `"warn"`, or `"raise"`.

    Returns:
        The flow field, channel `0` = dx and `1` = dy, in pixels.

    Raises:
        FileNotFoundError: If `path` does not exist.
        NotAFileError: If `path` exists but is not a regular file.
        ValueError: If the array is pickled, is not a `(2, H, W)` float32 field, or
            holds non-finite values while `on_nonfinite` is `"raise"`.
    """
    path = ensure_file_exists(path)
    data = np.load(path, allow_pickle=False)
    _validate_flow_shape(data.shape, path)
    return validate_float32_array(
        data, ndim=FLOW_NDIM, on_nonfinite=on_nonfinite, allow_stack=False
    )


def save_flow_npy(
    path: StrPath,
    flow: NDArray[np.float32],
    *,
    overwrite: bool = False,
    on_nonfinite: OnNonFinite = "warn",
) -> None:
    """Atomically save one `(2, H, W)` float32 flow field as an uncompressed `.npy`.

    Header-less: `.npy` stores the array alone, so the pixel size and frame interval a
    flow needs to become a physical velocity are not carried here.

    Args:
        path: The `.npy` file to write; the extension is added when absent.
        flow: The flow field to save, channel `0` = dx and `1` = dy.
        overwrite: Whether to replace `path` if it already exists.
        on_nonfinite: What to do about NaN / inf in `flow`: `"ignore"`, `"warn"`
            (default), or `"raise"`.

    Raises:
        ValueError: If `path` has a non-`.npy` extension, `flow` is not a
            `(2, H, W)` float32 field, or it holds non-finite values while
            `on_nonfinite` is `"raise"`.
        FileExistsError: If `path` exists and `overwrite` is False.
        FileNotFoundError: If the parent directory of `path` does not exist.
    """
    path = ensure_file_extension(path, OpticalFlowFolder.FILE_EXT, add=True)
    _validate_flow_shape(flow.shape, path)
    flow = validate_float32_array(
        flow, ndim=FLOW_NDIM, on_nonfinite=on_nonfinite, allow_stack=False
    )
    write_npy(path, flow, overwrite=overwrite)


def save_flow_folder(
    dest: StrPath,
    flows: Iterable[NDArray[np.float32]],
    *,
    overwrite: bool = False,
    on_nonfinite: OnNonFinite = "warn",
) -> None:
    """Write `flows` into `dest` as a numbered folder `OpticalFlowFolder` reads back.

    `flows` is consumed one field at a time, so a whole sequence is never held in
    memory, and the folder is built atomically -- a failure part-way leaves any
    existing `dest` untouched rather than a half-written folder.

    Args:
        dest: The folder to create and fill.
        flows: The flow fields to write, in order.
        overwrite: Whether to replace `dest` if it already exists.
        on_nonfinite: What to do about NaN / inf in each field: `"ignore"`, `"warn"`
            (default), or `"raise"`.

    Raises:
        ValueError: If `flows` is empty, or a field is not a `(2, H, W)` float32 array.
        FileExistsError: If `dest` exists and `overwrite` is False.
    """
    save = partial(save_flow_npy, overwrite=overwrite, on_nonfinite=on_nonfinite)
    save_koala_frames(
        dest,
        flows,
        save,
        stem=OpticalFlowFolder.FILE_STEM,
        ext=OpticalFlowFolder.FILE_EXT,
        kind="optical flow",
        overwrite=overwrite,
    )


class OpticalFlowFolder(
    KoalaFrameFolder[NDArray[np.float32]], ArrayFileList[np.float32]
):
    """An ordered folder of numbered `{index:05d}_flow.npy` float32 flow fields.

    One estimator run over one sequence: `N` frames give `N - 1` fields, each
    `(2, H, W)` with channel `0` = dx and `1` = dy, in pixels. Items come back as
    `numpy` arrays, the boundary every `iivs-lib` folder keeps; wrap the folder to
    reach `torch` -- `FilteredSequence` and `TransformedSequence` are how the rest of
    this project crosses it.

    Validation has three depths, and the middle one is why the default is affordable:
    `"names"` reads no file at all, `"headers"` reads each `.npy` header (shape and
    dtype, no pixels), and `"data"` decodes every field. On a 1000-field folder that
    is roughly 0 ms, 100 ms, and 4 s.

    Args:
        root: The folder to scan.
        validate: Run `validate` to this level at construction, or None to skip.
            Defaults to `"headers"`.

    Raises:
        DirectoryNotFoundError: If `root` does not exist.
        FileNotFoundError: If `root` holds no `NNNNN_flow.npy` files.
        ValueError: If `validate` is set and the folder fails at that level.
    """

    FILE_STEM: ClassVar[str] = "flow"
    FILE_EXT: ClassVar[str] = "npy"
    LEVELS: ClassVar[tuple[str, ...]] = ("names", "headers", "data")
    DEFAULT_LEVEL: ClassVar[str] = "headers"

    def __init__(
        self, root: StrPath, *, validate: ValidationLevel | None = "headers"
    ) -> None:
        super().__init__(root, validate=validate)

    @cached_property
    @override
    def frame_shape(self) -> tuple[int, int]:
        """The (height, width) of each field, from the first file's header.

        The trailing two axes, not the leading two: a flow field carries its channel
        axis first, so the frame is what follows it.
        """
        shape, _dtype = read_flow_npy_header(self.get_file(0))
        return shape[1], shape[2]

    @override
    def load_file(self, path: Path) -> NDArray[np.float32]:
        """Load the `(2, H, W)` float32 flow field at `path` (pickle disabled)."""
        return load_flow_npy(path)

    @override
    def _validate_content(self, path: Path, *, level: str) -> None:
        """Check `path`'s header against the first file; at `"data"`, decode it too.

        Reading the header already pins the `(2, H, W)` layout, so this adds the dtype
        and the shared frame shape. `"data"` then decodes with `on_nonfinite="raise"`,
        which is the only check that has to touch the pixels.

        Raises:
            ValueError: If the field is not float32, its frame shape differs from the
                first file's, or -- at `"data"` -- it holds non-finite values.
        """
        shape, dtype = read_flow_npy_header(path)

        if dtype != np.float32:
            msg = f"expected a float32 field, got {dtype}: {path}"
            raise ValueError(msg)

        if (frame := (shape[1], shape[2])) != self.frame_shape:
            expected = self.frame_shape
            msg = f"expected the first file's frame {expected}, got {frame}: {path}"
            raise ValueError(msg)

        if level == "data":
            load_flow_npy(path, on_nonfinite="raise")
