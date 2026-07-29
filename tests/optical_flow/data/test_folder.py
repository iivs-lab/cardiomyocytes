from __future__ import annotations

import numpy as np
import pytest

from iivs_cardio.optical_flow.data.folder import (
    OpticalFlowFolder,
    load_flow_npy,
    read_flow_npy_header,
    save_flow_folder,
    save_flow_npy,
)


def _flow(value: float, height: int = 6, width: int = 8) -> np.ndarray:
    # Distinct per channel, so a transposed or swapped axis shows up as a value,
    # not only as a shape.
    flow = np.empty((2, height, width), dtype=np.float32)
    flow[0] = value
    flow[1] = -value
    return flow


def _folder(root, count: int = 4, **kwargs) -> object:
    root.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        np.save(root / f"{index:05d}_flow.npy", _flow(index, **kwargs))
    return root


# ------------------------------ reading a folder ------------------------------ #


def test_folder_discovers_numbered_files_in_index_order(tmp_path):
    root = _folder(tmp_path / "flows", count=3)
    folder = OpticalFlowFolder(root)
    assert len(folder) == 3
    assert [p.name for p in folder.files] == [
        "00000_flow.npy",
        "00001_flow.npy",
        "00002_flow.npy",
    ]


def test_folder_loads_each_field_with_its_channels_intact(tmp_path):
    folder = OpticalFlowFolder(_folder(tmp_path / "flows"))
    field = folder[2]
    assert field.shape == (2, 6, 8)
    assert field.dtype == np.float32
    assert np.array_equal(field[0], np.full((6, 8), 2.0, dtype=np.float32))
    assert np.array_equal(field[1], np.full((6, 8), -2.0, dtype=np.float32))


def test_folder_meta_is_the_source_path(tmp_path):
    root = _folder(tmp_path / "flows")
    folder = OpticalFlowFolder(root)
    assert folder.get_meta(2) == root / "00002_flow.npy"


def test_folder_frame_shape_drops_the_channel_axis(tmp_path):
    # (2, H, W): the frame is the *trailing* two axes. Taking the leading two --
    # what an image folder does -- would give (2, 6) here.
    folder = OpticalFlowFolder(_folder(tmp_path / "flows", height=6, width=8))
    assert folder.frame_shape == (6, 8)


def test_folder_rejects_a_root_without_flow_files(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    np.save(empty / "00000_phase.npy", _flow(0))  # right numbering, wrong stem
    with pytest.raises(FileNotFoundError, match=r"NNNNN_flow\.npy"):
        OpticalFlowFolder(empty)


# -------------------------------- validation ---------------------------------- #


def test_folder_rejects_non_contiguous_numbering(tmp_path):
    root = _folder(tmp_path / "flows", count=3)
    (root / "00001_flow.npy").unlink()  # leaves 00000, 00002
    with pytest.raises(ValueError, match=r"non-contiguous: expected 00001_flow\.npy"):
        OpticalFlowFolder(root, validate="names")


def test_folder_rejects_a_field_without_a_channel_axis(tmp_path):
    root = _folder(tmp_path / "flows", count=2)
    np.save(root / "00001_flow.npy", np.zeros((6, 8), dtype=np.float32))
    with pytest.raises(ValueError, match=r"must hold one \(2, H, W\) flow field"):
        OpticalFlowFolder(root)


def test_folder_rejects_a_wrong_channel_count(tmp_path):
    root = _folder(tmp_path / "flows", count=2)
    np.save(root / "00001_flow.npy", np.zeros((3, 6, 8), dtype=np.float32))
    with pytest.raises(ValueError, match=r"got shape \(3, 6, 8\)"):
        OpticalFlowFolder(root)


def test_folder_rejects_a_frame_shape_that_differs_from_the_first(tmp_path):
    root = _folder(tmp_path / "flows", count=2)
    np.save(root / "00001_flow.npy", _flow(1, height=6, width=9))
    with pytest.raises(ValueError, match=r"must match the first file \(6, 8\)"):
        OpticalFlowFolder(root)


def test_folder_rejects_a_non_float32_dtype(tmp_path):
    # Only the header level catches this; the shape alone is right.
    root = _folder(tmp_path / "flows", count=2)
    np.save(root / "00001_flow.npy", np.zeros((2, 6, 8), dtype=np.float64))
    with pytest.raises(ValueError, match="must be float32"):
        OpticalFlowFolder(root)


def test_folder_names_level_accepts_what_headers_rejects(tmp_path):
    # The levels must differ in depth, not just in name: a float64 field passes
    # `names` (its filename is fine) and fails `headers`.
    root = _folder(tmp_path / "flows", count=2)
    np.save(root / "00001_flow.npy", np.zeros((2, 6, 8), dtype=np.float64))
    OpticalFlowFolder(root, validate="names")  # must not raise
    with pytest.raises(ValueError, match="must be float32"):
        OpticalFlowFolder(root, validate="headers")


def test_folder_headers_level_accepts_what_data_rejects(tmp_path):
    # Same, one level up: a NaN is invisible in the header and caught by `data`.
    root = _folder(tmp_path / "flows", count=2)
    field = _flow(1)
    field[0, 0, 0] = np.nan
    np.save(root / "00001_flow.npy", field)
    OpticalFlowFolder(root, validate="headers")  # must not raise
    with pytest.raises(ValueError, match=r"must be finite \(got 1 NaN"):
        OpticalFlowFolder(root, validate="data")


def test_folder_headers_level_never_decodes_the_pixels(monkeypatch, tmp_path):
    # The whole point of the `headers` level is that its cost does not scale with
    # the frame. Only `load_flow_npy` calls `np.load`, so spying there proves the
    # pixels stay on disk -- a result-only assertion cannot show this.
    root = _folder(tmp_path / "flows", count=3)
    real_load = np.load
    calls = 0

    def counting(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_load(*args, **kwargs)

    monkeypatch.setattr(np, "load", counting)
    folder = OpticalFlowFolder(root, validate="headers")
    assert calls == 0  # 3 files validated, none decoded

    folder.validate(level="data")
    assert calls == 3  # now every file is read


def test_folder_skips_validation_when_asked(tmp_path):
    root = _folder(tmp_path / "flows", count=2)
    np.save(root / "00001_flow.npy", np.zeros((2, 6, 8), dtype=np.float64))
    folder = OpticalFlowFolder(root, validate=None)  # must not raise
    assert len(folder) == 2


def test_folder_rejects_an_unsupported_level(tmp_path):
    folder = OpticalFlowFolder(_folder(tmp_path / "flows"))
    with pytest.raises(ValueError, match="level"):
        folder.validate(level="pixels")  # ty: ignore[invalid-argument-type]


# ------------------------------- header reading -------------------------------- #


def test_read_flow_npy_header_returns_shape_and_dtype(tmp_path):
    path = tmp_path / "00000_flow.npy"
    np.save(path, _flow(0, height=6, width=8))
    assert read_flow_npy_header(path) == ((2, 6, 8), np.dtype(np.float32))


def test_read_flow_npy_header_reports_a_foreign_dtype_rather_than_raising(tmp_path):
    # The dtype is returned for the caller to judge; only the shape is enforced.
    path = tmp_path / "00000_flow.npy"
    np.save(path, np.zeros((2, 6, 8), dtype=np.float64))
    shape, dtype = read_flow_npy_header(path)
    assert shape == (2, 6, 8)
    assert dtype == np.dtype(np.float64)


def test_read_flow_npy_header_rejects_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_flow_npy_header(tmp_path / "absent.npy")


def test_read_flow_npy_header_rejects_an_unsupported_npy_version(tmp_path):
    # Only `.npy` 1.0 and 2.0 expose a public header reader. Patch the major
    # version byte (offset 6, right after the 6-byte magic) to claim 3.0.
    path = tmp_path / "00000_flow.npy"
    np.save(path, _flow(0))
    raw = bytearray(path.read_bytes())
    raw[6] = 3
    path.write_bytes(raw)
    with pytest.raises(ValueError, match=r"unsupported \.npy format version \(3, 0\)"):
        read_flow_npy_header(path)


# ---------------------------------- writing ------------------------------------ #


def test_save_flow_npy_round_trips(tmp_path):
    path = tmp_path / "00000_flow.npy"
    saved = _flow(3.5)
    save_flow_npy(path, saved)
    assert np.array_equal(load_flow_npy(path), saved)


def test_save_flow_npy_rejects_a_non_flow_array(tmp_path):
    with pytest.raises(ValueError, match=r"must hold one \(2, H, W\) flow field"):
        save_flow_npy(tmp_path / "00000_flow.npy", np.zeros((6, 8), dtype=np.float32))


def test_save_flow_npy_refuses_to_clobber(tmp_path):
    path = tmp_path / "00000_flow.npy"
    save_flow_npy(path, _flow(1))
    with pytest.raises(FileExistsError):
        save_flow_npy(path, _flow(2))
    save_flow_npy(path, _flow(2), overwrite=True)
    assert np.array_equal(load_flow_npy(path), _flow(2))


def test_save_flow_folder_writes_names_the_folder_reads_back(tmp_path):
    dest = tmp_path / "flows"
    written = [_flow(i) for i in range(3)]
    save_flow_folder(dest, written)

    assert sorted(p.name for p in dest.iterdir()) == [
        "00000_flow.npy",
        "00001_flow.npy",
        "00002_flow.npy",
    ]
    folder = OpticalFlowFolder(dest, validate="data")
    assert len(folder) == len(written)
    for loaded, expected in zip(folder, written, strict=True):
        assert np.array_equal(loaded, expected)


def test_save_flow_folder_rejects_an_empty_sequence(tmp_path):
    dest = tmp_path / "flows"
    with pytest.raises(ValueError, match="empty optical flow sequence"):
        save_flow_folder(dest, [])
    assert not dest.exists()  # staged: nothing is left behind


def test_save_flow_folder_leaves_an_existing_folder_untouched_on_failure(tmp_path):
    dest = tmp_path / "flows"
    save_flow_folder(dest, [_flow(0)])

    bad = [_flow(1), np.zeros((6, 8), dtype=np.float32)]  # second field is not a flow
    with pytest.raises(ValueError, match=r"\(2, H, W\) flow field"):
        save_flow_folder(dest, bad, overwrite=True)

    folder = OpticalFlowFolder(dest, validate="data")
    assert len(folder) == 1
    assert np.array_equal(folder[0], _flow(0))
