# iivs-cardio

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg?logo=opensourceinitiative&logoColor=white)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.13%20%7C%203.14-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12.1-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-13.0-76B900.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-orange.json)](https://github.com/copier-org/copier)

*A monorepo for deep-learning and bio-imaging research on cardiomyocytes*

## 🚀 Getting started

Requires **Python 3.13 or newer** and an **NVIDIA GPU** with a CUDA
13.0-capable driver.

```bash
git clone https://github.com/iivs-lab/cardiomyocytes.git
cd cardiomyocytes
uv sync --group dev
```

The first `uv sync` pulls the CUDA build of `torch` (~1.8 GB). On Windows,
OpenCV additionally needs the one-time cuDNN step below.

### 🖥️ Compute environment

The whole stack is pinned to **CUDA 13.0** on NVIDIA GPUs and installed by
`uv sync`:

- **Python** 3.13 or newer
- **PyTorch** `2.12.1` / **torchvision** `0.27.1` — from the dedicated PyTorch
  index (`.../whl/cu130`) set in [`pyproject.toml`](./pyproject.toml)
- **OpenCV** `opencv-contrib-python 4.13.0.90` — a CUDA build
  ([`cudawarped`](https://github.com/cudawarped/opencv-python-cuda-wheels)
  wheels) linking the system CUDA 13.0 runtime and cuDNN

### 🪟 Windows: OpenCV CUDA setup

On Windows the OpenCV CUDA wheel can't find cuDNN (the cuDNN v9 installer keeps
it in its own folder, and Python 3.8+ no longer searches `PATH`). Run once, in
an **Administrator** PowerShell, to symlink cuDNN where the wheel looks:

```powershell
./scripts/compute_env/setup-opencv-cuda.ps1
```

A bare `import cv2` then loads the CUDA build in any environment (venv, uv,
conda, …). Notes:

- **cuDNN from a zip** — unpacked into `bin\x64`: the script no-ops; unpacked
  elsewhere: pass `-CUDNN_PATH <folder>`.
- **After a cuDNN/CUDA upgrade** — re-run to repoint the links.
- **Linux** — not needed; `ld.so` finds cuDNN via `RPATH` / `LD_LIBRARY_PATH` /
  `ldconfig`.

## 📋 TODO

See [TODO.md](./TODO.md) for tracked open items.

## 📜 Changelog

See [CHANGELOG.md](./CHANGELOG.md) for the version history.

## ⚖️ License

This project is distributed under the terms of the [MIT](./LICENSE) license.
