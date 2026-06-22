# iivs-cardio

[![Python](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-13.0-76B900.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-orange.json)](https://github.com/copier-org/copier)

*A monorepo for deep-learning and bio-imaging research on cardiomyocytes*

## 📦 Development setup

Requires Python 3.14+.

```bash
git clone https://github.com/iivs-lab/cardiomyocytes.git
cd cardiomyocytes
uv sync --group dev
```

## 🖥️ Compute environment

This project targets **PyTorch on NVIDIA GPUs**:

- **Python** 3.14+
- **PyTorch** `2.12.1` / **torchvision** `0.27.1`
- **Compute backend**: CUDA **13.0** — wheels are pulled from the dedicated
  PyTorch index (`https://download.pytorch.org/whl/cu130`) configured in
  [`pyproject.toml`](./pyproject.toml); `uv sync` installs them automatically.

> An NVIDIA GPU with a CUDA 13.0-capable driver is expected. The first
> `uv sync` downloads the CUDA build of `torch` (~1.8 GB).

### Switching the compute backend

The backend is selected by the `[[tool.uv.index]]` named `pytorch` in
`pyproject.toml`:

- **CUDA**: set the URL to `.../whl/cu126`, `.../whl/cu128`, or `.../whl/cu130`.
- **CPU-only**: set the URL to `.../whl/cpu`.

Then re-resolve with `uv lock && uv sync`. Alternatively re-run
`copier update --UNSAFE --vcs-ref pytorch` and re-answer
`compute_backend` / `cuda_version`.

## 📋 TODO

See [TODO.md](./TODO.md) for tracked open items.

## 📜 Changelog

See [CHANGELOG.md](./CHANGELOG.md) for the version history.

## ⚖️ License

This project is distributed under the terms of the [MIT](./LICENSE) license.
