"""Write a `.env` with the project paths the scripts read.

Finds the project root (the directory holding `pyproject.toml`) from this file's
location and writes `PROJECT_ROOT` -- the one machine-specific path -- plus
`CONFIGS_ROOT` interpolated from it. `CONFIGS_ROOT` is materialized here, rather
than left for each script to re-join, so the `${PROJECT_ROOT}/configs` derivation
lives in one place while `PROJECT_ROOT` stays the single source of truth. Refuses
to clobber an existing `.env` unless `--force` is passed, since it is a
machine-local, git-ignored file the user may have edited.
"""

from __future__ import annotations

import argparse
from pathlib import Path

_TEMPLATE = """\
PROJECT_ROOT={root}
CONFIGS_ROOT=${{PROJECT_ROOT}}/configs
"""


def _project_root() -> Path:
    """The nearest ancestor of this file that holds a `pyproject.toml`."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    msg = "no pyproject.toml found above this script"
    raise FileNotFoundError(msg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="overwrite an existing .env"
    )
    args = parser.parse_args()

    root = _project_root()
    env_path = root / ".env"
    if env_path.exists() and not args.force:
        print(f"{env_path} already exists; pass --force to overwrite")
        return

    # POSIX slashes so the same value reads cleanly on Windows and elsewhere.
    env_path.write_text(_TEMPLATE.format(root=root.as_posix()), encoding="utf-8")
    print(f"wrote {env_path}")


if __name__ == "__main__":
    main()
