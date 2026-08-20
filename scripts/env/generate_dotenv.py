import argparse
from pathlib import Path

_DESCRIPTION = """\
Write the .env at the project root. Every script under scripts/ loads it with
python-dotenv's load_dotenv() and then reads the values as os.environ[...].
"""
_FORCE_HELP = "overwrite an existing .env"

_TEMPLATE = """\
PROJECT_ROOT={root}
CONFIGS_ROOT=${{PROJECT_ROOT}}/configs
"""


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent

    msg = "no pyproject.toml above this script: run it from inside a checkout"
    raise FileNotFoundError(msg)


def main() -> None:
    parser = argparse.ArgumentParser(description=_DESCRIPTION)
    parser.add_argument("--force", action="store_true", help=_FORCE_HELP)
    args = parser.parse_args()

    root = _project_root()
    env_file = root / ".env"

    if env_file.exists() and not args.force:
        print(f"{env_file} already exists; pass --force to overwrite")
        return

    # POSIX slashes so the same value reads cleanly on Windows and elsewhere.
    env_file.write_text(_TEMPLATE.format(root=root.as_posix()), encoding="utf-8")
    print(f"wrote {env_file}")


if __name__ == "__main__":
    main()
