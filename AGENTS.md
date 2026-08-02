# Agent guide — `iivs-cardio`

Guidance for AI coding assistants working on this project.
`CLAUDE.md` loads this file via the `@AGENTS.md` import.

This file carries only what the repository cannot state for itself. The
dependency list, the `ruff` rule set, the coverage gate, and the PyTorch
index all live in `pyproject.toml` — read them there rather than expecting
a copy here.

## Project

An **application, not a distributed package**: the editable install exists
so `import iivs_cardio` resolves, nothing is published, and there is no
`py.typed`. Keep the `uv` / `ruff` / `ty` / `pytest` toolchain unless there
is a clear reason to change it.

## Commands

```bash
uv sync --group dev      # create/refresh the environment
uv run ruff check .      # lint
uv run ruff format .     # format
uv run ty check          # type-check
uv run pytest            # run tests (coverage included by default)
uv run pytest --no-cov   # skip coverage for quick iteration
```

`ty` and `pytest` run **locally only**. The runtime stack is CUDA-only —
`cupy-cuda13x` and the cudawarped OpenCV build ship no CPU wheels and
cannot even `import` without an NVIDIA driver — so CI never installs the
project. It runs `ruff` and `uv lock --check`, nothing else: a green CI
says nothing about the type check or the tests, so run both before you
push.

## Conventions

- Keep code fully typed — `ty` runs with `error-on-warning`.
- Fix `ruff` findings rather than suppressing them, unless there is a
  clear, commented reason.
- `ty` has no plugin system; rely on standard typing (PEP 681
  `dataclass_transform`, `.pyi` stubs), not type-checker plugins.
- Suppress `ty` errors with `# ty: ignore[<error-name>]` using `ty`'s
  own error names (e.g. `invalid-argument-type`), not mypy/pyright
  codes. Always include the specific code rather than bare
  `# ty: ignore` — bare suppressions can mask future regressions.
- Error messages name the valid set or the fix, not just the failure
  ("unsupported node type 'X': expected File, Directory, ..."; "balance
  pause/resume, or use `suspend()`"). The message should tell the caller
  what to do next.
- Write each message as a single line — `msg = f"..."` — never split
  across lines with implicit concatenation inside parentheses, which is
  harder to read and to grep for when it surfaces in a traceback. `ruff`'s
  `line-length` is the real constraint, so when a message does not fit,
  shorten the prose or the variable interpolated into it rather than
  reaching for the multi-line form.

## Tests

- Mirror the package layout under `tests/`: `iivs_cardio/sub/mod.py`
  is tested by `tests/sub/test_mod.py`. Keep `__init__.py` markers in
  test subpackages.
- Not every source file needs a dedicated test file — types-only
  modules, re-export `__init__.py` markers, and details covered through
  a public-facing module are intentional exceptions.
- Flat module-level `def test_*` functions by default; reach for a plain
  `class TestX:` (grouping only, no inheritance) to organize a large or
  multi-feature surface. Don't mix the two styles within one file.
- Fixtures live in `tests/conftest.py` when you need them, applying to
  `tests/` only. Use a *root* `conftest.py` only for `pytest_plugins`
  declarations, doctest fixtures shared with source files, or
  project-wide collection hooks.
- Cross-module test helpers live in `tests/<pkg>/helpers.py`; shared
  fixtures in `tests/fixtures/`; per-package configuration in
  `conftest.py`.
- Assertions check concrete return values *and* side effects, not merely
  "no exception raised"; error paths use `pytest.raises(..., match=...)`;
  verify numbers against independently computed values, not the
  implementation's own output; make timing/IO deterministic (an injected
  clock, fault injection) rather than flaky. A good test fails on
  *subtle* breakage, not just obvious breakage. When a contract is "one
  batched call per source", verify it with a spy, not only by the result.

## Python style

`ruff` enforces the mechanical rules — future annotations, builtin
generics, import grouping and ordering — so run `uv run ruff check --fix`
rather than applying them by hand. What it cannot check:

- Within a function body, separate logical groups with a single blank
  line and put a blank line before the final `return`; leave a tightly
  coupled one- or two-line body unbroken.
- In a long module, group related definitions under a boxed comment
  banner — a centred title between two `#`-bordered rules.
- Standalone runnable scripts carry PEP 723 inline metadata (the
  `# /// script` block). `uv` manages it (`uv add --script`); add or edit
  it by hand only when explicitly asked.

### Docstrings

Optional — write them where they clarify intent, not mechanically.
"Mechanically" targets two habits to avoid: comments (or docstrings) that
merely restate the code, and a base class whose docstring explains itself
in terms of its specific subclasses — except a *closed* hierarchy's base,
which may name its subclasses as a deliberate family map. It is *not* a
licence to leave a consumed method bare. When written, document *intent
and contracts, not mechanism*:

- Lead with a one-line summary — a declarative noun phrase for classes
  ("An ordered, read-only view over ..."), an imperative verb phrase for
  functions and methods ("Yield successive windows from `items`."). Two
  kinds take a noun phrase instead: a property getter ("The reporting
  unit ...") and a boolean-returning *method*, which leads with
  "Whether ..." ("Whether `path` exists."). A boolean *function* stays
  imperative ("Test whether a path exists.").
- A concrete public method a caller consumes must be self-explainable
  from its own docstring and signature — never lean on an inherited
  parent docstring. Abstract base methods document only the generic
  contract and never name a specific subclass.
- Surface what callers cannot infer from the signature alone:
  invariants, edge cases, what subclasses must override, policy
  trade-offs. Skip restating what the code already shows.
- Use [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
  sections (`Args:`, `Returns:`, `Yields:`, `Raises:`,
  `Type Parameters:`); omit types from `Args:` since the signature
  already carries them. Custom sections (`Example:`, `Truth table:`, and
  ad-hoc labels) are welcome when they clarify a real pitfall or pattern.
- Add an `Args:` / `Returns:` block only for what the summary and
  signature cannot already convey. When they make the behaviour obvious
  (a no-arg getter, a self-evident one-liner), the summary *is* the whole
  docstring. Document an edge case shared across a family once on the
  class.
- Reference identifiers in backticks (`my_method`, `param`,
  `MyClass.method`). Literal option values get backticks too (`"merge"`,
  `"error"`), as identifiers do.

## Commit convention

Commit messages use a [Gitmoji](https://gitmoji.dev/) prefix and wrap
package/tool names in backticks:

```
<emoji> <Imperative summary; tool names in `backticks`>

<Optional body explaining *why*>
```

Prefixes this project uses:

| Prefix | Use for                                       |
| ------ | --------------------------------------------- |
| ✨     | New feature                                   |
| 🚧     | Work in progress — a scaffold or a sketch     |
| ♻️     | Refactor (no user-visible behavior change)    |
| 🔥     | Remove dead / vestigial code                  |
| 🐛     | Bug fix                                       |
| 📝     | Docstrings, README, CHANGELOG                 |
| ✏️     | Typo or other small text fix                  |
| 💄     | Style (no behavior change)                    |
| ✅     | Tests added or updated                        |
| ⚡     | Performance optimization                      |
| 🏷️     | Type-hint-only change                         |
| 💬     | Code comment                                  |
| 🗑️     | Deprecation signal                            |
| 📦     | Re-export / packaging structure               |
| 🚚     | Move / rename files                           |
| ➕     | Add a dependency                              |
| ⬆️     | Bump a dependency or tool version             |
| 🔧     | Config (`pyproject.toml`, `ruff`, `ty`, ...)  |
| 🔖     | Release a version (commit + matching tag)     |

Keep commits single-purpose; don't rewrite published history; don't
skip git hooks. AI assistants append a `Co-Authored-By` trailer with
their own published identity (e.g. `Claude <noreply@anthropic.com>`).

## Template

Generated from the `pytorch` variant of a copier template.
`.copier-answers.yml` records the answers; run
`copier update --UNSAFE --vcs-ref pytorch` to pull later template
changes. `--vcs-ref pytorch` is required — the variant lives on a
branch, so `copier update` would otherwise jump to the latest Git tag
(which belongs to the base template). Re-run `copier` to switch the
compute backend, or edit the index URL in `pyproject.toml`.

---

<!-- Add project-specific guidance below. -->
