# Changelog

All notable changes to this project will be documented in this file.

The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `iivs-lib[torch]>=0.2.0` as a dependency. Beyond phase sequence IO, the
  `torch` extra enables `iivs.dhm.analysis.pytorch` (`phase_to_opd`,
  `calc_drymass`) and `iivs.common.data.pytorch` (masked `Mean` / `Variance` /
  `Norm` reductions) — tensor-in/tensor-out twins that preserve device and
  autograd.

### Changed

- Design docs folded into `docs/foundations.md` and `TODO.md`; the kinematic
  kernel sketches are channel-first, matching the settled CHW layout.
