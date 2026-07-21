# Changelog

All notable changes to this project will be documented in this file.

The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `iivs-lib[torch]>=0.2.0` as a dependency, for phase sequence IO. The `torch`
  extra additionally enables `iivs.dhm.analysis.pytorch` and
  `iivs.common.data.pytorch`.

### Changed

- `backward_warp` / `BackwardWarp` sample at `grid + offset` instead of
  `grid - transform`, and the second parameter is renamed accordingly. A forward
  optical flow is now passed unchanged rather than negated at every call site,
  which removes the bare sign flip that this project has already been bitten by;
  `offset` rather than `flow` keeps `common/` free of optical-flow vocabulary
  while stating the sign in the name. To displace an image *by* a field, negate
  it.
- `warp_consistency` / `WarpConsistency` now reconstruct `frame1` by sampling
  `frame2` at `grid + flow`, instead of reconstructing `frame2` from `frame1` at
  `grid - flow`. The forward flow is defined on `frame1`'s grid, so this
  direction needs no inverse and is exact; the previous one approximated the
  inverse with an error growing as `|flow| * |grad flow|`. Scores shift only in
  the 5th decimal at sub-pixel motion, and estimator rankings are unchanged.
