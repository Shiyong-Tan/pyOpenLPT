# Refractive Wand Calibration Algorithm Documentation

This document describes the refractive wand calibration pipeline implemented in:

- `refraction_wand_calibrator.py`
- `refraction_calibration_BA.py`

The pipeline calibrates cameras that view through refractive interfaces (air -> window -> water) and outputs PINPLATE camFiles plus triangulation reports.

## Overview

The refractive calibrator extends standard wand calibration to handle multi-layer refraction. It estimates and refines:

- Camera extrinsics (rvec, tvec)
- Camera intrinsics (f, cx, cy, k1, k2)
- Window plane geometry (plane_pt, plane_n)
- Window thickness

Key ideas:

- C++ kernel traces refracted rays (Snell's Law)
- Optimization uses ray distance + wand length residuals
- Window orientation is enforced by camera/object side constraints
- Final Round 4 refines intrinsics and thickness jointly with geometry

## Optical Model

Rays pass through three media:

```
Camera (Air, n1=1.0) -> Window (Glass/Acrylic, n2~1.49) -> Water (n3~1.33)
```

Window parameters per window:

- `plane_pt`: a point on the closest interface (air side)
- `plane_n`: unit normal pointing away from camera toward the object
- `thickness`: window thickness in mm
- `n_air`, `n_window`, `n_object`

## Pipeline Phases

### Phase 0: Pinhole Bootstrap (P0)

Goal: estimate initial camera extrinsics from a pinhole model (no refraction).

- Select best camera pair (max shared frames)
- Essential matrix + pose recovery (seed pair)
- Incremental registration (PnP) for other cameras
- Intrinsics are frozen to UI values
- Outputs: `cam_params` and scaled 3D points for all frames

Bootstrap cache:

- File: `bootstrap_cache.json`
- Validated by camera IDs, frame count, wand length

### Phase 1: Window Plane Initialization

Goal: initialize plane geometry from bootstrapped camera centers and 3D wand midpoints.

- Group cameras by window (active vs inactive)
- Estimate normal from camera optical axes + object direction
- Compute a safe plane distance d0
- Enforce camera-side and object-side sign conventions

### Phase 2/3: Ray Building (C++ Kernel)

Goal: build refracted rays using `pyopenlpt.Camera` PINPLATE model.

- C++ ray tracing is the authority
- Ray validity is tracked and summarized

### Phase 4: Bundle Adjustment (Selective BA)

Goal: refine window planes and camera extrinsics with staged constraints.

Loop (up to 6 passes):

- Step A: optimize planes only (strict angle bounds, weak-window bounds)
- Step B: optimize camera extrinsics only (free bounds)
- Early stop when plane angle constraints are inactive

Final joint (Round 3):

- Optimize planes + extrinsics with moderate bounds

Round 4 (new):

- Optimize planes + extrinsics + focal length + window thickness
- Tight bounds: plane angle +/- 2.5 deg, plane distance +/- 5 mm, rvec +/- 5 deg
- Intrinsics and thickness bounded by percentage (default 5%)

Residuals:

```
J = S_ray + lambda * S_len + regularization
```

- `S_ray`: sum of squared point-to-ray distances
- `S_len`: sum of squared wand length errors
- `lambda = 2 * N_cams` (forced to 1.0 if S_ray/S_len < 10)

Side constraints:

- Object points must lie on positive side of the plane
- Cameras must lie on negative side

Cache:

- File: `bundle_cache.json`
- Stores planes, cam_params, window_media, optional points_3d

### Phase 5: Alignment, Export, Verification

- Coordinate alignment (Y axis from plane intersection)
- Sync C++ state and save cache
- Export PINPLATE camFiles (plane shifted to farthest interface)
- Triangulation and residual report (JSON)
- Plane side verification
- Final close-loop verification
- Per-frame reprojection error for UI

## Output Artifacts

- `camFile/` PINPLATE camera files
- `triangulation_report.json` (residuals, worst frames, samples)
- Updated cache files

## Related Files

- `refraction_wand_calibrator.py`: pipeline orchestration
- `refraction_calibration_BA.py`: bundle adjustment implementation
- `refractive_bootstrap.py`: P0 bootstrap
- `refractive_geometry.py`: ray tracing helpers and geometry

## Algorithm Flow Diagram

```mermaid
graph TD
    A[Start] --> B[P0: Pinhole Bootstrap]
    B --> C{Bootstrap Cache Valid?}
    C -->|Yes| D[Load Bootstrap Cache]
    C -->|No| E[Run Bootstrap]
    D --> F[Plane Initialization]
    E --> F
    F --> G[Build Refracted Rays]
    G --> H[Selective BA Loop]
    H --> I[Final Joint (Round 3)]
    I --> J[Round 4: Intrinsics + Thickness]
    J --> K[Alignment + Export]
    K --> L[Triangulation Report + Verification]
    L --> M[End]
```

## Notes

- The C++ PINPLATE kernel is the source of truth for ray tracing and triangulation.
- All calibration steps assume wand length is correct and consistent.
- Side constraint violations are reported with counts for debugging.
