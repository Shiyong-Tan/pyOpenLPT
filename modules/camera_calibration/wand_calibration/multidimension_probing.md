
# Multidimensional Probing Strategy for Global Camera Calibration Search

## Purpose

This document defines an improved probing strategy for estimating parameter scales used in global optimization
(CMA‑ES or other evolutionary methods) for refractive camera calibration.

The goal is to estimate realistic basin scales around a reference solution produced by bundle adjustment (BA).
Simple 1D probing tends to severely underestimate useful step sizes because parameters in camera calibration
are strongly coupled. The method described here combines:

- 1D probing
- multidimensional block probing
- limited local compensation optimization

to produce robust parameter scales for global search.

---

# 1. Reference State

All probing begins from a reference solution:

θ_ref

This reference state must come from a consistent BA snapshot, including:

- camera extrinsics
- refractive plane parameters
- media parameters
- any other active calibration parameters

The reference solution must be geometrically valid.

Define baseline residual metrics:

Ray_RMSE_ref  
Wand_RMSE_ref

These are used as stopping criteria during probing.

---

# 2. Overview of Probing Stages

The probing process consists of three stages:

1. 1D probing (baseline per‑parameter scale)
2. Block directional probing (detect coupled basin directions)
3. Scale merging (combine results into final parameter scales)

---

# 3. Stage 1 — 1D Probing

For each parameter p_i:

θ_test = θ_ref + α e_i

where e_i is the unit vector for parameter i.

α increases progressively:

α ∈ {0.5, 1, 2, 4, 8, ...}

For each α:

1. Construct θ_test
2. Evaluate objective after small compensation optimization
3. Check stopping criteria

Stop when:

Ray_RMSE ≥ 1.1 × Ray_RMSE_ref

or geometry becomes invalid.

The largest valid step defines:

scale_1d(i)

---

# 4. Stage 2 — Block Directional Probing

Because calibration parameters are strongly coupled, we probe directions involving multiple parameters.

Typical blocks include:

Plane Depth Block

[plane_d, cam0_tz, cam1_tz, cam2_tz, ...]

Plane Tilt Blocks

[plane_a, cam_rx]  
[plane_b, cam_ry]

These reflect typical geometric couplings between planes and camera poses.

---

# 5. Direction Set

For each block we probe multiple directions.

Example for plane depth block:

v1 = plane_d only  
v2 = camera tz only  
v3 = plane_d − average(cam_tz)

Each direction is normalized before probing.

---

# 6. Directional Probing Procedure

For each direction v:

θ_test = θ_ref + α v

α increases exponentially:

α ∈ {0.5, 1, 2, 4, 8, ...}

At each step:

1. Build candidate state
2. Apply small compensation optimization
3. Evaluate residuals

Stopping criteria:

Ray_RMSE ≥ 1.1 × Ray_RMSE_ref  
or geometry invalid.

Let α* be the largest safe value.

---

# 7. Small Compensation Optimization

When evaluating θ_test we allow limited local compensation to avoid overestimating residuals.

Rules:

Block parameters being probed are locked.

Other parameters may move slightly.

Optimization runs for:

2–3 Gauss‑Newton or LM iterations.

Purpose:

Allow natural system compensation without performing a full BA.

---

# 8. Distributing Block Scale to Parameters

Given direction:

v = [v1, v2, ..., vk]

and scale α*

Each parameter receives:

scale_block(i) = |v_i| × α*

This converts the directional scale into per‑parameter scale estimates.

---

# 9. Final Parameter Scale

For each parameter i:

effective_scale(i) = max( scale_1d(i), scale_block(i) )

This ensures:

- individual safe step sizes
- larger scales along coupled directions

---

# 10. Practical Considerations

Geometry Validation

Candidate solutions must satisfy all geometric constraints:

- valid line‑of‑sight intersections
- valid refraction geometry
- cameras remain on correct side of planes
- 3D points remain observable

Invalid states terminate probing along that direction.

---

Compensation Iteration Limit

Compensation optimization must remain small:

2–3 iterations maximum.

This prevents probing from turning into a full BA.

---

Direction Normalization

All probing directions should be normalized before scaling to ensure consistent α interpretation.

---

# 11. Expected Benefits

Compared with naive probing, this method:

- captures real basin widths
- accounts for parameter coupling
- prevents artificially small CMA‑ES scales
- improves ability to escape local minima

---

# 12. Output

The probing stage outputs:

effective_scale(i) for all parameters.

These scales are then used to normalize parameters for global search.

Normalized variables:

x_norm = (θ − θ_ref) / effective_scale

The global optimizer searches in normalized space.

