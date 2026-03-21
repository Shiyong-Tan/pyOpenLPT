---
description: refracive camera calibration
---

# Refractive Camera Calibration Workflow

## Description
End-to-end refractive wand calibration with strict physical feasibility,
including PR5 violation resolution and optional polish of frozen parameters.

---

## Step 1 — Load Dataset & Sanity Checks
- Load all frames and observations.
- Verify:
  - All cameras observe A and B points in all frames.
  - Total frames count is consistent across cameras.

Abort if any visibility violation exists.

---

## Step 2 — Bootstrap (Pinhole)
- Run pinhole bootstrap calibration.
- Recover:
  - Relative extrinsics
  - Scale from wand length
- Cache results.

---

## Step 3 — Refractive Plane Initialization (P1)
- Initialize refractive planes.
- Accept that:
  - Single-camera windows may have inaccurate plane distance.
- Do NOT attempt to force physical accuracy here.

---

## Step 4 — PR4 Selective BA
- Build observability table.
- Optimize only parameters with sufficient baseline/angle.
- Freeze others strictly.

---

## Step 5 — PR5 Main Optimization (Violation Resolution Stage)

### Phase 5.1 — Standard PR5 Rounds
- Enable side-gate.
- Use default residual construction.
- Allow subsampling or worst-K for speed.

### Phase 5.2 — Violation-Driven Round (MANDATORY)
- Identify worst violating points (FULL dataset).
- Inject worst-K frames (K ≤ 10% of total frames).
- Gate MUST be enabled.
- Goal:
  - Reduce violations to ZERO.

### Phase 5.3 — Final PR5 Round (Hard Check)
- Use:
  - Full dataset OR
  - Full worst-K set
- Gate enabled.
- Accept result ONLY if:
  - FULL dataset audit reports 0 violations.

If violations remain → STOP and report.

---

## Step 6 — PR5_POLISH (Optional)

### Eligibility Check
- Identify cameras with FREEZE parameters in PR5 table.
- If none → SKIP polish.

### Per-Camera Sequential Polish
For each eligible camera:
- Unlock ONLY frozen parameters.
- Keep all others locked.

#### Bounds
- rvec: ±10 degrees
- tvec: ±5 mm
- focal length: ±1%
- plane normal: ±5 degrees
- plane distance: ±5 mm

Gate:
- Gate remains ACTIVE
- No relaxation allowed

Reject polish if:
- Any violation appears
- RMSE degrades significantly

---

## Step 7 — Final Audit & Export
- Run FULL dataset audit:
  - Ray RMSE
  - Length RMSE
  - Violation count
- Export camFiles only if:
  - Violation count == 0

---

## Completion Criteria
Calibration is considered VALID only if:
- Physical feasibility is satisfied
- Violations == 0
- Optimized parameters respect observability logic
