# Test Plan to Verify Whether Different Plane-Position Initializations Lead to Different Solutions

## Objective

Before implementing a large global-search modification, first verify the following hypothesis:

> Different initial values of plane position can lead the current alternating optimization to converge to different final solutions.

If this hypothesis is true, then a global search over plane positions is justified.  
If this hypothesis is false, then the benefit of adding a global-search layer may be limited.

---

# 1 Core Idea

Keep everything identical except for the **initial plane position**.

For each trial:

- use the same dataset
- use the same pinhole camera initialization
- use the same initial plane angle
- use the same optimizer settings
- use the same stopping criteria
- change only the initial position of one plane

Then compare the final optimized results.

---

# 2 Recommended First Test

## Start with a single plane

Even if the full system contains multiple planes, the first test should be kept as simple as possible.

Recommended setup:

- choose one important plane
- allow only this plane position to vary in initialization
- keep all other plane parameters fixed at the current best estimate
- run the existing alternating optimization exactly as it is now

This test is not meant to solve the full multi-plane problem.  
It is only meant to check whether **plane-position initialization changes the final basin of convergence**.

---

# 3 Test Design

## Step 1 Define a physically reasonable initialization range

Choose a range for the plane position using a physically meaningful prior, for example:

- between the camera pinhole and the reconstructed 3D points
- or between the camera pinhole and a representative point-cloud depth

Pick several initial positions inside this range.

Recommended:

- 7 to 9 initial positions
- approximately uniform coverage from near-camera to near-object

Example:

p0^(1), p0^(2), ..., p0^(7)

---

## Step 2 Run the current alternating optimization for each initialization

For each initial plane position:

- keep all other initial parameters unchanged
- run the full current alternating optimization
- do not add DE
- do not add trust-region constraints
- do not modify the current local optimizer

The purpose is to test the **current pipeline sensitivity** to plane-position initialization.

---

## Step 3 Record the final results

For each run, record at least the following:

### Required
- initial plane position
- final plane position
- final plane angle
- final reprojection error / final cost

### Strongly recommended
- final camera extrinsic correction
- final camera intrinsic correction
- wand length consistency
- residual mean/std for each camera
- number of iterations until convergence

---

# 4 How to Decide Whether the Solutions Are Different

Do not look only at the final cost.

Two runs should be considered different if one or more of the following are true:

- final plane position differs significantly
- final plane angle differs significantly
- final camera parameters differ significantly
- final reprojection error differs significantly

A particularly important case is:

> final cost is similar, but the optimized parameters are different

This suggests the existence of multiple plausible local minima or compensating solutions.

---

# 5 Recommended Plots

## Plot 1: Initial position vs final position

- x-axis: initial plane position
- y-axis: final plane position

Interpretation:

- if all initial points converge to the same final value, the system may have only one dominant basin
- if different initial points converge to different clusters, multiple basins likely exist

---

## Plot 2: Initial position vs final cost

- x-axis: initial plane position
- y-axis: final reprojection error

Interpretation:

- if some initializations consistently end at higher final cost, this is evidence of local minima
- if multiple distinct final solutions have similar cost, this suggests parameter degeneracy or multiple comparable basins

---

# 6 Stronger Version of the Test

If the first test already suggests multiple basins, a stronger follow-up experiment is recommended.

## Two-dimensional sensitivity test

For each plane-position initialization, also test a few different plane-angle initializations.

Example:

- 7 position initializations
- 3 angle initializations
- total = 21 runs

This helps answer the question:

> Is plane position more important than plane angle in determining the final convergence basin?

If position changes produce different final solutions while angle changes are mostly recovered during optimization, this strongly supports using plane position as the outer global-search variable.

---

# 7 Practical Thresholds

To avoid subjective interpretation, define thresholds in advance.

Example criteria for calling two solutions different:

- final plane position difference > 1 mm
- or final plane angle difference > 0.1° or 0.2°
- or final camera extrinsic difference above a chosen tolerance
- or final reprojection error difference above a meaningful tolerance

The exact thresholds should reflect the physical scale and calibration precision of the system.

---

# 8 Minimum-Cost Quick Test

If you want the fastest possible first validation, use this version:

- choose 1 plane
- choose 7 different initial positions
- keep all other initial values fixed
- run 7 full alternating optimizations
- record final position, angle, and cost
- make the two diagnostic plots

This is usually enough to determine whether different initial plane positions lead to different final solutions.

---

# 9 Repeatability Check

If the optimizer contains any randomness, repeat each run 2–3 times.

This helps separate:

- variability caused by random effects
from
- variability caused by different initial plane positions

If repeated runs from the same initialization always end at the same solution, then the observed differences are likely due to initialization basin effects rather than stochastic noise.

---

# 10 Possible Outcomes and Their Meaning

## Outcome A: All initializations converge to essentially the same final solution

Interpretation:

- plane-position initialization may not be the main issue
- a global-search layer over plane position may offer limited improvement

---

## Outcome B: Different initializations converge to several distinct clusters

Interpretation:

- multiple local minima likely exist
- plane-position initialization affects the basin of convergence
- adding a global-search layer is justified

---

## Outcome C: Final cost is similar, but optimized parameters are different

Interpretation:

- parameter compensation or degeneracy is likely present
- the problem may involve both local minima and identifiability issues

---

# 11 Recommended Next Step After This Test

If the test confirms that different plane-position initializations lead to different final solutions, then the next step should be:

- introduce an outer global-search layer over plane positions
- keep angle/camera refinement inside the inner alternating solver
- later add trust-region constraints to preserve basin identity during candidate evaluation

If the test does not confirm this behavior, then first investigate:

- model mismatch
- insufficient data coverage
- parameter degeneracy
- poor identifiability

before implementing a large global-search modification.

---

# 12 Summary

This test is designed to answer one key question:

> Does changing only the initial plane position cause the current alternating calibration algorithm to converge to different final solutions?

A simple single-plane initialization sweep is usually enough to answer this.

If yes, then plane-position global search is well motivated.  
If no, then the main limitation may lie elsewhere.

---
