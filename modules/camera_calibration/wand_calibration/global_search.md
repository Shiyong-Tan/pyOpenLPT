
# Global Search Strategy for Refractive Camera Calibration

## Overview

This document describes a robust **global–local optimization framework** for refractive camera calibration using wand data.

The goal is to prevent the calibration algorithm from becoming trapped in **local minima** when estimating refractive plane parameters.

The strategy combines:

- **Differential Evolution (DE)** for global search
- **Alternating optimization** for local refinement
- **Trust-region constraints** to preserve basin identity
- **Physically motivated search bounds**
- **Final full joint optimization**

This approach is suitable for systems containing **1–10 refractive planes**.

---

# 1 Problem Structure

The calibration parameters can be divided into two groups.

## Global search variables

Outer optimization searches over **plane positions**:

x = [p1, p2, ..., pM]

where

- M = number of refractive planes
- pi = position of plane i along its normal direction

These parameters determine the **basin of attraction**.

---

## Local refinement variables

During inner alternating optimization we refine:

- plane orientations
- camera intrinsic parameters
- camera extrinsic parameters
- small corrections to plane positions

---

# 2 Optimization Architecture

The full calibration pipeline is:

Global Search (Differential Evolution)
        ↓
Short Alternating Optimization
        ↓
Candidate Evaluation
        ↓
Population Update
        ↓
Repeat
        ↓
Final Full Optimization

The **outer algorithm explores the global space**, while the **inner solver performs local refinement**.

---

# 3 Physically Motivated Search Bounds

To reduce unnecessary search space, plane positions should be restricted to a physically meaningful region.

A refractive plane must lie **between the camera pinhole and the observed 3D points**.

## Bound construction

Let

- C = camera pinhole center
- X_j = reconstructed 3D points
- n = current estimate of plane normal

Compute projected distances:

s_j = n · (X_j - C)

These values represent the depth of each 3D point along the plane normal.

Define search bounds:

d_min = max(ε , q_5%)
d_max = q_50% or q_70%

where

- q_k% = k-th percentile of {s_j}
- ε = small positive safety margin near the camera

This ensures the plane remains between the camera and the observed scene.

Advantages:

- avoids searching behind the camera
- avoids placing planes behind the object
- robust to outliers
- significantly reduces global search volume

---

# 4 Differential Evolution (DE)

## Population representation

Each candidate is a full plane configuration:

x_k = [p1, p2, ..., pM]

Population:

X = {x1, x2, ..., xP}

Recommended population size:

P = 20 + 5M

Example:

planes | population
2 | 30
4 | 40
6 | 50
10 | 70

---

# 5 Initialization

Sample each candidate uniformly inside the bounds:

pi ∈ [d_min , d_max]

x_k ~ Uniform(bounds)

---

# 6 Candidate Evaluation

Each candidate is evaluated using **short alternating calibration**.

## Step 1 Initialization

plane positions = x
camera parameters = initial guess
plane orientations = initial guess

---

## Step 2 Alternating Optimization

Run **2–3 alternating iterations**.

### Optimize angles and cameras

optimize plane angles
optimize camera parameters

### Optimize plane positions (trust region)

Plane updates must remain near the outer candidate:

|pi - xi| ≤ Δ

where

Δ = trust region radius

Typical values:

Δ = 2–5% of search range

Example:

search range = 0–100 mm
Δ = 2 mm

This prevents the inner solver from drifting into a different basin.

---

## Step 3 Score computation

Return the calibration error:

score = reprojection_error

Lower score indicates a better candidate.

---

# 7 Differential Evolution Update

For each candidate xi:

## Mutation

Select three distinct candidates:

xa , xb , xc

Generate mutation vector:

v = xa + F*(xb - xc)

Typical parameter:

F = 0.5

---

## Crossover

Combine mutation and current candidate:

trial = crossover(xi , v)

Typical crossover rate:

CR = 0.8

---

## Selection

Evaluate trial candidate.

if score(trial) < score(xi):
    xi = trial

Population size remains constant.

---

# 8 Stopping Criteria

Optimization stops when **both conditions are satisfied**.

## Population convergence

Normalize parameters:

zi = (xi - Li) / (Ui - Li)

Compute spread:

ri = max(zi) - min(zi)

Stop if:

max(ri) < 0.02 – 0.05

Interpretation: population has collapsed to **2–5% of the search range**.

---

## Cost plateau

Let f_best(t) be the best score at generation t.

Stop if:

|f_best(t) - f_best(t-K)| / max(1, |f_best(t)|) < 1e-3

Typical:

K = 8 – 10 generations

---

## Maximum iteration limit

Also impose hard limits:

max_generations = 40 – 60

or

max_evaluations ≈ 2000

---

# 9 Final Refinement

After global search converges:

Select the best candidates:

top 3 candidates

Then run full calibration:

remove trust region
run full alternating optimization
run final joint optimization

Select the candidate with the lowest final reprojection error.

---

# 10 Recommended Parameters

Typical configuration:

population size      P = 20 + 5M
generations          40
mutation factor      F = 0.5
crossover rate       CR = 0.8
alternating steps    2–3
trust region Δ       2–5% of search range

---

# 11 Computational Cost

Example:

M = 6 planes
population = 50
generations = 40

Total evaluations:

50 × 40 = 2000

Each evaluation runs only **short alternating optimization**, making the approach computationally manageable.

---

# 12 Advantages

This strategy provides several benefits.

Avoids local minima
Population-based search explores multiple basins simultaneously.

Avoids combinatorial explosion
No grid search over plane combinations.

Preserves basin identity
Trust-region constraints prevent inner optimization from drifting across basins.

Uses physical priors
Search bounds are constrained between camera and observed scene.

---

# 13 Summary

The calibration algorithm becomes:

Initialize population
        ↓
DE mutation & crossover
        ↓
Short alternating calibration
        ↓
Candidate scoring
        ↓
Population update
        ↓
Repeat until convergence
        ↓
Full alternating optimization
        ↓
Final joint optimization

This global–local strategy greatly improves robustness against local minima while keeping the computational cost manageable.
