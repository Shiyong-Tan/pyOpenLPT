Full Global Search Plan for Refractive Camera Calibration

1. Objective

The purpose of the full global search stage is to reduce the risk of getting trapped in poor local minima during refractive camera calibration. The global search stage does **not** replace the existing bundle adjustment (BA) pipeline. Instead, it generates globally explored candidate solutions that are later refined by the current BA procedure.

The final workflow is:

```text
reference calibration
    ↓
1D probing for parameter scaling
    ↓
normalized parameterization
    ↓
CMA-ES global search
    ↓
top-K candidate solutions
    ↓
existing BA refinement
    ↓
final best calibration

The optimization target in the global stage is the same physical target used in calibration quality assessment:

minimize Ray RMSE

while respecting geometric feasibility and refractive-side constraints

2. Inputs and Existing Assets

The full global search should reuse the current calibration codebase and existing camera/plane representations as much as possible.

2.1 Existing BA code

The global search must be compatible with the current BA implementation and must produce candidate parameters that can be directly passed into the existing refinement pipeline.

2.2 Camera parameter input

The camera initialization for the global search will be read from the camFile folder.

The folder contains the camera parameter files.

Each file can be read using the existing lpt.camera object.

These camera files should be treated as the reference camera parameters for initializing the global search variables.

In practice, the global search should begin from the current/reference calibration state reconstructed from:

the plane parameters used by the current refractive calibration

the camera parameters loaded from camFile through lpt.camera

2.3 Output of global search

The global stage should output:

the best objective value found during the global stage

the top-K candidate solutions

per-candidate diagnostic statistics

candidate parameter sets that can be passed directly into BA refinement

3. Global Search Variables

The first implementation will use Parameter Set A:

all plane parameters
+
all camera extrinsics
3.1 Plane parameters

Each plane has exactly 3 DOF.

The global search should use the same plane parameterization as the existing BA code. That means:

two tangent-space parameters for plane normal perturbation

one depth/distance parameter relative to the anchor representation already used in BA

So each plane is parameterized as:

[a, b, d]

where:

a, b are tangent-space normal perturbations

d is the signed depth/distance perturbation in the same representation used in BA

The global search should not switch to an alternative plane representation such as raw point-plus-normal. It should reuse the BA representation to ensure:

no extra redundancy

no unit-normal constraint issues

direct compatibility with BA refinement

3.2 Camera extrinsics

Each camera contributes 6 DOF:

[ωx, ωy, ωz, tx, ty, tz]

where:

ωx, ωy, ωz define a rotation increment around the reference pose

tx, ty, tz define a translation increment around the reference pose

The camera update should follow a reference-pose increment form:

R = Exp(ω) · R_ref
t = t_ref + Δt

This means the global search variables for each camera are perturbations around the reference camera parameters loaded from the camFile directory.

3.3 Total dimension

If there are:

N_plane planes

N_cam cameras

then the total CMA-ES dimension is:

d = 3 * N_plane + 6 * N_cam

This dimension determines the CMA-ES population size and approximate evaluation budget.

4. Reference Solution

A reference solution θ_ref is required before global search begins.

This reference solution contains:

the reference plane parameters in the same BA-compatible representation

the reference camera extrinsics loaded from the camFile folder via lpt.camera

All global-search variables will be defined as perturbations around this reference solution.

The reference solution may come from:

the current best refractive calibration result, or

the current initialized state before refinement

For the first version, the recommendation is to use the current available reference calibration state that is closest to the current working solution.

5. Parameter Scaling via 1D Probing

Before running CMA-ES, every search variable must be normalized so that all dimensions have comparable scale.

5.1 Why scaling is needed

The variables in this problem live in different physical units and sensitivities:

plane tangent perturbations

plane depth

camera rotation increments

camera translation increments

Without normalization, CMA-ES may waste effort because some variables are effectively too large while others are too small.

5.2 1D probing procedure

For each variable, perform a local 1D probe around the reference solution:

θ = θ_ref + δ e_i

where:

e_i is the unit direction for parameter i

δ is a small trial perturbation

Evaluate Ray RMSE for several trial magnitudes of δ on both positive and negative sides.

Define the scale scale_i as the smallest perturbation magnitude such that:

RMSE(θ_ref + δ e_i) ≈ 1.05 – 1.10 × RMSE_ref

In words:

the normalization scale is the smallest perturbation size that causes a noticeable change in the objective.

5.3 Normalized variables

After the scales are computed, define normalized CMA-ES variables:

x_i = (θ_i - θ_ref_i) / scale_i

so that:

θ_i = θ_ref_i + scale_i * x_i

This makes |x_i| ≈ 1 correspond to a meaningful objective change.

5.4 Search bounds in normalized space

Use a common bound for all normalized variables:

x_i ∈ [-3, 3]

This means each physical parameter is allowed to move approximately within:

θ_i ∈ θ_ref_i ± 3 * scale_i

This is large enough to explore beyond the immediate local basin but still tied to the local sensitivity of the problem.

6. Objective Function

The global search objective should combine:

Ray RMSE

invalid-ray penalty

LOS failure penalty

geometry-side penalty

The final objective is:

f = RMSE_valid
  + penalty_invalid
  + penalty_LOS
  + penalty_geom
6.1 RMSE term

RMSE_valid is the Ray RMSE computed on the valid rays/observations that can be evaluated under the candidate geometry.

6.2 Invalid-ray penalty

If some rays cannot be evaluated because of geometric or numerical failure, define:

invalid_fraction = invalid_rays / total_rays

and add

penalty_invalid = λ_invalid * invalid_fraction
6.3 LOS failure penalty

If LOS or related refractive path checks fail for some observations, define:

LOS_fraction = failed_LOS / total_rays

and add

penalty_LOS = λ_LOS * LOS_fraction
7. Geometry-Side Penalty

This is a critical part of the plan.

The global search must preserve the same geometric side logic already used in the BA code.

7.1 Required geometry conditions

For every plane:

all reconstructed/used 3D points must remain on the observation side

all camera centers must remain on the camera side

7.2 Signed distances

For a plane with normal n and point P_plane:

sX = dot(n, X - P_plane)
sC = dot(n, C - P_plane)

where:

X is a 3D point

C is a camera center

7.3 Required inequalities

Using the current side convention:

sX ≥ margin_obj + r_ball
sC ≤ -margin_cam

That is:

3D points must lie safely on the observation side

cameras must lie safely on the camera side

7.4 Violation gaps

Define gap values:

gapX = (margin_obj + r_ball) - sX
gapC = margin_cam + sC

If gapX > 0, the point-side constraint is violated or too close.
If gapC > 0, the camera-side constraint is violated or too close.

7.5 Barrier-style penalty

The global search should follow the same spirit as the BA penalty implementation and use a barrier-like penalty rather than immediate rejection for mild violations.

For each violated gap, use a combination of:

a smooth step/barrier term

a linear gradient-like term

That is, conceptually:

pen_obj += step(gapX) + grad(gapX)
pen_cam += step(gapC) + grad(gapC)

Then:

penalty_geom = pen_obj + pen_cam

The exact functional form should follow the current BA implementation as closely as possible, so that global search and BA use consistent feasibility logic.

7.6 Why barrier penalty instead of immediate rejection

Mildly infeasible candidates should not immediately receive the same score as catastrophic failures. The optimizer should be able to distinguish:

nearly feasible candidates

moderately infeasible candidates

completely invalid candidates

This is especially important for CMA-ES, because it learns from relative ranking.

8. Hard Failures

Some candidates should bypass soft penalties and receive a hard failure score.

A candidate should be marked as a hard failure if any of the following occurs:

NaN or infinite objective value

extremely small valid-ray fraction

completely broken refractive geometry

severe camera/plane side violation

numerical failure that makes the candidate non-interpretable

For such cases:

f = P_hard

where P_hard is a large penalty constant.

9. Penalty Weight Initialization

Let:

R0 = RMSE_reference

Use the following first-version settings:

λ_invalid = 3 * R0
λ_LOS     = 3 * R0
λ_geom    = 3 * R0   (through the geometry barrier scaling)
P_hard    = 20 * R0

The goal is:

mild violations affect ranking

moderate violations are clearly worse

catastrophic failures are strongly separated

These weights do not need to be perfectly tuned initially. They only need to place the penalties on a comparable scale to meaningful RMSE differences.

10. Global Optimizer: CMA-ES

The first implementation will use CMA-ES.

10.1 Why CMA-ES

CMA-ES is chosen because:

the search space is continuous

parameters are strongly coupled

gradients are not reliable in the global regime

the objective may be non-smooth because of validity switching and path failures

CMA-ES can adapt covariance structure automatically

10.2 CMA-ES search space

CMA-ES operates only in the normalized variable space:

x = 0  → reference solution

The optimizer never directly manipulates raw physical parameters. Instead, physical parameters are reconstructed from normalized variables before evaluation.

10.3 Initial settings

Use:

initial mean   x0 = 0
initial sigma  σ0 = 0.6
bounds         [-3, 3] for all dimensions
11. Population Size

For dimension d, use the standard CMA-ES population-size rule:

λ ≈ 4 + 3 ln(d)

Then round to a practical value for implementation and parallel evaluation.

Recommended first-version value:

population size λ = 16

This is a good default for typical calibration dimensions and is simple for batching/parallel evaluation.

12. Evaluation Budget

The first implementation should use:

budget per run = 2500 evaluations

This is large enough for a meaningful first test while remaining computationally manageable.

With population size λ = 16, this corresponds to approximately:

about 150 generations
13. Multiple Independent Runs

The global search should not be trusted from a single CMA-ES run.

Use:

3 to 5 independent runs

with different random seeds.

Reason:

different seeds may discover different basins

repeated runs help assess search stability

repeated runs help detect whether the problem is genuinely multimodal

For the first version, a good setting is:

5 runs

if the runtime is acceptable; otherwise start from 3.

14. Termination Criteria

Each CMA-ES run should stop when any one of the following conditions is satisfied:

14.1 Maximum evaluation budget reached
evaluations ≥ 2500
14.2 Objective stagnation

If the best objective improvement is negligible for too many generations, stop early.

Recommended first-version rule:

no meaningful improvement for 25 generations
14.3 Step-size collapse

If the CMA-ES step size becomes too small in normalized space, stop.

Recommended threshold:

σ < 1e-3

These criteria prevent wasteful evaluations once the run has clearly converged or stalled.

15. Full-Dataset Evaluation

For the first implementation, the global search should evaluate the objective using the full observation set, not a subsampled subset.

Reason:

candidate ranking should be as faithful as possible to the true objective

subsampling may distort ranking and mislead CMA-ES

the global stage evaluates Ray RMSE only, which is usually cheaper than BA

Therefore:

use full observations in global search evaluation

If runtime later becomes prohibitive, approximation strategies can be revisited, but not in the first implementation.

16. Candidate Selection After Global Search

At the end of all global-search runs, collect the candidate solutions and keep the top-K according to the global objective.

Recommended value:

K = 5 to 10

A practical first choice is:

K = 5

These candidates should include:

the best global objective values

enough diversity to allow BA refinement to reveal basin differences

If possible, near-duplicate candidates should be filtered so that Top-K is not filled with essentially identical solutions.

17. BA Refinement Stage

Each selected global-search candidate must be passed into the existing BA pipeline without changing the BA logic.

For each candidate:

convert normalized variables back to physical plane/camera parameters

build the BA-compatible parameter representation

run the current calibration refinement pipeline:

alternating optimization

joint bundle adjustment

The BA stage remains the final precision-refinement stage.

The global search only provides improved initial seeds.

18. Final Model Selection

After BA refinement, compare all refined candidates using the final calibration metric.

Final selection rule:

choose the refined candidate with the smallest final Ray RMSE

This is important because:

the globally best raw candidate may not produce the best refined result

BA may refine different seeds into different nearby basins

the true comparison should be based on post-refinement calibration quality

19. Diagnostics and Logging

Detailed logging is required to make the global-search behavior interpretable.

19.1 Per-evaluation logging

For each evaluated candidate, record:

run index

generation index

candidate index

normalized parameter vector

reconstructed physical parameter vector

RMSE_valid

invalid_fraction

LOS_fraction

geometry penalty

total objective

hard-failure flag

failure reason summary

19.2 Per-generation logging

For each generation, record:

best objective

median objective

feasible fraction

hard-failure fraction

current CMA-ES sigma

current best candidate ID

19.3 Final candidate logging

For each Top-K candidate, record:

candidate rank in global stage

objective components

parameter set

BA-refined RMSE

BA-refined final parameters

This logging is essential for answering:

whether the optimizer spends too much time in infeasible space

whether penalty weights are reasonable

whether multiple basins are actually being found

20. Expected Outcomes

This framework can lead to three scientifically useful outcomes.

Outcome A: Better basin found

After BA refinement, some globally searched candidate gives a noticeably lower Ray RMSE than the current pipeline.

Interpretation:

the original calibration pipeline was trapped in a poorer local minimum

global search adds practical value

Outcome B: Same basin recovered repeatedly

Global search repeatedly returns candidates that refine to the same final solution.

Interpretation:

the current calibration result is already near the dominant/global basin

a more aggressive global stage may not be necessary

Outcome C: Many similar-quality solutions exist

Global search finds many different parameter combinations with similar RMSE.

Interpretation:

the objective may be flat or partially degenerate

parameter identifiability may be limited

model/constraint design may matter more than search strategy

All three outcomes are useful and should be treated as informative results.

21. Implementation Notes

Reuse the existing BA-compatible plane parameterization exactly.

Read reference camera parameters from the camFile folder using lpt.camera.

Keep the global-search parameter representation directly convertible into BA parameters.

Use full observations in the first implementation.

Use soft barrier penalties for mild geometry violations and hard rejection only for catastrophic failures.

Keep all diagnostics, because the first successful implementation will likely require one or two rounds of penalty tuning.

22. Final Summary

The final design is:

reference planes + camera extrinsics from current calibration / camFile
    ↓
1D probing to estimate parameter scales
    ↓
normalized parameterization
    ↓
CMA-ES full global search over:
    all plane parameters
    all camera extrinsics
    ↓
objective = Ray RMSE + invalid/LOS/geometry penalties
    ↓
multiple CMA-ES runs
    ↓
top-K global candidates
    ↓
existing BA refinement
    ↓
best refined calibration

This strategy is designed to be:

compatible with the current calibration system

consistent with existing BA geometry logic

suitable for strongly coupled refractive calibration parameters

practical to implement and test in stages
