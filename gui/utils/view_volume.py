"""Bounded, scale-independent sampling of the common camera field of view.

The result is an approximate axis-aligned bounding box, not a guarantee that
every point inside it is visible. Solver failures are missing evidence.
"""

from dataclasses import dataclass

import cv2
import numpy as np


OUTSIDE, VISIBLE, UNKNOWN = 0, 1, 2


class CameraVisibility:
    """Prepare cameras once and classify points without changing camera models."""

    def __init__(self, cameras):
        if len(cameras) < 2:
            raise ValueError("At least two valid cameras are required.")
        self.cameras = []
        self.lpt = None
        for cam in cameras:
            model = str(cam.get("model", "PINHOLE")).strip().upper()
            h, w = cam.get("h", 0), cam.get("w", 0)
            if not np.isfinite([h, w]).all() or min(h, w) <= 0:
                raise ValueError("A camera has missing or invalid image dimensions.")
            if model == "PINPLATE":
                try:
                    if self.lpt is None:
                        import pyopenlpt
                        self.lpt = pyopenlpt
                    obj = self.lpt.Camera(cam["cam_file_path"])
                    if not hasattr(obj, "projectBatchStatus"):
                        raise ValueError("projectBatchStatus is unavailable")
                except Exception as exc:
                    raise ValueError(f"Cannot load a refractive camera: {exc}") from exc
                self.cameras.append((model, h, w, obj))
            elif model == "PINHOLE":
                try:
                    K = np.asarray(cam["K"], dtype=float).reshape(3, 3)
                    rvec = np.asarray(cam["rvec"], dtype=float).reshape(3)
                    tvec = np.asarray(cam["tvec"], dtype=float).reshape(3)
                    dist = np.asarray(cam.get("dist", np.zeros(5)), dtype=float)
                    if (not all(np.isfinite(a).all() for a in (K, rvec, tvec, dist))
                            or K[0, 0] == 0 or K[1, 1] == 0):
                        raise ValueError("non-finite or invalid camera parameters")
                    R, _ = cv2.Rodrigues(rvec)
                except Exception as exc:
                    raise ValueError(f"Invalid pinhole camera: {exc}") from exc
                self.cameras.append((model, h, w, (K, rvec, tvec, dist, R)))
            else:
                raise ValueError(f"Unsupported camera model: {model}")

    def __call__(self, points):
        outside = np.zeros(len(points), dtype=bool)
        unknown = np.zeros(len(points), dtype=bool)
        world_points = None
        for model, h, w, camera in self.cameras:
            state = np.full(len(points), UNKNOWN, dtype=np.uint8)
            if model == "PINPLATE":
                try:
                    if world_points is None:
                        world_points = [self.lpt.Pt3D(*map(float, p)) for p in points]
                    statuses = camera.projectBatchStatus(world_points, False)
                    if len(statuses) != len(points):
                        raise ValueError("Incomplete projection batch")
                    for i, status in enumerate(statuses):
                        if status[0]:
                            u, v = float(status[1][0]), float(status[1][1])
                            if np.isfinite([u, v]).all():
                                state[i] = VISIBLE if 0 <= u < w and 0 <= v < h else OUTSIDE
                except Exception:
                    # A batch failure supplies no visibility evidence. In
                    # particular, never fall back to a pinhole projection.
                    state.fill(UNKNOWN)
            else:
                K, rvec, tvec, dist, R = camera
                depth = (points @ R.T + tvec)[:, 2]
                state[depth <= 0] = OUTSIDE
                try:
                    image, _ = cv2.projectPoints(points, rvec, tvec, K, dist)
                    uv = image.reshape(-1, 2)
                    valid = np.isfinite(uv).all(axis=1) & (depth > 0)
                    inside = ((uv[:, 0] >= 0) & (uv[:, 0] < w)
                              & (uv[:, 1] >= 0) & (uv[:, 1] < h))
                    state[valid] = np.where(inside[valid], VISIBLE, OUTSIDE)
                except cv2.error:
                    pass
            outside |= state == OUTSIDE
            unknown |= state == UNKNOWN
        # One definite exclusion is sufficient, even if another camera failed.
        return np.where(outside, OUTSIDE, np.where(unknown, UNKNOWN, VISIBLE))


@dataclass
class GridScan:
    minimum: np.ndarray | None
    maximum: np.ndarray | None
    possible_minimum: np.ndarray
    possible_maximum: np.ndarray
    step: np.ndarray
    layers: np.ndarray
    touch_lower: np.ndarray
    touch_upper: np.ndarray
    visible_count: int
    unknown_count: int
    unknown_minimum: np.ndarray
    unknown_maximum: np.ndarray
    samples: int


@dataclass
class VolumeEstimate:
    minimum: np.ndarray | None = None
    maximum: np.ndarray | None = None
    step: np.ndarray | None = None
    confident: bool = False
    reason: str = "No resolved common field of view was found."
    samples: int = 0


def scan_grid(classify, lower, upper, requested_step, max_samples, chunk_size=8192):
    """Honor the requested resolution or fail the budget; never coarsen it."""
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    requested_step = np.broadcast_to(np.asarray(requested_step, dtype=float), (3,))
    if (lower.shape != (3,) or upper.shape != (3,)
            or not np.isfinite([lower, upper, requested_step]).all()
            or np.any(upper <= lower) or np.any(requested_step <= 0) or chunk_size <= 0):
        raise ValueError("Invalid sampling bounds or grid spacing.")
    span = upper - lower
    counts_float = np.maximum(1, np.ceil(span / requested_step))
    total = float(np.prod(counts_float + 1))
    if not np.isfinite(total) or total > max_samples:
        raise ValueError("The sampling budget was reached before convergence.")
    counts = counts_float.astype(int)
    shape = counts + 1
    total = int(np.prod(shape))
    step = span / counts
    axes = [np.linspace(lower[i], upper[i], shape[i]) for i in range(3)]
    occupied = [np.zeros(n, dtype=bool) for n in shape]
    vmin = np.full(3, np.inf)
    vmax = np.full(3, -np.inf)
    umin, umax = vmin.copy(), vmax.copy()
    visible_count = unknown_count = 0
    for start in range(0, total, chunk_size):
        indices = np.column_stack(np.unravel_index(
            np.arange(start, min(start + chunk_size, total)), tuple(shape)))
        points = np.column_stack([axes[i][indices[:, i]] for i in range(3)])
        state = np.asarray(classify(points))
        if state.shape != (len(points),) or not np.isin(state, [OUTSIDE, VISIBLE, UNKNOWN]).all():
            raise ValueError("Invalid camera visibility results.")
        visible, unknown = state == VISIBLE, state == UNKNOWN
        if visible.any():
            vmin = np.minimum(vmin, points[visible].min(axis=0))
            vmax = np.maximum(vmax, points[visible].max(axis=0))
            for i in range(3):
                occupied[i][indices[visible, i]] = True
            visible_count += int(visible.sum())
        if unknown.any():
            umin = np.minimum(umin, points[unknown].min(axis=0))
            umax = np.maximum(umax, points[unknown].max(axis=0))
            unknown_count += int(unknown.sum())
    return GridScan(
        vmin if visible_count else None, vmax if visible_count else None,
        np.minimum(vmin, umin), np.maximum(vmax, umax), step,
        np.array([a.sum() for a in occupied]),
        np.array([a[0] for a in occupied]), np.array([a[-1] for a in occupied]),
        visible_count, unknown_count, umin, umax, total,
    )


def estimate_volume(classify, lower, upper, max_iterations=12, max_samples=2_000_000):
    """Refine a seed box; accept only resolved, stable, unclipped bounds.

    Each axis needs at least 32 intervals across its visible span and a
    coarse/fine comparison with a genuinely smaller effective step. Resolved
    axes retain their spacing while thin axes continue refining. Unknown
    samples at the visible boundary prevent acceptance; sparse interior
    failures (at most 5% of potentially visible samples) may be tolerated.
    """
    lower, upper = np.array(lower, dtype=float), np.array(upper, dtype=float)
    if (lower.shape != (3,) or upper.shape != (3,)
            or not np.isfinite([lower, upper]).all() or np.any(upper <= lower)):
        return VolumeEstimate(reason="Invalid initial search box.")
    step = (upper - lower) / 24
    previous = None
    confirmed = np.zeros(3, dtype=bool)
    samples = 0
    reason = "The common field of view did not converge."
    for _ in range(max_iterations):
        try:
            scan = scan_grid(classify, lower, upper, step, max_samples - samples)
        except ValueError as exc:
            return VolumeEstimate(reason=f"{reason} {exc}", samples=samples)
        samples += scan.samples
        h = scan.step
        if not scan.visible_count:
            if scan.unknown_count:
                return VolumeEstimate(reason="Projection failures leave the common field of view unknown.",
                                      samples=samples)
            # An empty grid may have missed a thin region. Refine in place;
            # emptiness supplies no evidence for expanding the search domain.
            step = h / 2
            previous = None
            confirmed[:] = False
            reason = "No common visible samples were found at the available resolution."
            continue

        if scan.touch_lower.any() or scan.touch_upper.any():
            expansion = np.maximum(4 * h, 0.25 * (upper - lower))
            lower = lower - expansion * scan.touch_lower
            upper = upper + expansion * scan.touch_upper
            # Keep the search affordable while seeking a finite boundary.
            # This is expansion, not refinement; reset convergence evidence.
            step = (upper - lower) / 24
            confirmed[:] = False
            previous = None
            reason = "Visible points still touch the search boundary; the overlap may be unbounded."
            continue

        span = scan.maximum - scan.minimum
        resolved = (scan.layers >= 4) & (span >= 32 * h * (1 - 1e-10))
        if previous is not None:
            movement = np.maximum(abs(scan.minimum - previous.minimum),
                                  abs(scan.maximum - previous.maximum))
            stable = movement <= 1.5 * previous.step
            finer = h <= 0.75 * previous.step
            confirmed = resolved & stable & (confirmed | finer)

        unknown_fraction = scan.unknown_count / (scan.visible_count + scan.unknown_count)
        boundary_unknown = scan.unknown_count and (
            np.any(scan.unknown_minimum <= scan.minimum + h)
            or np.any(scan.unknown_maximum >= scan.maximum - h))
        if confirmed.all() and unknown_fraction <= 0.05 and not boundary_unknown:
            # Boundary locations are uncertain by roughly one grid interval.
            # Pad by that numerical uncertainty, not a fixed physical length.
            return VolumeEstimate(scan.minimum - h, scan.maximum + h, h, True,
                                  "Resolved and stable across coarse/fine scans.", samples)

        if boundary_unknown or unknown_fraction > 0.05:
            reason = ("Projection failures remain near the visibility boundary or are too frequent "
                      f"({unknown_fraction:.1%} of potentially visible samples).")
            # Once geometrically resolved, more identical failure samples do
            # not improve confidence. Leave the user's settings intact.
            if confirmed.all():
                return VolumeEstimate(reason=reason, samples=samples)
        else:
            reason = "The common field of view is under-resolved or its bounds are not stable."

        # Retain UNKNOWN regions too: cropping them away would turn missing
        # evidence into a falsely confident, smaller volume.
        lower = np.maximum(lower, scan.possible_minimum - 2 * h)
        upper = np.minimum(upper, scan.possible_maximum + 2 * h)
        step = np.where(confirmed, h, h / 2)
        previous = scan
    return VolumeEstimate(reason=reason, samples=samples)
