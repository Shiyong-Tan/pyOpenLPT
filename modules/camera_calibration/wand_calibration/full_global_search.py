"""
Full-global-search Task 1: Reference-state reconstruction (BA snapshot).

Loads a coherent BA reference state from on-disk artifacts produced by
``RefractiveBAOptimizer``.  Two complementary sources are supported:

1. **camFiles** (``cam{id}.txt`` in PINPLATE format) — authoritative for
   ``cam_params`` and ``cam_to_window``.
2. **bundle_cache.json** — authoritative for ``window_planes`` (closest-
   interface ``plane_pt``), ``window_media``, and ``cam_params`` (redundant
   cross-check).

The public entry-point is :func:`load_reference_state`, which returns a dict
fully compatible with ``RefractiveBAOptimizer.__init__`` arguments.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  CamFile parser
# ---------------------------------------------------------------------------

def _strip_comments_and_blanks(lines: List[str]) -> List[str]:
    """Return non-comment, non-blank data lines from a PINPLATE camfile."""
    out: List[str] = []
    for raw in lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            out.append(stripped)          # keep for meta-block scanning
        else:
            out.append(stripped)
    return out


def _parse_data_lines(lines: List[str]) -> List[str]:
    """Extract ordered *data* lines (non-comment, non-blank) from camfile."""
    return [ln for ln in lines if ln and not ln.startswith("#")]


def _parse_refraction_meta(lines: List[str]) -> Dict[str, str]:
    """Extract ``BEGIN_REFRACTION_META`` … ``END_REFRACTION_META`` block."""
    meta: Dict[str, str] = {}
    inside = False
    for ln in lines:
        s = ln.strip()
        if "BEGIN_REFRACTION_META" in s:
            inside = True
            continue
        if "END_REFRACTION_META" in s:
            break
        if inside and s.startswith("#"):
            # e.g. "# CAM_ID=0"
            m = re.match(r"#\s*(\w+)=(.*)", s)
            if m:
                meta[m.group(1).strip()] = m.group(2).strip()
    return meta


def parse_camfile(path: Path) -> Dict[str, Any]:
    """Parse a single PINPLATE camfile into a structured dict.

    Returns
    -------
    dict with keys:
        cam_id      : int
        window_id   : int
        cam_params  : np.ndarray, shape (11,)
                      [rvec(3), tvec(3), f, cx, cy, k1, k2]
        plane_pt_export : np.ndarray, shape (3,)  — farthest interface
        plane_n     : np.ndarray, shape (3,)
        refract_array : list[float]  — [n_obj, n_win, n_air]
        thickness   : float
        image_size  : tuple[int, int]  — (n_row, n_col)
    """
    text = path.read_text()
    all_lines = text.splitlines()

    # --- refraction meta (cam_id, window_id) ---
    meta = _parse_refraction_meta(all_lines)
    cam_id = int(meta.get("CAM_ID", -1))
    window_id = int(meta.get("WINDOW_ID", -1))

    if cam_id < 0:
        # fallback: infer from filename  cam3.txt -> 3
        m = re.search(r"cam(\d+)", path.stem)
        if m:
            cam_id = int(m.group(1))
            logger.warning("CAM_ID not in meta; inferred %d from filename %s",
                           cam_id, path.name)
        else:
            raise ValueError(f"Cannot determine CAM_ID from {path}")

    if window_id < 0:
        raise ValueError(
            f"WINDOW_ID missing from refraction meta in {path}")

    # --- data lines (skip comments / blank) ---
    data = _parse_data_lines(all_lines)
    # Expected order of data lines (index):
    #  0  PINPLATE
    #  1  proj_err  (mean,std)
    #  2  tri_err   (mean,std)
    #  3  img_size  (n_row,n_col)
    #  4  cam_matrix row0
    #  5  cam_matrix row1
    #  6  cam_matrix row2  ("0 0 1")
    #  7  distortion coeffs
    #  8  rotation vector
    #  9  rotation matrix row0
    # 10  rotation matrix row1
    # 11  rotation matrix row2
    # 12  inv rotation row0
    # 13  inv rotation row1
    # 14  inv rotation row2
    # 15  translation vector
    # 16  inv translation
    # 17  plane_pt (farthest)
    # 18  plane_n
    # 19  refract_array
    # 20  w_array
    # 21  proj_tol
    # 22  proj_nmax
    # 23  lr

    if len(data) < 21:
        raise ValueError(
            f"CamFile {path} has only {len(data)} data lines; expected ≥21")

    model = data[0]
    if model != "PINPLATE":
        raise ValueError(f"Unsupported camera model '{model}' in {path}")

    # image size
    n_row, n_col = (int(x) for x in data[3].split(","))

    # camera matrix row 0 → f, cx
    cm0 = data[4].split()
    f_val = float(cm0[0])
    cx_val = float(cm0[2])
    # camera matrix row 1 → cy
    cm1 = data[5].split()
    cy_val = float(cm1[2])

    # distortion
    dist_parts = data[7].split(",")
    k1 = float(dist_parts[0])
    k2 = float(dist_parts[1])

    # rotation vector
    rvec_parts = data[8].split(",")
    rvec = np.array([float(x) for x in rvec_parts], dtype=np.float64)

    # translation vector (space-separated)
    tvec_parts = data[15].split()
    tvec = np.array([float(x) for x in tvec_parts], dtype=np.float64)

    cam_params = np.array([
        rvec[0], rvec[1], rvec[2],
        tvec[0], tvec[1], tvec[2],
        f_val, cx_val, cy_val,
        k1, k2,
    ], dtype=np.float64)

    # plane point (farthest interface)
    pp_parts = data[17].split()
    plane_pt_export = np.array([float(x) for x in pp_parts], dtype=np.float64)

    # plane normal
    pn_parts = data[18].split()
    plane_n = np.array([float(x) for x in pn_parts], dtype=np.float64)

    # refract array
    refract_parts = data[19].split(",")
    refract_array = [float(x) for x in refract_parts]

    # thickness (w_array — single plate)
    w_parts = data[20].split(",")
    thickness = float(w_parts[0])

    return {
        "cam_id": cam_id,
        "window_id": window_id,
        "cam_params": cam_params,
        "plane_pt_export": plane_pt_export,
        "plane_n": plane_n,
        "refract_array": refract_array,
        "thickness": thickness,
        "image_size": (n_row, n_col),
    }


# ---------------------------------------------------------------------------
#  bundle_cache.json loader
# ---------------------------------------------------------------------------

def _load_bundle_cache(path: Path) -> Dict[str, Any]:
    """Load ``bundle_cache.json`` and convert to typed dicts.

    Returns
    -------
    dict with keys:
        cam_params   : Dict[int, np.ndarray]   — shape (11,) per camera
        window_planes: Dict[int, Dict]          — plane_pt (closest), plane_n
        window_media : Dict[int, Dict]          — n1, n2, n3, thickness
    """
    with open(path, "r") as fh:
        raw = json.load(fh)

    cam_params: Dict[int, np.ndarray] = {}
    for cid_str, arr in raw.get("cam_params", {}).items():
        cam_params[int(cid_str)] = np.array(arr, dtype=np.float64)

    window_planes: Dict[int, Dict[str, np.ndarray]] = {}
    for wid_str, pdata in raw.get("planes", {}).items():
        window_planes[int(wid_str)] = {
            "plane_pt": np.array(pdata["plane_pt"], dtype=np.float64),
            "plane_n": np.array(pdata["plane_n"], dtype=np.float64),
        }

    window_media: Dict[int, Dict[str, float]] = {}
    for wid_str, mdata in raw.get("window_media", {}).items():
        window_media[int(wid_str)] = {
            "n1": float(mdata["n1"]),
            "n2": float(mdata["n2"]),
            "n3": float(mdata["n3"]),
            "thickness": float(mdata["thickness"]),
        }

    return {
        "cam_params": cam_params,
        "window_planes": window_planes,
        "window_media": window_media,
    }


# ---------------------------------------------------------------------------
#  Cross-source consistency helpers
# ---------------------------------------------------------------------------

def _plane_pt_farthest_to_closest(
    pt_farthest: np.ndarray,
    plane_n: np.ndarray,
    thickness: float,
) -> np.ndarray:
    """Convert farthest-interface plane point to closest (internal repr).

    ``CamFileExporter`` stores ``P_farthest = P_closest + n * thickness``.
    Invert: ``P_closest = P_farthest - n * thickness``.
    """
    return pt_farthest - plane_n * thickness


def _compare_cam_params(
    camfile_params: Dict[int, np.ndarray],
    cache_params: Dict[int, np.ndarray],
    atol: float = 1e-4,
) -> List[str]:
    """Compare cam_params from camfiles vs bundle_cache, return warnings."""
    warnings: List[str] = []
    common = sorted(set(camfile_params) & set(cache_params))
    for cid in common:
        diff = np.abs(camfile_params[cid] - cache_params[cid])
        max_diff = float(np.max(diff))
        if max_diff > atol:
            labels = ["rx", "ry", "rz", "tx", "ty", "tz",
                      "f", "cx", "cy", "k1", "k2"]
            worst_idx = int(np.argmax(diff))
            warnings.append(
                f"cam {cid}: max param diff = {max_diff:.6g} "
                f"at {labels[worst_idx]} (idx {worst_idx})"
            )
    only_camfile = sorted(set(camfile_params) - set(cache_params))
    only_cache = sorted(set(cache_params) - set(camfile_params))
    if only_camfile:
        warnings.append(f"cam_ids only in camfiles: {only_camfile}")
    if only_cache:
        warnings.append(f"cam_ids only in bundle_cache: {only_cache}")
    return warnings


def _compare_planes(
    camfile_planes: Dict[int, Dict[str, np.ndarray]],
    cache_planes: Dict[int, Dict[str, np.ndarray]],
    atol: float = 0.05,
) -> List[str]:
    """Compare planes derived from camfiles vs bundle_cache."""
    warnings: List[str] = []
    common_wids = sorted(set(camfile_planes) & set(cache_planes))
    for wid in common_wids:
        for key in ("plane_pt", "plane_n"):
            a = camfile_planes[wid][key]
            b = cache_planes[wid][key]
            diff = float(np.max(np.abs(a - b)))
            if diff > atol:
                warnings.append(
                    f"window {wid} {key}: max diff = {diff:.6g}")
    return warnings


# ---------------------------------------------------------------------------
#  Validation
# ---------------------------------------------------------------------------

def validate_reference_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Check internal consistency of a loaded reference state.

    Parameters
    ----------
    state : dict
        As returned by :func:`load_reference_state`.

    Returns
    -------
    dict with keys:
        valid    : bool
        errors   : list[str]  — fatal inconsistencies
        warnings : list[str]  — non-fatal mismatches
    """
    errors: List[str] = []
    warnings: List[str] = []

    cam_params: Dict[int, np.ndarray] = state.get("cam_params", {})
    cam_to_window: Dict[int, int] = state.get("cam_to_window", {})
    window_planes: Dict[int, Dict] = state.get("window_planes", {})
    window_media: Dict[int, Dict] = state.get("window_media", {})

    # --- cam_params shape ---
    for cid, p in cam_params.items():
        if p.shape != (11,):
            errors.append(
                f"cam_params[{cid}] has shape {p.shape}; expected (11,)")
        if not np.all(np.isfinite(p)):
            errors.append(f"cam_params[{cid}] contains non-finite values")

    # --- cam_to_window completeness ---
    cam_ids = sorted(cam_params.keys())
    for cid in cam_ids:
        if cid not in cam_to_window:
            errors.append(f"cam {cid} has params but no cam_to_window entry")

    referenced_wids = set(cam_to_window.values())

    # --- window_planes completeness & validity ---
    for wid in sorted(referenced_wids):
        if wid not in window_planes:
            errors.append(
                f"window {wid} referenced by cam_to_window but missing "
                f"from window_planes")
        else:
            wp = window_planes[wid]
            pt = wp.get("plane_pt")
            n = wp.get("plane_n")
            if pt is None or n is None:
                errors.append(
                    f"window_planes[{wid}] missing plane_pt or plane_n")
            else:
                pt = np.asarray(pt)
                n = np.asarray(n)
                if pt.shape != (3,):
                    errors.append(
                        f"window_planes[{wid}].plane_pt shape {pt.shape}")
                if n.shape != (3,):
                    errors.append(
                        f"window_planes[{wid}].plane_n shape {n.shape}")
                norm_len = float(np.linalg.norm(n))
                if not (0.99 <= norm_len <= 1.01):
                    warnings.append(
                        f"window_planes[{wid}].plane_n |n|={norm_len:.6f}")
                if not np.all(np.isfinite(pt)):
                    errors.append(
                        f"window_planes[{wid}].plane_pt non-finite")
                if not np.all(np.isfinite(n)):
                    errors.append(
                        f"window_planes[{wid}].plane_n non-finite")

    # --- window_media completeness & validity ---
    for wid in sorted(referenced_wids):
        if wid not in window_media:
            errors.append(
                f"window {wid} referenced by cam_to_window but missing "
                f"from window_media")
        else:
            wm = window_media[wid]
            for key in ("n1", "n2", "n3", "thickness"):
                if key not in wm:
                    errors.append(
                        f"window_media[{wid}] missing '{key}'")
                else:
                    val = wm[key]
                    if not np.isfinite(val):
                        errors.append(
                            f"window_media[{wid}].{key} = {val} (non-finite)")
                    if key.startswith("n") and val <= 0:
                        errors.append(
                            f"window_media[{wid}].{key} = {val} (non-positive)")
            thickness = wm.get("thickness", 0.0)
            if thickness <= 0:
                errors.append(
                    f"window_media[{wid}].thickness = {thickness} (non-positive)")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
#  Public entry-point
# ---------------------------------------------------------------------------

def load_reference_state(
    camfile_dir: str | Path,
    bundle_cache_path: Optional[str | Path] = None,
    *,
    cross_check: bool = True,
    cross_check_atol_params: float = 1e-4,
    cross_check_atol_planes: float = 0.05,
) -> Dict[str, Any]:
    """Load a coherent BA reference state from on-disk artifacts.

    Priority logic
    --------------
    * **cam_params** and **cam_to_window** are always read from camfiles
      (authoritative).
    * **window_planes** and **window_media** are read from
      ``bundle_cache.json`` when available (it stores the closest-interface
      ``plane_pt`` used internally by BA).  When the cache is absent, planes
      and media are derived from camfiles (with the farthest→closest
      conversion applied).
    * When both sources exist and *cross_check* is True, ``cam_params`` and
      planes are compared and warnings are logged.

    Parameters
    ----------
    camfile_dir : path-like
        Directory containing ``cam{id}.txt`` PINPLATE files.
    bundle_cache_path : path-like, optional
        Path to ``bundle_cache.json``.  When *None*, the loader looks for
        ``bundle_cache.json`` in the *parent* of *camfile_dir*.
    cross_check : bool
        If True, compare redundant data between sources and log warnings.
    cross_check_atol_params : float
        Absolute tolerance for ``cam_params`` cross-check.
    cross_check_atol_planes : float
        Absolute tolerance (mm) for ``plane_pt`` cross-check.

    Returns
    -------
    dict with keys:
        cam_params    : Dict[int, np.ndarray]  — shape (11,) per camera
        cam_to_window : Dict[int, int]
        window_planes : Dict[int, Dict]        — {wid: {plane_pt, plane_n}}
        window_media  : Dict[int, Dict]        — {wid: {n1, n2, n3, thickness}}
        metadata      : dict                   — source paths, cam_ids, etc.
        validation    : dict                   — output of validate_reference_state
    """
    camfile_dir = Path(camfile_dir)
    if not camfile_dir.is_dir():
        raise FileNotFoundError(f"camfile_dir not found: {camfile_dir}")

    # --- discover camfiles ---
    camfiles = sorted(camfile_dir.glob("cam*.txt"))
    if not camfiles:
        raise FileNotFoundError(
            f"No cam*.txt files found in {camfile_dir}")

    # --- parse all camfiles ---
    cam_params: Dict[int, np.ndarray] = {}
    cam_to_window: Dict[int, int] = {}
    camfile_planes_by_window: Dict[int, Dict[str, np.ndarray]] = {}
    camfile_media_by_window: Dict[int, Dict[str, float]] = {}
    image_sizes: Dict[int, Tuple[int, int]] = {}

    for cf_path in camfiles:
        parsed = parse_camfile(cf_path)
        cid = parsed["cam_id"]
        wid = parsed["window_id"]

        cam_params[cid] = parsed["cam_params"]
        cam_to_window[cid] = wid
        image_sizes[cid] = parsed["image_size"]

        # derive closest-interface plane_pt from export (farthest)
        plane_pt_closest = _plane_pt_farthest_to_closest(
            parsed["plane_pt_export"],
            parsed["plane_n"],
            parsed["thickness"],
        )

        # keep *first* encountered plane per window (all cams sharing a
        # window should have identical plane data in a consistent BA export)
        if wid not in camfile_planes_by_window:
            camfile_planes_by_window[wid] = {
                "plane_pt": plane_pt_closest,
                "plane_n": parsed["plane_n"].copy(),
            }

        if wid not in camfile_media_by_window:
            # refract_array is [n_obj, n_win, n_air] (farthest→nearest)
            ra = parsed["refract_array"]
            camfile_media_by_window[wid] = {
                "n1": ra[2],           # n_air   (nearest / n1)
                "n2": ra[1],           # n_window (n2)
                "n3": ra[0],           # n_object (farthest / n3)
                "thickness": parsed["thickness"],
            }

    logger.info("Parsed %d camfiles: cam_ids=%s", len(cam_params),
                sorted(cam_params.keys()))

    # --- bundle_cache ---
    if bundle_cache_path is None:
        candidate = camfile_dir.parent / "bundle_cache.json"
        if candidate.is_file():
            bundle_cache_path = candidate
            logger.info("Auto-discovered bundle_cache at %s", candidate)

    cache_data: Optional[Dict[str, Any]] = None
    if bundle_cache_path is not None:
        bundle_cache_path = Path(bundle_cache_path)
        if bundle_cache_path.is_file():
            cache_data = _load_bundle_cache(bundle_cache_path)
            logger.info("Loaded bundle_cache.json (%d cams, %d windows)",
                        len(cache_data["cam_params"]),
                        len(cache_data["window_planes"]))
        else:
            logger.warning("bundle_cache_path provided but file not found: %s",
                           bundle_cache_path)

    # --- resolve window_planes and window_media ---
    if cache_data is not None:
        # Prefer bundle_cache for planes (closest-interface, no conversion
        # rounding) and media (float-precision thickness).
        window_planes = cache_data["window_planes"]
        window_media = cache_data["window_media"]
        source_planes = "bundle_cache"

        # cross-check
        if cross_check:
            pw = _compare_cam_params(
                cam_params, cache_data["cam_params"],
                atol=cross_check_atol_params,
            )
            for w in pw:
                logger.warning("[cross-check] cam_params: %s", w)

            plw = _compare_planes(
                camfile_planes_by_window, window_planes,
                atol=cross_check_atol_planes,
            )
            for w in plw:
                logger.warning("[cross-check] planes: %s", w)
    else:
        # Derive from camfiles only (farthest→closest already applied).
        window_planes = camfile_planes_by_window
        window_media = camfile_media_by_window
        source_planes = "camfiles"
        logger.info("No bundle_cache; using camfile-derived planes/media")

    # --- assemble state ---
    state: Dict[str, Any] = {
        "cam_params": cam_params,
        "cam_to_window": cam_to_window,
        "window_planes": window_planes,
        "window_media": window_media,
        "metadata": {
            "camfile_dir": str(camfile_dir),
            "bundle_cache_path": (
                str(bundle_cache_path) if bundle_cache_path else None
            ),
            "cam_ids": sorted(cam_params.keys()),
            "window_ids": sorted(window_planes.keys()),
            "image_sizes": image_sizes,
            "source_planes": source_planes,
        },
    }

    # --- validate ---
    validation = validate_reference_state(state)
    state["validation"] = validation

    if not validation["valid"]:
        for e in validation["errors"]:
            logger.error("[validation] %s", e)
    for w in validation["warnings"]:
        logger.warning("[validation] %s", w)

    return state


# ===========================================================================
#  Task 2: BA-Compatible Candidate Evaluator
# ===========================================================================

# ---------------------------------------------------------------------------
#  CSV observation loader
# ---------------------------------------------------------------------------

def load_observations_csv(
    csv_path: str | Path,
    wand_length: float,
    *,
    dist_coeff_num: int = 0,
) -> Dict[str, Any]:
    """Parse a wand-point CSV into a BA-compatible *dataset* dict.

    The CSV must have the header::

        Frame,Camera,Status,PointIdx,X,Y,Radius,Metric

    ``Filtered_Small`` rows become endpoint **A** and ``Filtered_Large``
    rows become endpoint **B**.  Rows with other ``Status`` values are
    silently skipped.

    Parameters
    ----------
    csv_path : path-like
        Path to the ``wandpoints_filtered*.csv`` file.
    wand_length : float
        Physical wand length in mm.
    dist_coeff_num : int
        Number of distortion coefficients to include (0, 1, or 2).

    Returns
    -------
    dict
        Keys compatible with ``RefractiveBAOptimizer.__init__``:
        frames, cam_ids, obsA, obsB, radii_small, radii_large, maskA,
        maskB, num_frames, num_cams, wand_length, dist_coeff_num,
        total_observations.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Observation CSV not found: {csv_path}")

    # --- first pass: collect raw rows ---
    # {frame_id: {cam_id: {'small': (x, y, r), 'large': (x, y, r)}}}
    raw: Dict[int, Dict[int, Dict[str, Tuple[float, float, float]]]] = {}

    with open(csv_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            status = row["Status"].strip()
            if status not in ("Filtered_Small", "Filtered_Large"):
                continue
            fid = int(row["Frame"])
            cid = int(row["Camera"])
            x = float(row["X"])
            y = float(row["Y"])
            r = float(row.get("Radius", 0.0))

            frame_dict = raw.setdefault(fid, {})
            cam_dict = frame_dict.setdefault(cid, {})
            key = "small" if status == "Filtered_Small" else "large"
            if key in cam_dict:
                logger.warning(
                    "Duplicate %s in frame %d cam %d; keeping first",
                    status, fid, cid,
                )
                continue
            cam_dict[key] = (x, y, r)

    if not raw:
        raise ValueError(f"No Filtered_Small / Filtered_Large rows in {csv_path}")

    # --- determine active cameras (present in every frame) ---
    all_cams_per_frame = [set(cams.keys()) for cams in raw.values()]
    active_cams_set = all_cams_per_frame[0]
    for s in all_cams_per_frame[1:]:
        active_cams_set = active_cams_set & s
    active_cams = sorted(active_cams_set)

    if not active_cams:
        # Fallback: union of all cam IDs
        active_cams = sorted({c for cams in raw.values() for c in cams})
        logger.warning(
            "No cameras consistently present across all frames; "
            "using union: %s", active_cams,
        )

    frames_list = sorted(raw.keys())

    # --- build BA-compatible dicts ---
    obsA: Dict[int, Dict[int, np.ndarray]] = {}
    obsB: Dict[int, Dict[int, np.ndarray]] = {}
    radii_small: Dict[int, Dict[int, float]] = {}
    radii_large: Dict[int, Dict[int, float]] = {}
    maskA: Dict[int, Dict[int, bool]] = {}
    maskB: Dict[int, Dict[int, bool]] = {}

    for fid in frames_list:
        obsA[fid] = {}
        obsB[fid] = {}
        radii_small[fid] = {}
        radii_large[fid] = {}
        maskA[fid] = {}
        maskB[fid] = {}
        frame_raw = raw.get(fid, {})

        for cid in active_cams:
            cam_raw = frame_raw.get(cid, {})

            if "small" in cam_raw:
                x, y, r = cam_raw["small"]
                uv = np.array([x, y], dtype=np.float64)
                if np.all(np.isfinite(uv)):
                    obsA[fid][cid] = uv
                    radii_small[fid][cid] = r
                    maskA[fid][cid] = True
                else:
                    maskA[fid][cid] = False
            else:
                maskA[fid][cid] = False

            if "large" in cam_raw:
                x, y, r = cam_raw["large"]
                uv = np.array([x, y], dtype=np.float64)
                if np.all(np.isfinite(uv)):
                    obsB[fid][cid] = uv
                    radii_large[fid][cid] = r
                    maskB[fid][cid] = True
                else:
                    maskB[fid][cid] = False
            else:
                maskB[fid][cid] = False

    dataset = {
        "frames": frames_list,
        "cam_ids": active_cams,
        "obsA": obsA,
        "obsB": obsB,
        "radii_small": radii_small,
        "radii_large": radii_large,
        "maskA": maskA,
        "maskB": maskB,
        "num_frames": len(frames_list),
        "num_cams": len(active_cams),
        "wand_length": wand_length,
        "dist_coeff_num": dist_coeff_num,
        "total_observations": len(frames_list) * len(active_cams),
    }

    logger.info(
        "Loaded %d frames, %d cams, %d total obs from %s",
        len(frames_list), len(active_cams),
        dataset["total_observations"], csv_path.name,
    )
    return dataset


# ---------------------------------------------------------------------------
#  Evaluation context
# ---------------------------------------------------------------------------

@dataclass
class EvaluationContext:
    """Immutable context for evaluating candidate parameter vectors.

    Created once by :func:`build_evaluation_context`; passed to every call
    of :func:`evaluate_candidate`.
    """
    optimizer: Any          # RefractiveBAOptimizer
    layout: List[Tuple]     # parameter layout (type, id, subidx)
    lambda_eff: float       # effective wand-length weight
    n_params: int           # length of delta vector x
    sigma_ray: float        # global ray sigma (mm)
    sigma_wand: float       # wand-length sigma (mm)
    cam_ids: List[int]
    window_ids: List[int]


@dataclass
class WorkerEvaluationRuntime:
    """Worker-local runtime reconstructed from serializable shared setup."""

    setup_key: str
    ctx: EvaluationContext
    scales: np.ndarray


_WORKER_RUNTIME: Optional[WorkerEvaluationRuntime] = None


class _CameraSettingsStub:
    """Minimal stand-in for the ``base`` object expected by
    :class:`CppCameraFactory.init_cams_cpp_in_memory`.

    Provides ``camera_settings`` (per-cam dict) and ``image_size`` (fallback).
    """

    def __init__(
        self,
        image_sizes: Dict[int, Tuple[int, int]],
        cam_params: Dict[int, np.ndarray],
    ) -> None:
        # image_sizes: {cam_id: (n_row, n_col)}
        # camera_settings: {cam_id: {focal, width, height}}
        self.camera_settings: Dict[int, Dict[str, Any]] = {}
        for cid, (n_row, n_col) in image_sizes.items():
            f_val = float(cam_params[cid][6]) if cid in cam_params else 1000.0
            self.camera_settings[cid] = {
                "focal": f_val,
                "width": n_col,
                "height": n_row,
            }
        # Fallback image_size (first camera's values)
        if image_sizes:
            first = next(iter(image_sizes.values()))
            self.image_size: Tuple[int, int] = first
        else:
            self.image_size = (800, 1280)


def build_evaluation_context(
    ref_state: Dict[str, Any],
    dataset: Dict[str, Any],
    wand_length: float,
    *,
    lambda_base_per_cam: float = 2.0,
    max_frames: int = 50000,
) -> EvaluationContext:
    """Build the evaluation context needed by :func:`evaluate_candidate`.

    This creates C++ Camera objects and a :class:`RefractiveBAOptimizer`
    instance pre-loaded with the reference state, observations, and sigma
    values.  The optimizer is **not** run; it is used only for its
    ``evaluate_residuals`` / ``_unpack_params_delta`` infrastructure.

    Parameters
    ----------
    ref_state : dict
        As returned by :func:`load_reference_state`.
    dataset : dict
        As returned by :func:`load_observations_csv`.
    wand_length : float
        Physical wand length in mm.
    lambda_base_per_cam : float
        Per-camera weight for wand-length residuals.
    max_frames : int
        Maximum number of frames to use (random sub-sample if exceeded).

    Returns
    -------
    EvaluationContext
    """
    # Lazy imports — keep module-level import list light
    from .refraction_calibration_BA import RefractiveBAConfig, RefractiveBAOptimizer
    from .refraction_wand_calibrator import CppCameraFactory

    cam_params = ref_state["cam_params"]
    cam_to_window = ref_state["cam_to_window"]
    window_planes = ref_state["window_planes"]
    window_media = ref_state["window_media"]
    image_sizes = ref_state["metadata"]["image_sizes"]

    # --- create C++ Camera objects ---
    base_stub = _CameraSettingsStub(image_sizes, cam_params)
    cams_cpp = CppCameraFactory.init_cams_cpp_in_memory(
        base_stub, cam_params, window_media, cam_to_window, window_planes,
    )

    # --- BA config: no regularization, intrinsics fixed ---
    n_cams = len(cam_params)
    cfg = RefractiveBAConfig(
        use_regularization=False,
        max_frames=max_frames,
        lambda_base_per_cam=lambda_base_per_cam,
        dist_coeff_num=dataset.get("dist_coeff_num", 0),
        use_proj_residuals=False,
        skip_optimization=True,      # we never call optimize()
        verbosity=0,
    )

    # --- instantiate optimizer (sets up obs_cache, anchors, etc.) ---
    optimizer = RefractiveBAOptimizer(
        dataset=dataset,
        cam_params=cam_params,
        cams_cpp=cams_cpp,
        cam_to_window=cam_to_window,
        window_media=window_media,
        window_planes=window_planes,
        wand_length=wand_length,
        config=cfg,
    )

    # --- compute sigma values (requires C++ cameras + observations) ---
    optimizer._compute_physical_sigmas()

    # --- parameter layout: planes + cam extrinsics, intrinsics fixed ---
    layout = optimizer._get_param_layout(
        enable_planes=True,
        enable_cam_t=True,
        enable_cam_r=True,
        enable_cam_f=False,
        enable_win_t=False,
        enable_cam_k1=False,
        enable_cam_k2=False,
    )

    lambda_eff = lambda_base_per_cam * n_cams

    ctx = EvaluationContext(
        optimizer=optimizer,
        layout=layout,
        lambda_eff=lambda_eff,
        n_params=len(layout),
        sigma_ray=optimizer.sigma_ray_global,
        sigma_wand=optimizer.sigma_wand,
        cam_ids=optimizer.active_cam_ids,
        window_ids=optimizer.window_ids,
    )

    logger.info(
        "EvaluationContext ready: %d params, %d cams, %d windows, "
        "sigma_ray=%.4f mm, sigma_wand=%.4f mm, lambda_eff=%.1f",
        ctx.n_params, n_cams, len(ctx.window_ids),
        ctx.sigma_ray, ctx.sigma_wand, ctx.lambda_eff,
    )
    return ctx


def build_shared_setup(
    ref_state: Dict[str, Any],
    dataset: Dict[str, Any],
    probe_scales: np.ndarray,
    *,
    wand_length: float,
    lambda_base_per_cam: float = 2.0,
    max_frames: int = 50000,
    dist_coeff_num: int = 0,
) -> Dict[str, Any]:
    """Build a worker-safe serializable setup payload.

    The returned dict intentionally contains only plain Python containers,
    scalars, and NumPy arrays. Native/pybind objects are excluded by design.
    """
    scales = np.asarray(probe_scales, dtype=np.float64).copy()
    if scales.ndim != 1:
        raise ValueError(
            f"probe_scales must be 1-D, got shape={scales.shape}"
        )

    setup_key = (
        f"cams={len(ref_state.get('cam_params', {}))}|"
        f"frames={len(dataset.get('frames', []))}|"
        f"params={scales.size}|"
        f"wl={float(wand_length):.8f}|"
        f"lmb={float(lambda_base_per_cam):.8f}|"
        f"mf={int(max_frames)}"
    )

    return {
        "setup_key": setup_key,
        "wand_length": float(wand_length),
        "lambda_base_per_cam": float(lambda_base_per_cam),
        "max_frames": int(max_frames),
        "dist_coeff_num": int(dist_coeff_num),
        "probe_scales": scales,
        "n_params_expected": int(scales.size),
        "ref_state": ref_state,
        "dataset": dataset,
    }


def _init_probing_worker():
    """Suppress BLAS multi-threading in worker processes."""
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

def rebuild_worker_evaluation_runtime(
    shared_setup: Dict[str, Any],
) -> WorkerEvaluationRuntime:
    """Rebuild worker-local EvaluationContext and native cameras from setup."""
    ref_state = shared_setup["ref_state"]
    dataset = dict(shared_setup["dataset"])
    dataset["dist_coeff_num"] = int(shared_setup.get(
        "dist_coeff_num", dataset.get("dist_coeff_num", 0)
    ))

    ctx = build_evaluation_context(
        ref_state,
        dataset,
        float(shared_setup["wand_length"]),
        lambda_base_per_cam=float(shared_setup["lambda_base_per_cam"]),
        max_frames=int(shared_setup["max_frames"]),
    )

    scales = np.asarray(shared_setup["probe_scales"], dtype=np.float64)
    n_params_expected = int(shared_setup.get("n_params_expected", scales.size))
    if scales.ndim != 1:
        raise ValueError(
            f"shared_setup.probe_scales must be 1-D, got shape={scales.shape}"
        )
    if scales.size != n_params_expected:
        raise ValueError(
            f"shared_setup probe scale count mismatch: {scales.size} != {n_params_expected}"
        )
    if scales.size != ctx.n_params:
        raise ValueError(
            f"shared_setup probe scale count {scales.size} != ctx.n_params {ctx.n_params}"
        )

    return WorkerEvaluationRuntime(
        setup_key=str(shared_setup.get("setup_key", "")),
        ctx=ctx,
        scales=scales.copy(),
    )


def initialize_worker_evaluation_runtime(
    shared_setup: Dict[str, Any],
) -> WorkerEvaluationRuntime:
    """Initialize worker runtime once; reuse on subsequent calls."""
    global _WORKER_RUNTIME

    setup_key = str(shared_setup.get("setup_key", ""))
    if _WORKER_RUNTIME is None:
        _WORKER_RUNTIME = rebuild_worker_evaluation_runtime(shared_setup)
        return _WORKER_RUNTIME

    if _WORKER_RUNTIME.setup_key != setup_key:
        raise RuntimeError(
            "Worker runtime already initialized for a different shared setup: "
            f"loaded={_WORKER_RUNTIME.setup_key!r}, requested={setup_key!r}"
        )
    return _WORKER_RUNTIME


def get_worker_evaluation_runtime() -> WorkerEvaluationRuntime:
    """Return previously initialized worker runtime."""
    if _WORKER_RUNTIME is None:
        raise RuntimeError(
            "Worker evaluation runtime not initialized. "
            "Call initialize_worker_evaluation_runtime(shared_setup) first."
        )
    return _WORKER_RUNTIME


def _run_cma_worker(
    shared_setup: Dict[str, Any],
    run_id: int,
    sigma0: float,
    popsize: Optional[int],
    max_evals: int,
    max_generations: Optional[int],
    stagnation_gens: int,
    sigma_stop: float,
    seed: Optional[int],
) -> CMARunResult:
    """Worker entrypoint for parallel CMA-ES runs.

    Designed for ``ProcessPoolExecutor`` with ``spawn`` context.
    Reconstructs worker-local evaluation runtime from the serializable
    ``shared_setup`` payload (no native/C++ objects are pickled), then
    executes one full ``_run_cma_single()`` to completion.

    Parameters
    ----------
    shared_setup : dict
        Serializable setup from :func:`build_shared_setup`.
    run_id : int
        Run identifier (used for logging and seed schedule).
    sigma0, popsize, max_evals, max_generations, stagnation_gens,
    sigma_stop, seed
        CMA-ES hyperparameters forwarded to :func:`_run_cma_single`.

    Returns
    -------
    CMARunResult
    """
    runtime = initialize_worker_evaluation_runtime(shared_setup)
    return _run_cma_single(
        runtime.ctx,
        runtime.scales,
        run_id=run_id,
        sigma0=sigma0,
        popsize=popsize,
        max_evals=max_evals,
        max_generations=max_generations,
        stagnation_gens=stagnation_gens,
        sigma_stop=sigma_stop,
        seed=seed,
    )

# ---------------------------------------------------------------------------
#  Candidate evaluator
# ---------------------------------------------------------------------------

_SENTINEL_OBJECTIVE = 1e18


@dataclass
class CandidateResult:
    """Outcome of evaluating a single candidate parameter vector."""
    objective: float            # scalar cost (S_ray + lambda_eff * S_len)
    ray_rmse: float             # sqrt(S_ray / N_ray)  [mm]
    len_rmse: float             # sqrt(S_len / N_len)  [mm]
    proj_rmse: float            # sqrt(S_proj / N_proj) [px] (0 if disabled)
    n_ray: int                  # number of ray residuals
    n_len: int                  # number of length residuals
    n_proj: int                 # number of projection residuals
    s_ray: float                # sum of squared ray residuals (normalised)
    s_len: float                # sum of squared len residuals (normalised)
    ok: bool = True             # False if evaluation failed
    failure_reason: str = ""    # human-readable failure info


@dataclass
class ProbeCompensationResult:
    """Outcome of one probe-step compensation evaluation."""
    ray_rmse: float
    is_valid: bool
    compensated_x_delta: np.ndarray
    failure_reason: str = ""
    compensation_nfev: int = 0
    compensation_njev: int = 0
    compensation_status: int = 0
    compensation_message: str = ""
    compensation_method: str = ""
    compensation_max_nfev: int = 0
    lock_drift_max: float = 0.0


def _candidate_result_dict(
    objective: float, ray_rmse: float, len_rmse: float, proj_rmse: float,
    n_ray: int, n_len: int, n_proj: int,
    s_ray: float, s_len: float,
    ok: bool = True, failure_reason: str = "",
) -> Dict[str, Any]:
    """Build the public result dict for :func:`evaluate_candidate`."""
    return {
        "objective": objective,
        "ray_rmse": ray_rmse,
        "len_rmse": len_rmse,
        "proj_rmse": proj_rmse,
        "success": ok,
        "error": failure_reason if not ok else None,
        # extended diagnostics (available but not required by spec)
        "n_ray": n_ray,
        "n_len": n_len,
        "n_proj": n_proj,
        "s_ray": s_ray,
        "s_len": s_len,
    }


def _check_probe_geometry_validity(
    ctx: EvaluationContext,
    planes: Dict[int, Dict[str, Any]],
    cam_params: Dict[int, np.ndarray],
    window_media: Dict[int, Dict[str, Any]],
) -> Tuple[bool, str]:
    """Validate geometry for probe compensation before residual evaluation."""
    opt = ctx.optimizer
    state = {
        "cam_params": cam_params,
        "cam_to_window": dict(getattr(opt, "cam_to_window", {})),
        "window_planes": planes,
        "window_media": window_media,
    }
    validation = validate_reference_state(state)
    if not validation.get("valid", False):
        errors = validation.get("errors", [])
        return False, "; ".join(str(e) for e in errors[:3]) or "invalid_geometry"

    for cid, p in cam_params.items():
        if p.shape != (11,):
            return False, f"cam_params[{cid}] shape {p.shape}"
        if not np.all(np.isfinite(p)):
            return False, f"cam_params[{cid}] non-finite"
        if float(p[6]) <= 0.0:
            return False, f"cam_params[{cid}].f <= 0"

    for wid, pl in planes.items():
        n = np.asarray(pl.get("plane_n"), dtype=np.float64)
        if n.shape != (3,) or not np.all(np.isfinite(n)):
            return False, f"window_planes[{wid}].plane_n invalid"
        n_norm = float(np.linalg.norm(n))
        if n_norm <= 1e-12:
            return False, f"window_planes[{wid}].plane_n degenerate"

    return True, ""


def _restore_optimizer_native_reference_state(ctx: EvaluationContext) -> None:
    """Restore optimizer native camera state to zero-delta reference."""
    opt = ctx.optimizer
    try:
        x0 = np.zeros(ctx.n_params, dtype=np.float64)
        planes0, cams0, media0 = opt._unpack_params_delta(x0, ctx.layout)
        opt.evaluate_residuals(planes0, cams0, ctx.lambda_eff, window_media=media0)
    except Exception as exc:
        logger.debug("native-state restore after probe compensation failed: %s", exc)


def evaluate_probe_step_with_compensation(
    x_probe: np.ndarray,
    ctx: EvaluationContext,
    *,
    locked_param_indices: List[int],
    enable_compensation: bool = True,
    max_compensation_iters: int = 3,
    lock_tolerance: float = 1e-12,
) -> ProbeCompensationResult:
    """Evaluate one probe step with limited ray-only compensation.

    Locked indices are held fixed to ``x_probe`` during compensation.
    Only non-locked active parameters are optimized, and optimization is capped
    by ``max_compensation_iters`` (typically 2-3).
    """
    opt = ctx.optimizer
    x_probe = np.asarray(x_probe, dtype=np.float64).copy()

    if x_probe.shape != (ctx.n_params,):
        return ProbeCompensationResult(
            ray_rmse=np.inf,
            is_valid=False,
            compensated_x_delta=x_probe,
            failure_reason=(
                f"x_probe shape {x_probe.shape} != expected ({ctx.n_params},)"
            ),
            compensation_max_nfev=int(max(0, max_compensation_iters)),
        )
    if not np.all(np.isfinite(x_probe)):
        return ProbeCompensationResult(
            ray_rmse=np.inf,
            is_valid=False,
            compensated_x_delta=x_probe,
            failure_reason="x_probe contains NaN/Inf",
            compensation_max_nfev=int(max(0, max_compensation_iters)),
        )

    locked_idx = np.unique(np.asarray(locked_param_indices, dtype=np.int64))
    if np.any((locked_idx < 0) | (locked_idx >= ctx.n_params)):
        return ProbeCompensationResult(
            ray_rmse=np.inf,
            is_valid=False,
            compensated_x_delta=x_probe,
            failure_reason="locked_param_indices out of range",
            compensation_max_nfev=int(max(0, max_compensation_iters)),
        )

    try:
        cand_planes, cand_cams, cand_media = opt._unpack_params_delta(x_probe, ctx.layout)
    except Exception as exc:
        return ProbeCompensationResult(
            ray_rmse=np.inf,
            is_valid=False,
            compensated_x_delta=x_probe,
            failure_reason=f"unpack_failed: {exc!r}",
            compensation_max_nfev=int(max(0, max_compensation_iters)),
        )

    is_geom_valid, geom_reason = _check_probe_geometry_validity(
        ctx, cand_planes, cand_cams, cand_media
    )
    if not is_geom_valid:
        return ProbeCompensationResult(
            ray_rmse=np.inf,
            is_valid=False,
            compensated_x_delta=x_probe,
            failure_reason=f"invalid_geometry_pre: {geom_reason}",
            compensation_max_nfev=int(max(0, max_compensation_iters)),
        )

    try:
        free_idx = np.setdiff1d(np.arange(ctx.n_params, dtype=np.int64), locked_idx)
        compensation_nfev = 0
        compensation_njev = 0
        compensation_status = 0
        compensation_message = ""
        compensation_method = ""
        compensated_x = x_probe.copy()
        invalid_during = False
        invalid_reason = ""

        least_squares = __import__(
            "scipy.optimize", fromlist=["least_squares"]
        ).least_squares

        ray_slots = 1
        if hasattr(opt, "_compute_slot_counts"):
            try:
                ray_slots = max(int(opt._compute_slot_counts()[0]), 1)
            except Exception:
                ray_slots = 1
        penalty_res = np.full(ray_slots, 1e6, dtype=np.float64)

        if enable_compensation and max_compensation_iters > 0 and free_idx.size > 0:
            x0_free = x_probe[free_idx].copy()

            def _ray_only_residuals(x_free: np.ndarray) -> np.ndarray:
                nonlocal invalid_during, invalid_reason
                x_full = x_probe.copy()
                x_full[free_idx] = np.asarray(x_free, dtype=np.float64)
                if locked_idx.size > 0:
                    x_full[locked_idx] = x_probe[locked_idx]
                try:
                    pl, cp, wm = opt._unpack_params_delta(x_full, ctx.layout)
                except Exception as exc:
                    invalid_during = True
                    invalid_reason = f"unpack_failed_during: {exc!r}"
                    return penalty_res

                valid_now, reason_now = _check_probe_geometry_validity(ctx, pl, cp, wm)
                if not valid_now:
                    invalid_during = True
                    invalid_reason = f"invalid_geometry_during: {reason_now}"
                    return penalty_res

                try:
                    residuals, S_ray, _, _, _, _, _ = opt.evaluate_residuals(
                        pl, cp, ctx.lambda_eff, window_media=wm
                    )
                except Exception as exc:
                    invalid_during = True
                    invalid_reason = f"eval_failed_during: {exc!r}"
                    return penalty_res

                if not np.isfinite(float(S_ray)):
                    invalid_during = True
                    invalid_reason = "non_finite_s_ray_during"
                    return penalty_res

                ray_res = np.asarray(residuals[:ray_slots], dtype=np.float64)
                if ray_res.size != ray_slots or not np.all(np.isfinite(ray_res)):
                    invalid_during = True
                    invalid_reason = "non_finite_ray_residuals_during"
                    return penalty_res
                return ray_res

            try:
                lsq_res = least_squares(
                    _ray_only_residuals,
                    x0_free,
                    method="lm",
                    max_nfev=int(max_compensation_iters),
                    ftol=1e-9,
                    xtol=1e-9,
                    gtol=1e-9,
                )
                compensation_method = "lm"
            except Exception:
                lsq_res = least_squares(
                    _ray_only_residuals,
                    x0_free,
                    method="trf",
                    max_nfev=int(max_compensation_iters),
                    ftol=1e-9,
                    xtol=1e-9,
                    gtol=1e-9,
                )
                compensation_method = "trf"

            compensation_nfev = int(getattr(lsq_res, "nfev", 0) or 0)
            compensation_njev = int(getattr(lsq_res, "njev", 0) or 0)
            compensation_status = int(getattr(lsq_res, "status", 0) or 0)
            compensation_message = str(getattr(lsq_res, "message", "") or "")
            compensated_x[free_idx] = np.asarray(lsq_res.x, dtype=np.float64)

        if locked_idx.size > 0:
            # Measure drift BEFORE restoring locked values so the check is meaningful
            lock_drift_max = float(np.max(np.abs(compensated_x[locked_idx] - x_probe[locked_idx])))
            # Now restore locked parameters to their original values
            compensated_x[locked_idx] = x_probe[locked_idx]
        else:
            lock_drift_max = 0.0

        if lock_drift_max > lock_tolerance:
            return ProbeCompensationResult(
                ray_rmse=np.inf,
                is_valid=False,
                compensated_x_delta=compensated_x,
                failure_reason=(
                    f"locked_parameters_drifted: max={lock_drift_max:.3e} > tol={lock_tolerance:.3e}"
                ),
                compensation_nfev=compensation_nfev,
                compensation_njev=compensation_njev,
                compensation_status=compensation_status,
                compensation_message=compensation_message,
                compensation_method=compensation_method,
                compensation_max_nfev=int(max(0, max_compensation_iters)),
                lock_drift_max=lock_drift_max,
            )

        try:
            final_planes, final_cams, final_media = opt._unpack_params_delta(compensated_x, ctx.layout)
        except Exception as exc:
            return ProbeCompensationResult(
                ray_rmse=np.inf,
                is_valid=False,
                compensated_x_delta=compensated_x,
                failure_reason=f"final_unpack_failed: {exc!r}",
                compensation_nfev=compensation_nfev,
                compensation_njev=compensation_njev,
                compensation_status=compensation_status,
                compensation_message=compensation_message,
                compensation_method=compensation_method,
                compensation_max_nfev=int(max(0, max_compensation_iters)),
                lock_drift_max=lock_drift_max,
            )

        valid_final, reason_final = _check_probe_geometry_validity(
            ctx, final_planes, final_cams, final_media
        )
        if not valid_final:
            return ProbeCompensationResult(
                ray_rmse=np.inf,
                is_valid=False,
                compensated_x_delta=compensated_x,
                failure_reason=f"invalid_geometry_post: {reason_final}",
                compensation_nfev=compensation_nfev,
                compensation_njev=compensation_njev,
                compensation_status=compensation_status,
                compensation_message=compensation_message,
                compensation_method=compensation_method,
                compensation_max_nfev=int(max(0, max_compensation_iters)),
                lock_drift_max=lock_drift_max,
            )

        if invalid_during:
            return ProbeCompensationResult(
                ray_rmse=np.inf,
                is_valid=False,
                compensated_x_delta=compensated_x,
                failure_reason=invalid_reason or "invalid_geometry_during",
                compensation_nfev=compensation_nfev,
                compensation_njev=compensation_njev,
                compensation_status=compensation_status,
                compensation_message=compensation_message,
                compensation_method=compensation_method,
                compensation_max_nfev=int(max(0, max_compensation_iters)),
                lock_drift_max=lock_drift_max,
            )

        final_eval = evaluate_candidate(compensated_x, ctx)
        if not final_eval.get("success", False):
            return ProbeCompensationResult(
                ray_rmse=np.inf,
                is_valid=False,
                compensated_x_delta=compensated_x,
                failure_reason=f"final_eval_failed: {final_eval.get('error', 'unknown')}",
                compensation_nfev=compensation_nfev,
                compensation_njev=compensation_njev,
                compensation_status=compensation_status,
                compensation_message=compensation_message,
                compensation_method=compensation_method,
                compensation_max_nfev=int(max(0, max_compensation_iters)),
                lock_drift_max=lock_drift_max,
            )

        return ProbeCompensationResult(
            ray_rmse=float(final_eval["ray_rmse"]),
            is_valid=True,
            compensated_x_delta=compensated_x,
            compensation_nfev=compensation_nfev,
            compensation_njev=compensation_njev,
            compensation_status=compensation_status,
            compensation_message=compensation_message,
            compensation_method=compensation_method,
            compensation_max_nfev=int(max(0, max_compensation_iters)),
            lock_drift_max=lock_drift_max,
        )
    finally:
        _restore_optimizer_native_reference_state(ctx)


def evaluate_candidate(
    x_delta: Optional[np.ndarray],
    ctx: EvaluationContext,
) -> Dict[str, Any]:
    """Evaluate one candidate parameter vector through BA residuals.

    Parameters
    ----------
    x_delta : ndarray or None
        Delta vector of length ``ctx.n_params``.  *None* or all-zeros
        evaluates the reference state itself.
    ctx : EvaluationContext
        As returned by :func:`build_evaluation_context`.

    Returns
    dict
        Keys: ``objective``, ``ray_rmse``, ``len_rmse``, ``proj_rmse``,
        ``success``, ``error``, plus extended diagnostics.
    """
    # --- default to zero delta (evaluate reference) ---
    if x_delta is None:
        x_delta = np.zeros(ctx.n_params, dtype=np.float64)
    else:
        x_delta = np.asarray(x_delta, dtype=np.float64)
        if x_delta.shape != (ctx.n_params,):
            return _candidate_result_dict(
                objective=_SENTINEL_OBJECTIVE,
                ray_rmse=np.inf, len_rmse=np.inf, proj_rmse=np.inf,
                n_ray=0, n_len=0, n_proj=0,
                s_ray=np.inf, s_len=np.inf,
                ok=False,
                failure_reason=(
                    f"x_delta shape {x_delta.shape} != expected ({ctx.n_params},)"
                ),
            )

    # --- guard: non-finite input ---
    if not np.all(np.isfinite(x_delta)):
        return _candidate_result_dict(
            objective=_SENTINEL_OBJECTIVE,
            ray_rmse=np.inf, len_rmse=np.inf, proj_rmse=np.inf,
            n_ray=0, n_len=0, n_proj=0,
            s_ray=np.inf, s_len=np.inf,
            ok=False,
            failure_reason="x_delta contains NaN/Inf",
        )

    opt = ctx.optimizer

    try:
        # 1. Decode deltas into concrete planes / cam_params / media
        curr_planes, curr_cams, curr_media = opt._unpack_params_delta(
            x_delta, ctx.layout,
        )

        # 2. Evaluate BA residuals (syncs C++ cameras internally)
        (residuals, S_ray, S_len, N_ray, N_len,
         S_proj, N_proj) = opt.evaluate_residuals(
            curr_planes, curr_cams, ctx.lambda_eff,
            window_media=curr_media,
        )

    except Exception as exc:
        logger.warning("evaluate_candidate failed: %s", exc, exc_info=True)
        return _candidate_result_dict(
            objective=_SENTINEL_OBJECTIVE,
            ray_rmse=np.inf, len_rmse=np.inf, proj_rmse=np.inf,
            n_ray=0, n_len=0, n_proj=0,
            s_ray=np.inf, s_len=np.inf,
            ok=False,
            failure_reason=f"Exception: {exc!r}",
        )

    # --- guard: non-finite sums ---
    if not (np.isfinite(S_ray) and np.isfinite(S_len)):
        return _candidate_result_dict(
            objective=_SENTINEL_OBJECTIVE,
            ray_rmse=np.inf, len_rmse=np.inf, proj_rmse=np.inf,
            n_ray=int(N_ray), n_len=int(N_len), n_proj=int(N_proj),
            s_ray=float(S_ray), s_len=float(S_len),
            ok=False,
            failure_reason="S_ray or S_len is non-finite",
        )

    # --- compute RMSE diagnostics ---
    ray_rmse = float(np.sqrt(S_ray / max(N_ray, 1)))
    len_rmse = float(np.sqrt(S_len / max(N_len, 1))) if N_len > 0 else 0.0
    proj_rmse = float(np.sqrt(S_proj / max(N_proj, 1))) if N_proj > 0 else 0.0

    # Scalar objective: data cost = S_ray + lambda_eff * S_len
    objective = float(S_ray + ctx.lambda_eff * S_len)

    if not np.isfinite(objective):
        return _candidate_result_dict(
            objective=_SENTINEL_OBJECTIVE,
            ray_rmse=ray_rmse, len_rmse=len_rmse, proj_rmse=proj_rmse,
            n_ray=int(N_ray), n_len=int(N_len), n_proj=int(N_proj),
            s_ray=float(S_ray), s_len=float(S_len),
            ok=False,
            failure_reason="objective is non-finite",
        )

    return _candidate_result_dict(
        objective=objective,
        ray_rmse=ray_rmse,
        len_rmse=len_rmse,
        proj_rmse=proj_rmse,
        n_ray=int(N_ray),
        n_len=int(N_len),
        n_proj=int(N_proj),
        s_ray=float(S_ray),
        s_len=float(S_len),
        ok=True,
    )


# ===========================================================================
#  Task 3: Search Parameter Vector Layout & 1-D Scale Probing
# ===========================================================================

# ---------------------------------------------------------------------------
#  Parameter layout descriptor
# ---------------------------------------------------------------------------

@dataclass
class SearchParamEntry:
    """Descriptor for a single scalar parameter in the search vector."""
    index: int              # position in the flat search vector
    ptype: str              # layout type tag: 'plane_d', 'plane_a', 'plane_b',
                            #                  'cam_t', 'cam_r'
    entity_id: int          # window_id or cam_id
    sub_index: int          # sub-parameter index within type (e.g. 0/1/2 for xyz)
    label: str              # human-readable label, e.g. 'win0_d', 'cam3_ty'
    group: str              # 'plane' or 'extrinsic'


@dataclass
class SearchParameterLayout:
    """Explicit description of the global-search parameter vector.

    Built from the BA-compatible layout in :class:`EvaluationContext`.
    Provides human-readable labels, grouping, and index lookup for every
    scalar component of the flat delta vector.
    """
    entries: List[SearchParamEntry]
    n_params: int
    n_plane_params: int
    n_extrinsic_params: int
    cam_ids: List[int]
    window_ids: List[int]

    # Index ranges for quick slicing
    plane_indices: List[int] = field(default_factory=list)
    extrinsic_indices: List[int] = field(default_factory=list)

    def labels(self) -> List[str]:
        """Return ordered list of human-readable parameter labels."""
        return [e.label for e in self.entries]

    def indices_for_group(self, group: str) -> List[int]:
        """Return indices belonging to *group* ('plane' or 'extrinsic')."""
        return [e.index for e in self.entries if e.group == group]

    def indices_for_entity(self, entity_id: int) -> List[int]:
        """Return indices tied to a specific camera or window id."""
        return [e.index for e in self.entries if e.entity_id == entity_id]


_PLANE_SUB_LABELS = {0: 'd', 1: 'a', 2: 'b'}
_CAM_T_SUB_LABELS = {0: 'tx', 1: 'ty', 2: 'tz'}
_CAM_R_SUB_LABELS = {0: 'rx', 1: 'ry', 2: 'rz'}


def build_search_parameter_layout(
    ctx: EvaluationContext,
) -> SearchParameterLayout:
    """Build an explicit :class:`SearchParameterLayout` from an evaluation context.

    The layout mirrors ``ctx.layout`` but enriches every scalar parameter with
    a human-readable label, group tag, and entity id so that downstream code
    (CMA-ES driver, diagnostics) can reason about the search vector without
    reverse-engineering BA internals.

    Parameters
    ----------
    ctx : EvaluationContext
        As returned by :func:`build_evaluation_context`.

    Returns
    -------
    SearchParameterLayout
    """
    entries: List[SearchParamEntry] = []
    plane_indices: List[int] = []
    extrinsic_indices: List[int] = []

    for idx, (ptype, entity_id, sub_idx) in enumerate(ctx.layout):
        if ptype == 'plane_d':
            label = f'win{entity_id}_d'
            group = 'plane'
        elif ptype == 'plane_a':
            label = f'win{entity_id}_a'
            group = 'plane'
        elif ptype == 'plane_b':
            label = f'win{entity_id}_b'
            group = 'plane'
        elif ptype == 'cam_t':
            sub_lbl = _CAM_T_SUB_LABELS.get(sub_idx, f't{sub_idx}')
            label = f'cam{entity_id}_{sub_lbl}'
            group = 'extrinsic'
        elif ptype == 'cam_r':
            sub_lbl = _CAM_R_SUB_LABELS.get(sub_idx, f'r{sub_idx}')
            label = f'cam{entity_id}_{sub_lbl}'
            group = 'extrinsic'
        else:
            label = f'{ptype}_{entity_id}_{sub_idx}'
            group = 'other'

        entry = SearchParamEntry(
            index=idx,
            ptype=ptype,
            entity_id=entity_id,
            sub_index=sub_idx,
            label=label,
            group=group,
        )
        entries.append(entry)

        if group == 'plane':
            plane_indices.append(idx)
        elif group == 'extrinsic':
            extrinsic_indices.append(idx)

    return SearchParameterLayout(
        entries=entries,
        n_params=len(entries),
        n_plane_params=len(plane_indices),
        n_extrinsic_params=len(extrinsic_indices),
        cam_ids=list(ctx.cam_ids),
        window_ids=list(ctx.window_ids),
        plane_indices=plane_indices,
        extrinsic_indices=extrinsic_indices,
    )


# ---------------------------------------------------------------------------
#  1-D scale probing
# ---------------------------------------------------------------------------

# Default probe step sizes by parameter type.
# Chosen conservatively to measure local curvature without leaving the
# basin of convergence around the BA reference point.
_DEFAULT_PROBE_STEPS: Dict[str, float] = {
    'plane_d': 0.5,       # mm — signed distance along normal
    'plane_a': 0.01,      # rad — normal tilt angle alpha
    'plane_b': 0.01,      # rad — normal tilt angle beta
    'cam_t':   0.5,       # mm — camera translation component
    'cam_r':   0.005,     # rad — camera rotation component (~0.3°)
}
_DEFAULT_PROBE_STEP_FALLBACK = 0.01


@dataclass
class ProbeResult:
    """Outcome of 1-D scale probing for the global search vector.

    Attributes
    ----------
    scales : np.ndarray
        Per-parameter scale estimates, shape ``(n_params,)``.
        For CMA-ES this is used as ``sigma0 * scales``.
    sensitivities : np.ndarray
        Per-parameter |d(objective)/d(param)| estimate at the reference.
    ref_objective : float
        Objective value at the reference state (zero delta).
    labels : List[str]
        Human-readable labels matching the scales vector.
    n_evals : int
        Total number of ``evaluate_candidate`` calls made.
    wall_seconds : float
        Total wall-clock time spent probing.
    early_stop_reason : str
        Empty string if probing completed normally; otherwise describes
        which guardrail triggered early termination.
    param_layout : SearchParameterLayout
        The layout object used for probing.
    block_scales : np.ndarray
        Stage-2 block-direction scale contributions per parameter, shape
        ``(n_params,)``. Zero when Stage-2 is skipped.
    block_probe_summary : str
        Human-readable Stage-2 block probing summary / stop reasons.
    """
    scales: np.ndarray
    sensitivities: np.ndarray
    ref_objective: float
    labels: List[str]
    n_evals: int
    wall_seconds: float
    early_stop_reason: str
    param_layout: SearchParameterLayout
    block_scales: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    block_probe_summary: str = ''


@dataclass
class ProbeDirection:
    """One normalized probing direction inside a dynamic parameter block."""
    name: str
    full_vector: np.ndarray


@dataclass
class ProbeBlock:
    """Dynamic coupled block definition derived from layout/topology."""
    name: str
    window_id: int
    kind: str
    param_indices: List[int]
    param_labels: List[str]
    directions: List[ProbeDirection]


@dataclass
class BlockProbeResult:
    """Outcome of Stage-2 dynamic block directional probing."""
    block_scales: np.ndarray
    n_evals: int
    wall_seconds: float
    early_stop_reason: str


def probe_scales(
    ctx: EvaluationContext,
    *,
    probe_steps: Optional[Dict[str, float]] = None,
    max_evals: int = 500,
    max_wall_seconds: float = float('inf'),
    min_scale: float = 1e-8,
    max_scale: float = 100.0,
) -> ProbeResult:
    """Estimate per-parameter scales via 1-D central-difference probing.

    For each dimension *i* of the search vector, two evaluations are made
    at ``x_ref ± h_i * e_i`` (central difference).  The second derivative
    (curvature) along that axis is approximated and the scale is set to
    ``1 / sqrt(curvature)`` — i.e. the perturbation magnitude that changes
    the objective by ~1 unit.

    When curvature is near-zero or negative (flat / noisy landscape) the
    scale is clamped to ``[min_scale, max_scale]``.

    Parameters
    ----------
    ctx : EvaluationContext
        As returned by :func:`build_evaluation_context`.
    probe_steps : dict, optional
        Override for per-type step sizes.  Keys are parameter types
        (``'plane_d'``, ``'cam_t'``, …); missing types fall back to
        built-in defaults.
    max_evals : int
        Maximum total calls to ``evaluate_candidate``.  Probing stops
        early (and returns partial scales) if this limit is hit.
    max_wall_seconds : float
        Maximum wall-clock time (seconds) before early termination.
    min_scale : float
        Floor for any single scale value.
    max_scale : float
        Ceiling for any single scale value.

    Returns
    -------
    ProbeResult
        Contains scales vector, sensitivities, metadata, and the
        :class:`SearchParameterLayout` used.
    """
    import time as _time

    layout = build_search_parameter_layout(ctx)
    n = layout.n_params

    # Merge probe step overrides with defaults
    steps = dict(_DEFAULT_PROBE_STEPS)
    if probe_steps is not None:
        steps.update(probe_steps)

    # Build per-parameter step vector
    h_vec = np.empty(n, dtype=np.float64)
    for entry in layout.entries:
        h_vec[entry.index] = steps.get(entry.ptype, _DEFAULT_PROBE_STEP_FALLBACK)

    # --- Evaluate reference state ---
    t0 = _time.monotonic()
    n_evals = 0
    early_stop = ''

    ref_result = evaluate_candidate(None, ctx)
    n_evals += 1
    if not ref_result['success']:
        raise RuntimeError(
            f'Reference-state evaluation failed: {ref_result["error"]}'
        )
    f0 = ref_result['objective']

    scales = np.full(n, max_scale, dtype=np.float64)
    sensitivities = np.zeros(n, dtype=np.float64)

    # --- Probe each dimension ---
    for i in range(n):
        # Guardrails
        elapsed = _time.monotonic() - t0
        if elapsed > max_wall_seconds:
            early_stop = f'wall_time_exceeded ({elapsed:.1f}s > {max_wall_seconds}s)'
            logger.warning('probe_scales: early stop — %s (dim %d/%d)', early_stop, i, n)
            break
        if n_evals + 2 > max_evals:  # need 2 evals per dimension
            early_stop = f'max_evals_reached ({n_evals} + 2 > {max_evals})'
            logger.warning('probe_scales: early stop — %s (dim %d/%d)', early_stop, i, n)
            break

        h_i = h_vec[i]

        # +h probe
        x_plus = np.zeros(n, dtype=np.float64)
        x_plus[i] = +h_i
        res_plus = evaluate_candidate(x_plus, ctx)
        n_evals += 1

        # -h probe
        x_minus = np.zeros(n, dtype=np.float64)
        x_minus[i] = -h_i
        res_minus = evaluate_candidate(x_minus, ctx)
        n_evals += 1

        f_plus = res_plus['objective'] if res_plus['success'] else _SENTINEL_OBJECTIVE
        f_minus = res_minus['objective'] if res_minus['success'] else _SENTINEL_OBJECTIVE

        # Central-difference curvature: d2f/dx2 ≈ (f+ - 2*f0 + f-) / h^2
        curvature = (f_plus - 2.0 * f0 + f_minus) / (h_i * h_i)

        # Central-difference gradient magnitude: |df/dx| ≈ |f+ - f-| / (2*h)
        grad_mag = abs(f_plus - f_minus) / (2.0 * h_i)
        sensitivities[i] = grad_mag

        # Scale = 1 / sqrt(curvature) — perturbation for unit objective change
        if np.isfinite(curvature) and curvature > 1e-20:
            raw_scale = 1.0 / np.sqrt(curvature)
            scales[i] = float(np.clip(raw_scale, min_scale, max_scale))
        else:
            # Flat or noisy — use step size as fallback scale
            scales[i] = float(np.clip(h_i, min_scale, max_scale))

    elapsed_total = _time.monotonic() - t0

    logger.info(
        'probe_scales complete: %d/%d dims probed, %d evals, %.1fs, '
        'ref_obj=%.4f, scale_range=[%.2e, %.2e]',
        min(i + 1 if early_stop == '' else i, n), n, n_evals, elapsed_total,
        f0, float(np.min(scales)), float(np.max(scales)),
    )

    return ProbeResult(
        scales=scales,
        sensitivities=sensitivities,
        ref_objective=f0,
        labels=layout.labels(),
        n_evals=n_evals,
        wall_seconds=elapsed_total,
        early_stop_reason=early_stop,
        param_layout=layout,
    )


def _probe_stage1_single_param(
    shared_setup: Dict[str, Any],
    param_idx: int,
    layout_entry,  # SearchParameterLayoutEntry
    base_h: np.ndarray,
    ray_rmse_ref: float,
    ray_stop_threshold: float,
    max_evals: int,
    max_wall_seconds: float,
    min_scale: float,
    max_scale: float,
    max_alpha_steps: int,
    enable_compensation: bool,
    max_compensation_iters: int,
    alpha_growth: float,
) -> Dict[str, Any]:
    """Worker function for single-parameter Stage 1 probing.
    
    Performs alpha-expansion loop for a single parameter to determine
    its scale and sensitivity via compensated 1-D probing.
    
    Parameters
    ----------
    shared_setup : dict
        Serializable setup from :func:`build_shared_setup`.
    param_idx : int
        Index of parameter to probe.
    layout_entry : SearchParameterLayoutEntry
        Layout entry for this parameter.
    base_h : np.ndarray
        Initial step sizes for all parameters.
    ray_rmse_ref : float
        Reference ray RMSE (computed in main process).
    ray_stop_threshold : float
        Ray RMSE threshold to stop expansion.
    max_evals : int
        Budget for evaluations.
    max_wall_seconds : float
        Wall time budget.
    min_scale : float
        Minimum step size.
    max_scale : float
        Maximum step size.
    max_alpha_steps : int
        Maximum alpha iterations.
    enable_compensation : bool
        Whether to enable compensation.
    max_compensation_iters : int
        Maximum compensation iterations.
    alpha_growth : float
        Alpha growth factor per iteration.
    
    Returns
    -------
    dict
        Keys: 'param_idx', 'scale', 'sensitivity', 'stop_reason', 'n_evals'
    """
    import time as _time
    
    runtime = initialize_worker_evaluation_runtime(shared_setup)
    ctx = runtime.ctx
    
    t0 = _time.monotonic()
    n_evals = 0
    n = len(base_h)
    
    safe_step = float(np.clip(base_h[param_idx], min_scale, max_scale))
    last_safe_ray = ray_rmse_ref
    reason = 'alpha_step_limit_reached (%d)' % max_alpha_steps
    alpha = 1.0
    
    for alpha_step in range(max_alpha_steps):
        elapsed = _time.monotonic() - t0
        if elapsed > max_wall_seconds:
            reason = 'wall_time_exceeded (%.1fs > %.1fs)' % (elapsed, max_wall_seconds)
            break
        if n_evals + 1 > max_evals:
            reason = 'max_evals_reached (%d + 1 > %d)' % (n_evals, max_evals)
            break
        
        step = base_h[param_idx] * alpha
        if step > max_scale:
            reason = 'max_scale_cap_reached (%.3g > %.3g)' % (step, max_scale)
            break
        
        x_probe = np.zeros(n, dtype=np.float64)
        x_probe[param_idx] = step
        
        comp = evaluate_probe_step_with_compensation(
            x_probe,
            ctx,
            locked_param_indices=[param_idx],
            enable_compensation=enable_compensation,
            max_compensation_iters=max_compensation_iters,
        )
        n_evals += 1
        
        logger.debug(
            'probe_stage1_worker: [%d] %s alpha_step=%d/%d step=%.6g '
            'ray_rmse=%.6g (ref=%.6g) valid=%s',
            param_idx,
            layout_entry.label,
            alpha_step + 1,
            max_alpha_steps,
            step,
            float(comp.ray_rmse) if comp.is_valid else float('nan'),
            ray_rmse_ref,
            comp.is_valid,
        )
        
        if not comp.is_valid:
            reason = 'invalid_geometry_or_eval (%s)' % (comp.failure_reason or 'unknown')
            break
        
        ray_now = float(comp.ray_rmse)
        if not np.isfinite(ray_now):
            reason = 'non_finite_compensated_ray_rmse'
            break
        
        if ray_now >= ray_stop_threshold:
            reason = 'compensated_ray_rmse_stop (%.6g >= %.6g)' % (ray_now, ray_stop_threshold)
            break
        
        safe_step = float(np.clip(abs(step), min_scale, max_scale))
        last_safe_ray = ray_now
        alpha *= alpha_growth
    
    final_scale = max(safe_step, min_scale)
    final_sensitivity = abs(last_safe_ray - ray_rmse_ref) / max(final_scale, min_scale)
    
    logger.debug(
        'probe_stage1_worker: [%d] %s DONE scale=%.6g ray_rmse=%.6g (ref=%.6g) stop_reason=%s',
        param_idx,
        layout_entry.label,
        final_scale,
        last_safe_ray,
        ray_rmse_ref,
        reason,
    )
    
    return {
        'param_idx': param_idx,
        'scale': final_scale,
        'sensitivity': final_sensitivity,
        'stop_reason': reason,
        'n_evals': n_evals,
    }


def probe_scales_multidim_stage1(
    ctx: EvaluationContext,
    *,
    probe_steps: Optional[Dict[str, float]] = None,
    shared_setup: Optional[Dict[str, Any]] = None,
    max_evals: int = 500,
    max_wall_seconds: float = float('inf'),
    min_scale: float = 1e-8,
    max_scale: float = 100.0,
    ray_rmse_stop_factor: float = 1.1,
    enable_compensation: bool = True,
    max_compensation_iters: int = 3,
    alpha_growth: float = 2.0,
    max_alpha_steps: int = 12,
) -> ProbeResult:
    """Stage-1 multidimensional probing via compensated 1-D alpha expansion.

    For each parameter axis, this probes progressively larger perturbations
    (``h = base_step * alpha``), runs limited compensation with the probed
    parameter locked, and records the largest safe perturbation.

    Stop condition for each axis is based on compensated Ray-RMSE only:
    ``compensated_ray_rmse >= ray_rmse_stop_factor * ray_rmse_ref``.
    Geometry invalidation or evaluation failure also stops that axis.
    """
    import time as _time
    import os
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from concurrent.futures.process import BrokenProcessPool

    layout = build_search_parameter_layout(ctx)
    n = layout.n_params

    if alpha_growth <= 1.0:
        raise ValueError(f'alpha_growth must be > 1.0, got {alpha_growth}')
    if max_alpha_steps <= 0:
        raise ValueError(f'max_alpha_steps must be > 0, got {max_alpha_steps}')

    steps = dict(_DEFAULT_PROBE_STEPS)
    if probe_steps is not None:
        steps.update(probe_steps)

    base_h = np.empty(n, dtype=np.float64)
    for entry in layout.entries:
        step_i = float(steps.get(entry.ptype, _DEFAULT_PROBE_STEP_FALLBACK))
        if not np.isfinite(step_i) or abs(step_i) <= 0.0:
            step_i = _DEFAULT_PROBE_STEP_FALLBACK
        base_h[entry.index] = abs(step_i)

    # --- Auto-detection and parallel decision ---
    _cpu = os.cpu_count() or 1
    max_workers = min(32, max(1, int(_cpu * 0.8)))
    n_params = len(layout.entries)
    # Relaxed threshold: enable parallel if we have enough items to distribute
    # Original: n_params >= 2 * max_workers (conservative, high overhead threshold)
    # New: n_params >= max_workers (enable parallel when items >= workers)
    use_parallel = (shared_setup is not None and max_workers > 1 and n_params >= max_workers)

    logger.info(
        'probe_scales_multidim_stage1: %d params, cpu_count=%d, max_workers=%d, parallel=%s',
        n_params, _cpu, max_workers, use_parallel
    )

    # --- Compute reference eval once (needed for both sequential and parallel) ---
    ref_result = evaluate_candidate(None, ctx)
    n_evals = 1
    if not ref_result['success']:
        raise RuntimeError(
            f'Reference-state evaluation failed: {ref_result["error"]}'
        )

    f0 = float(ref_result['objective'])
    ray_rmse_ref = float(ref_result['ray_rmse'])
    if not np.isfinite(ray_rmse_ref) or ray_rmse_ref <= 0.0:
        ray_rmse_ref = 1e-12
    ray_stop_threshold = float(max(ray_rmse_stop_factor, 1.0) * ray_rmse_ref)

    logger.debug(
        'probe_scales_multidim_stage1: starting — n_params=%d max_evals=%d '
        'max_wall_seconds=%s ray_rmse_ref=%.6g ray_stop_threshold=%.6g',
        n,
        max_evals,
        'unlimited' if max_wall_seconds == float('inf') else f'{max_wall_seconds:.0f}s',
        ray_rmse_ref,
        ray_stop_threshold,
    )

    if not use_parallel:
        # --- Sequential path (original loop) ---
        t0 = _time.monotonic()
        early_stop = ''
        scales = np.full(n, min_scale, dtype=np.float64)
        sensitivities = np.zeros(n, dtype=np.float64)
        param_stop_reasons: List[str] = [''] * n

        for i, entry in enumerate(layout.entries):
            if i % 5 == 0 or i == n - 1:
                elapsed = _time.monotonic() - t0
                logger.debug(
                    'probe_scales_multidim_stage1: param %d/%d (%s) elapsed=%.1fs evals=%d',
                    i + 1,
                    n,
                    entry.label,
                    elapsed,
                    n_evals,
                )

            elapsed = _time.monotonic() - t0
            if elapsed > max_wall_seconds:
                early_stop = f'wall_time_exceeded ({elapsed:.1f}s > {max_wall_seconds}s)'
                logger.warning(
                    'probe_scales_multidim_stage1: early stop - %s (dim %d/%d)',
                    early_stop, i, n,
                )
                break
            if n_evals + 1 > max_evals:
                early_stop = f'max_evals_reached ({n_evals} + 1 > {max_evals})'
                logger.warning(
                    'probe_scales_multidim_stage1: early stop - %s (dim %d/%d)',
                    early_stop, i, n,
                )
                break

            safe_step = float(np.clip(base_h[i], min_scale, max_scale))
            last_safe_ray = ray_rmse_ref
            reason = f'alpha_step_limit_reached ({max_alpha_steps})'
            alpha = 1.0

            for alpha_step in range(max_alpha_steps):
                elapsed = _time.monotonic() - t0
                if elapsed > max_wall_seconds:
                    early_stop = f'wall_time_exceeded ({elapsed:.1f}s > {max_wall_seconds}s)'
                    reason = early_stop
                    break
                if n_evals + 1 > max_evals:
                    early_stop = f'max_evals_reached ({n_evals} + 1 > {max_evals})'
                    reason = early_stop
                    break

                step = base_h[i] * alpha
                if step > max_scale:
                    reason = f'max_scale_cap_reached ({step:.3g} > {max_scale:.3g})'
                    break

                x_probe = np.zeros(n, dtype=np.float64)
                x_probe[i] = step

                comp = evaluate_probe_step_with_compensation(
                    x_probe,
                    ctx,
                    locked_param_indices=[i],
                    enable_compensation=enable_compensation,
                    max_compensation_iters=max_compensation_iters,
                )
                n_evals += 1
                logger.debug(
                    'probe_scales_multidim_stage1: [%d/%d] %s alpha_step=%d/%d '
                    'step=%.6g ray_rmse=%.6g (ref=%.6g) valid=%s',
                    i + 1,
                    n,
                    entry.label,
                    alpha_step + 1,
                    max_alpha_steps,
                    step,
                    float(comp.ray_rmse) if comp.is_valid else float('nan'),
                    ray_rmse_ref,
                    comp.is_valid,
                )

                if not comp.is_valid:
                    reason = f'invalid_geometry_or_eval ({comp.failure_reason or "unknown"})'
                    break

                ray_now = float(comp.ray_rmse)
                if not np.isfinite(ray_now):
                    reason = 'non_finite_compensated_ray_rmse'
                    break

                if ray_now >= ray_stop_threshold:
                    reason = (
                        'compensated_ray_rmse_stop '
                        f'({ray_now:.6g} >= {ray_stop_threshold:.6g})'
                    )
                    break

                safe_step = float(np.clip(abs(step), min_scale, max_scale))
                last_safe_ray = ray_now
                alpha *= alpha_growth

            scales[i] = max(safe_step, min_scale)
            sensitivities[i] = abs(last_safe_ray - ray_rmse_ref) / max(scales[i], min_scale)
            param_stop_reasons[i] = reason

            logger.debug(
                'probe_scales_multidim_stage1: [%d/%d] %s scale=%.6g '
                'ray_rmse=%.6g (ref=%.6g) stop_reason=%s',
                i + 1,
                n,
                entry.label,
                scales[i],
                last_safe_ray,
                ray_rmse_ref,
                reason,
            )

            if early_stop:
                break

        elapsed_total = _time.monotonic() - t0

        if not early_stop:
            reason_examples = [
                f'{layout.entries[idx].label}:{r}'
                for idx, r in enumerate(param_stop_reasons)
                if r.startswith('compensated_ray_rmse_stop') or r.startswith('invalid_geometry_or_eval')
            ]
            if reason_examples:
                early_stop = 'stage1_stop_reasons: ' + ' | '.join(reason_examples[:6])

        logger.info(
            'probe_scales_multidim_stage1 complete: %d params, %d evals, %.1fs, '
            'ray_ref=%.6g, ray_stop=%.6g, scale_range=[%.2e, %.2e], stopped_by=%s',
            n,
            n_evals,
            elapsed_total,
            ray_rmse_ref,
            ray_stop_threshold,
            float(np.min(scales)),
            float(np.max(scales)),
            early_stop or 'normal',
        )

        return ProbeResult(
            scales=scales,
            sensitivities=sensitivities,
            ref_objective=f0,
            labels=layout.labels(),
            n_evals=n_evals,
            wall_seconds=elapsed_total,
            early_stop_reason=early_stop,
            param_layout=layout,
            block_scales=np.zeros(n, dtype=np.float64),
            block_probe_summary='',
        )

    else:
        # --- Parallel path ---
        scales = np.full(n, min_scale, dtype=np.float64)
        sensitivities = np.zeros(n, dtype=np.float64)
        param_stop_reasons = [''] * n

        t0 = _time.monotonic()
        n_evals_total = n_evals  # Already counted ref_result
        early_stop = None

        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp.get_context('spawn'),
            initializer=_init_probing_worker
        ) as executor:
            futures = {
                executor.submit(
                    _probe_stage1_single_param,
                    shared_setup,
                    i,
                    layout.entries[i],
                    base_h,
                    ray_rmse_ref,
                    ray_stop_threshold,
                    max_evals,
                    max_wall_seconds,
                    min_scale,
                    max_scale,
                    max_alpha_steps,
                    enable_compensation,
                    max_compensation_iters,
                    alpha_growth,
                ): i
                for i in range(n_params)
            }

            for future in as_completed(futures):
                result = future.result()
                param_idx = result['param_idx']
                scales[param_idx] = result['scale']
                sensitivities[param_idx] = result['sensitivity']
                param_stop_reasons[param_idx] = result['stop_reason']
                n_evals_total += result['n_evals']

        elapsed_total = _time.monotonic() - t0

        # Create ProbeResult (same structure as sequential path)
        logger.info(
            'probe_scales_multidim_stage1 complete (parallel): %d params, %d evals, %.1fs, '
            'ray_ref=%.6g, ray_stop=%.6g, scale_range=[%.2e, %.2e]',
            n,
            n_evals_total,
            elapsed_total,
            ray_rmse_ref,
            ray_stop_threshold,
            float(np.min(scales)),
            float(np.max(scales)),
        )

        return ProbeResult(
            scales=scales,
            sensitivities=sensitivities,
            ref_objective=f0,
            labels=layout.labels(),
            n_evals=n_evals_total,
            wall_seconds=elapsed_total,
            early_stop_reason='',
            param_layout=layout,
            block_scales=np.full(n, min_scale, dtype=np.float64),
            block_probe_summary='stage2_disabled (parallel stage1)',
        )


def _normalize_direction_vector(v: np.ndarray, *, tol: float = 1e-15) -> Optional[np.ndarray]:
    """Return a normalized copy of ``v`` (or ``None`` if degenerate)."""
    norm = float(np.linalg.norm(v))
    if not np.isfinite(norm) or norm <= tol:
        return None
    return v / norm


def _build_dynamic_probe_blocks(
    layout: SearchParameterLayout,
    *,
    cam_to_window: Dict[int, int],
) -> List[ProbeBlock]:
    """Build Stage-2 coupled probing blocks from actual layout/topology."""
    n = layout.n_params
    entry_by_index = {e.index: e for e in layout.entries}

    plane_d_idx_by_wid: Dict[int, int] = {}
    plane_a_idx_by_wid: Dict[int, int] = {}
    plane_b_idx_by_wid: Dict[int, int] = {}
    cam_tz_idx_by_cam: Dict[int, int] = {}
    cam_rx_idx_by_cam: Dict[int, int] = {}
    cam_ry_idx_by_cam: Dict[int, int] = {}
    for e in layout.entries:
        if e.ptype == 'plane_d':
            plane_d_idx_by_wid[e.entity_id] = e.index
        elif e.ptype == 'plane_a':
            plane_a_idx_by_wid[e.entity_id] = e.index
        elif e.ptype == 'plane_b':
            plane_b_idx_by_wid[e.entity_id] = e.index
        elif e.ptype == 'cam_t' and e.sub_index == 2:
            cam_tz_idx_by_cam[e.entity_id] = e.index
        elif e.ptype == 'cam_r' and e.sub_index == 0:
            cam_rx_idx_by_cam[e.entity_id] = e.index
        elif e.ptype == 'cam_r' and e.sub_index == 1:
            cam_ry_idx_by_cam[e.entity_id] = e.index

    cams_by_window: Dict[int, List[int]] = {}
    active_cams = set(layout.cam_ids)
    for cam_id, win_id in cam_to_window.items():
        if cam_id not in active_cams:
            continue
        cams_by_window.setdefault(int(win_id), []).append(int(cam_id))
    for win_id in cams_by_window:
        cams_by_window[win_id] = sorted(set(cams_by_window[win_id]))

    blocks: List[ProbeBlock] = []

    def _make_block(
        name: str,
        *,
        window_id: int,
        kind: str,
        plane_idx: int,
        coupled_cam_indices: List[int],
    ) -> Optional[ProbeBlock]:
        if plane_idx < 0 or not coupled_cam_indices:
            return None
        block_indices = [int(plane_idx)] + [int(i) for i in coupled_cam_indices]
        if len(block_indices) < 2:
            return None

        labels = [entry_by_index[i].label for i in block_indices]
        directions: List[ProbeDirection] = []

        v_plane_only = np.zeros(n, dtype=np.float64)
        v_plane_only[plane_idx] = 1.0
        nv = _normalize_direction_vector(v_plane_only)
        if nv is not None:
            directions.append(ProbeDirection(name='plane_only', full_vector=nv))

        v_cams_only = np.zeros(n, dtype=np.float64)
        for idx in coupled_cam_indices:
            v_cams_only[idx] = 1.0
        nv = _normalize_direction_vector(v_cams_only)
        if nv is not None:
            directions.append(ProbeDirection(name='cams_only', full_vector=nv))

        v_coupled_minus = np.zeros(n, dtype=np.float64)
        v_coupled_minus[plane_idx] = 1.0
        cam_w = -1.0 / float(len(coupled_cam_indices))
        for idx in coupled_cam_indices:
            v_coupled_minus[idx] = cam_w
        nv = _normalize_direction_vector(v_coupled_minus)
        if nv is not None:
            directions.append(ProbeDirection(name='coupled_minus', full_vector=nv))

        v_coupled_plus = np.zeros(n, dtype=np.float64)
        v_coupled_plus[plane_idx] = 1.0
        cam_w = 1.0 / float(len(coupled_cam_indices))
        for idx in coupled_cam_indices:
            v_coupled_plus[idx] = cam_w
        nv = _normalize_direction_vector(v_coupled_plus)
        if nv is not None:
            directions.append(ProbeDirection(name='coupled_plus', full_vector=nv))

        if not directions:
            return None
        return ProbeBlock(
            name=name,
            window_id=int(window_id),
            kind=kind,
            param_indices=block_indices,
            param_labels=labels,
            directions=directions,
        )

    all_windows = sorted(set(layout.window_ids) | set(cams_by_window.keys()))
    for wid in all_windows:
        win_cams = cams_by_window.get(wid, [])

        depth_cam_indices = [
            cam_tz_idx_by_cam[cid]
            for cid in win_cams
            if cid in cam_tz_idx_by_cam
        ]
        blk = _make_block(
            f'win{wid}_depth',
            window_id=wid,
            kind='depth',
            plane_idx=plane_d_idx_by_wid.get(wid, -1),
            coupled_cam_indices=depth_cam_indices,
        )
        if blk is not None:
            blocks.append(blk)

        tilt_x_cam_indices = [
            cam_rx_idx_by_cam[cid]
            for cid in win_cams
            if cid in cam_rx_idx_by_cam
        ]
        blk = _make_block(
            f'win{wid}_tilt_a_rx',
            window_id=wid,
            kind='tilt_a',
            plane_idx=plane_a_idx_by_wid.get(wid, -1),
            coupled_cam_indices=tilt_x_cam_indices,
        )
        if blk is not None:
            blocks.append(blk)

        tilt_y_cam_indices = [
            cam_ry_idx_by_cam[cid]
            for cid in win_cams
            if cid in cam_ry_idx_by_cam
        ]
        blk = _make_block(
            f'win{wid}_tilt_b_ry',
            window_id=wid,
            kind='tilt_b',
            plane_idx=plane_b_idx_by_wid.get(wid, -1),
            coupled_cam_indices=tilt_y_cam_indices,
        )
        if blk is not None:
            blocks.append(blk)

    return blocks


def build_multidim_probe_blocks(
    ctx: EvaluationContext,
    *,
    layout: Optional[SearchParameterLayout] = None,
) -> List[Dict[str, Any]]:
    """Return serializable Stage-2 dynamic block definitions for inspection."""
    if layout is None:
        layout = build_search_parameter_layout(ctx)
    cam_to_window = dict(getattr(ctx.optimizer, 'cam_to_window', {}))
    blocks = _build_dynamic_probe_blocks(layout, cam_to_window=cam_to_window)
    return [
        {
            'name': b.name,
            'window_id': b.window_id,
            'kind': b.kind,
            'param_indices': list(b.param_indices),
            'param_labels': list(b.param_labels),
            'direction_names': [d.name for d in b.directions],
        }
        for b in blocks
    ]


def _probe_stage2_single_block(
    shared_setup,
    block_idx,
    block,
    ray_rmse_ref,
    ray_stop_threshold,
    max_evals,
    max_wall_seconds,
    min_scale,
    max_scale,
    max_alpha_steps,
    alpha_growth,
    enable_compensation,
    max_compensation_iters,
    base_h,
):
    """Worker function for single-block Stage 2 probing.
    
    Parameters
    ----------
    shared_setup : dict
        Shared setup dictionary containing optimizer state for worker.
    block_idx : int
        Index of block being probed.
    block : ProbeBlock
        Block definition containing directions and parameter indices.
    ray_rmse_ref : float
        Reference ray RMSE (computed in main process).
    ray_stop_threshold : float
        Ray RMSE threshold to stop expansion.
    max_evals : int
        Budget for evaluations.
    max_wall_seconds : float
        Wall time budget.
    min_scale : float
        Minimum step size.
    max_scale : float
        Maximum step size.
    max_alpha_steps : int
        Maximum alpha iterations per direction.
    alpha_growth : float
        Alpha growth factor per iteration.
    enable_compensation : bool
        Whether to enable compensation.
    max_compensation_iters : int
        Maximum compensation iterations.
    base_h : np.ndarray
        Initial step sizes for all parameters.
    
    Returns
    -------
    dict
        Keys: 'block_idx', 'block_scales', 'stop_reason', 'n_evals'
    """
    import time as _time
    
    runtime = initialize_worker_evaluation_runtime(shared_setup)
    ctx = runtime.ctx
    
    t0 = _time.monotonic()
    n_evals = 0
    
    block_scales = np.zeros(ctx.n_params, dtype=np.float64)
    locked_idx = list(block.param_indices)
    early_stop = ''
    
    for direction in block.directions:
        if early_stop:
            break
        
        logger.debug(
            'probe_stage2_worker: block=%s starting direction=%s',
            block.name,
            direction.name,
        )
        
        elapsed = _time.monotonic() - t0
        if elapsed > max_wall_seconds:
            early_stop = 'wall_time_exceeded (%.1fs > %.1fs)' % (elapsed, max_wall_seconds)
            break
        if n_evals + 1 > max_evals:
            early_stop = 'max_evals_reached (%d + 1 > %d)' % (n_evals, max_evals)
            break
        
        active = np.abs(direction.full_vector) > 0.0
        if not np.any(active):
            continue
        
        alpha_base_candidates = base_h[active] / np.maximum(np.abs(direction.full_vector[active]), 1e-15)
        alpha_base = float(np.min(alpha_base_candidates))
        if not np.isfinite(alpha_base) or alpha_base <= 0.0:
            alpha_base = _DEFAULT_PROBE_STEP_FALLBACK
        alpha_base = float(np.clip(alpha_base, min_scale, max_scale))
        
        safe_alpha = alpha_base
        reason = 'alpha_step_limit_reached (%d)' % max_alpha_steps
        alpha = alpha_base
        
        for _ in range(max_alpha_steps):
            elapsed = _time.monotonic() - t0
            if elapsed > max_wall_seconds:
                early_stop = 'wall_time_exceeded (%.1fs > %.1fs)' % (elapsed, max_wall_seconds)
                reason = early_stop
                break
            if n_evals + 1 > max_evals:
                early_stop = 'max_evals_reached (%d + 1 > %d)' % (n_evals, max_evals)
                reason = early_stop
                break
            
            per_param_step = np.abs(alpha * direction.full_vector)
            if float(np.max(per_param_step)) > max_scale:
                reason = 'max_scale_cap_reached (%.3g > %.3g)' % (float(np.max(per_param_step)), max_scale)
                break
            
            x_probe = alpha * direction.full_vector
            comp = evaluate_probe_step_with_compensation(
                x_probe,
                ctx,
                locked_param_indices=locked_idx,
                enable_compensation=enable_compensation,
                max_compensation_iters=max_compensation_iters,
            )
            n_evals += 1
            
            if not comp.is_valid:
                reason = 'invalid_geometry_or_eval (%s)' % (comp.failure_reason or 'unknown')
                break
            
            ray_now = float(comp.ray_rmse)
            if not np.isfinite(ray_now):
                reason = 'non_finite_compensated_ray_rmse'
                break
            
            # Log compensated metric BEFORE stop check (Fix 2: capture actual value even if stop triggers)
            logger.debug(
                'probe_stage2_worker: block=%s dir=%s BEFORE_STOP_CHECK alpha=%.6f '
                'comp.ray_rmse=%.6f ray_stop_threshold=%.6f check_result=%s is_valid=%s',
                block.name,
                direction.name,
                alpha,
                ray_now,
                ray_stop_threshold,
                'WILL_STOP' if ray_now >= ray_stop_threshold else 'continue',
                comp.is_valid,
            )
            
            if ray_now >= ray_stop_threshold:
                reason = 'compensated_ray_rmse_stop (%.6g >= %.6g)' % (ray_now, ray_stop_threshold)
                break
            
            safe_alpha = float(np.clip(alpha, min_scale, max_scale))
            alpha *= alpha_growth
            logger.debug(
                'probe_stage2_worker: block=%s dir=%s alpha_step safe_alpha=%.6g '
                'next_alpha=%.6g ray_now=%.6g n_evals=%d',
                block.name,
                direction.name,
                safe_alpha,
                alpha,
                ray_now,
                n_evals,
            )
        
        contrib = np.abs(direction.full_vector) * safe_alpha
        block_scales = np.maximum(block_scales, contrib)
        
        logger.debug(
            'probe_stage2_worker: block=%s dir=%s RESULT safe_alpha=%.6g '
            'reason=%s contrib_max=%.6g',
            block.name,
            direction.name,
            safe_alpha,
            reason,
            float(np.max(contrib)) if contrib.size else 0.0,
        )
    
    final_reason = early_stop if early_stop else 'completed'
    
    logger.debug(
        'probe_stage2_worker: block_idx=%d name=%s completed n_evals=%d '
        'block_scale_max=%.6g stop_reason=%s',
        block_idx,
        block.name,
        n_evals,
        float(np.max(block_scales)) if block_scales.size else 0.0,
        final_reason,
    )
    
    return {
        'block_idx': block_idx,
        'block_scales': block_scales.copy(),
        'stop_reason': final_reason,
        'n_evals': n_evals,
    }


def probe_scales_multidim_stage2_blocks(
    ctx: EvaluationContext,
    *,
    layout: Optional[SearchParameterLayout] = None,
    probe_steps: Optional[Dict[str, float]] = None,
    shared_setup: Optional[Dict[str, Any]] = None,
    max_evals: int = 500,
    max_wall_seconds: float = float('inf'),
    min_scale: float = 1e-8,
    max_scale: float = 100.0,
    ray_rmse_stop_factor: float = 1.1,
    enable_compensation: bool = True,
    max_compensation_iters: int = 3,
    alpha_growth: float = 2.0,
    max_alpha_steps: int = 12,
) -> BlockProbeResult:
    """Stage-2 dynamic block-direction probing with block-lock compensation."""
    import time as _time
    import os
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if layout is None:
        layout = build_search_parameter_layout(ctx)
    n = layout.n_params

    if alpha_growth <= 1.0:
        raise ValueError(f'alpha_growth must be > 1.0, got {alpha_growth}')
    if max_alpha_steps <= 0:
        raise ValueError(f'max_alpha_steps must be > 0, got {max_alpha_steps}')

    steps = dict(_DEFAULT_PROBE_STEPS)
    if probe_steps is not None:
        steps.update(probe_steps)

    base_h = np.empty(n, dtype=np.float64)
    for entry in layout.entries:
        step_i = float(steps.get(entry.ptype, _DEFAULT_PROBE_STEP_FALLBACK))
        if not np.isfinite(step_i) or abs(step_i) <= 0.0:
            step_i = _DEFAULT_PROBE_STEP_FALLBACK
        base_h[entry.index] = abs(step_i)

    # --- Compute reference eval once ---
    ref_result = evaluate_candidate(None, ctx)
    n_evals = 1
    if not ref_result['success']:
        raise RuntimeError(f"Reference-state evaluation failed: {ref_result['error']}")
    ray_rmse_ref = float(ref_result['ray_rmse'])
    if not np.isfinite(ray_rmse_ref) or ray_rmse_ref <= 0.0:
        ray_rmse_ref = 1e-12
    ray_stop_threshold = float(max(ray_rmse_stop_factor, 1.0) * ray_rmse_ref)

    logger.debug(
        'probe_scales_multidim_stage2: ref_eval ray_rmse=%.6g '
        'ray_stop_threshold=%.6g n_params=%d',
        ray_rmse_ref,
        ray_stop_threshold,
        n,
    )

    cam_to_window = dict(getattr(ctx.optimizer, 'cam_to_window', {}))
    blocks = _build_dynamic_probe_blocks(layout, cam_to_window=cam_to_window)

    # --- Auto-detection and parallel decision ---
    _cpu = os.cpu_count() or 1
    max_workers = min(32, max(1, int(_cpu * 0.8)))
    n_blocks = len(blocks)
    # No threshold: enable parallel whenever ProcessPoolExecutor is available
    # Removed n_blocks >= max_workers condition to maximize parallelism on multi-core systems
    # Dynamic scheduling handles imbalanced workload distribution automatically
    use_parallel = (shared_setup is not None and max_workers > 1)

    logger.info(
        'probe_scales_multidim_stage2: %d blocks, cpu_count=%d, max_workers=%d, parallel=%s',
        n_blocks, _cpu, max_workers, use_parallel
    )

    if not use_parallel:
        # --- Sequential path (original loop) ---
        block_scales = np.zeros(n, dtype=np.float64)
        t0 = _time.monotonic()
        early_stop = ''
        block_stop_examples: List[str] = []

        for blk_idx, block in enumerate(blocks):
            logger.debug(
                'probe_scales_multidim_stage2: starting block %d/%d name=%s '
                'n_directions=%d param_indices=%s',
                blk_idx + 1,
                len(blocks),
                block.name,
                len(block.directions),
                list(block.param_indices),
            )
            elapsed = _time.monotonic() - t0
            if elapsed > max_wall_seconds:
                early_stop = f'wall_time_exceeded ({elapsed:.1f}s > {max_wall_seconds}s)'
                break
            if n_evals + 1 > max_evals:
                early_stop = f'max_evals_reached ({n_evals} + 1 > {max_evals})'
                break

            locked_idx = list(block.param_indices)

            for direction in block.directions:
                logger.debug(
                    'probe_scales_multidim_stage2: block=%s starting direction=%s',
                    block.name,
                    direction.name,
                )
                elapsed = _time.monotonic() - t0
                if elapsed > max_wall_seconds:
                    early_stop = f'wall_time_exceeded ({elapsed:.1f}s > {max_wall_seconds}s)'
                    break
                if n_evals + 1 > max_evals:
                    early_stop = f'max_evals_reached ({n_evals} + 1 > {max_evals})'
                    break

                active = np.abs(direction.full_vector) > 0.0
                if not np.any(active):
                    continue
                alpha_base_candidates = base_h[active] / np.maximum(np.abs(direction.full_vector[active]), 1e-15)
                alpha_base = float(np.min(alpha_base_candidates))
                if not np.isfinite(alpha_base) or alpha_base <= 0.0:
                    alpha_base = _DEFAULT_PROBE_STEP_FALLBACK
                alpha_base = float(np.clip(alpha_base, min_scale, max_scale))

                safe_alpha = alpha_base
                reason = f'alpha_step_limit_reached ({max_alpha_steps})'
                alpha = alpha_base

                for _ in range(max_alpha_steps):
                    elapsed = _time.monotonic() - t0
                    if elapsed > max_wall_seconds:
                        early_stop = f'wall_time_exceeded ({elapsed:.1f}s > {max_wall_seconds}s)'
                        reason = early_stop
                        break
                    if n_evals + 1 > max_evals:
                        early_stop = f'max_evals_reached ({n_evals} + 1 > {max_evals})'
                        reason = early_stop
                        break

                    per_param_step = np.abs(alpha * direction.full_vector)
                    if float(np.max(per_param_step)) > max_scale:
                        reason = f'max_scale_cap_reached ({float(np.max(per_param_step)):.3g} > {max_scale:.3g})'
                        break

                    x_probe = alpha * direction.full_vector
                    comp = evaluate_probe_step_with_compensation(
                        x_probe,
                        ctx,
                        locked_param_indices=locked_idx,
                        enable_compensation=enable_compensation,
                        max_compensation_iters=max_compensation_iters,
                    )
                    n_evals += 1

                    # Log compensated result BEFORE validity check (Fix 2: capture even on invalid)
                    logger.debug(
                        'probe_scales_multidim_stage2: block=%s dir=%s BEFORE_VALIDITY_CHECK '
                        'alpha=%.6f is_valid=%s failure_reason=%s',
                        block.name,
                        direction.name,
                        alpha,
                        comp.is_valid,
                        comp.failure_reason or 'N/A',
                    )
                    
                    if not comp.is_valid:
                        reason = f'invalid_geometry_or_eval ({comp.failure_reason or "unknown"})'
                        break

                    ray_now = float(comp.ray_rmse)
                    
                    # Log compensated metric BEFORE finiteness check (Fix 2)
                    logger.debug(
                        'probe_scales_multidim_stage2: block=%s dir=%s BEFORE_FINITE_CHECK '
                        'alpha=%.6f ray_now=%.6f is_finite=%s',
                        block.name,
                        direction.name,
                        alpha,
                        ray_now,
                        np.isfinite(ray_now),
                    )
                    
                    if not np.isfinite(ray_now):
                        reason = 'non_finite_compensated_ray_rmse'
                        break

                    # Log compensated metric BEFORE threshold stop check (Fix 2: capture actual value even if stop triggers)
                    logger.debug(
                        'probe_scales_multidim_stage2: block=%s dir=%s BEFORE_THRESHOLD_CHECK '
                        'alpha=%.6f ray_now=%.6f ray_stop_threshold=%.6f check_result=%s',
                        block.name,
                        direction.name,
                        alpha,
                        ray_now,
                        ray_stop_threshold,
                        'WILL_STOP' if ray_now >= ray_stop_threshold else 'continue',
                    )
                    
                    if ray_now >= ray_stop_threshold:
                        reason = (
                            'compensated_ray_rmse_stop '
                            f'({ray_now:.6g} >= {ray_stop_threshold:.6g})'
                        )
                        break

                    safe_alpha = float(np.clip(alpha, min_scale, max_scale))
                    alpha *= alpha_growth
                    logger.debug(
                        'probe_scales_multidim_stage2: block=%s dir=%s '
                        'alpha_step safe_alpha=%.6g next_alpha=%.6g '
                        'ray_now=%.6g n_evals=%d',
                        block.name,
                        direction.name,
                        safe_alpha,
                        alpha,
                        ray_now,
                        n_evals,
                    )

                contrib = np.abs(direction.full_vector) * safe_alpha
                block_scales = np.maximum(block_scales, contrib)

                if reason.startswith('compensated_ray_rmse_stop') or reason.startswith('invalid_geometry_or_eval'):
                    block_stop_examples.append(f'{block.name}/{direction.name}:{reason}')

                logger.debug(
                    'probe_scales_multidim_stage2: block=%s dir=%s RESULT '
                    'safe_alpha=%.6g reason=%s contrib_max=%.6g',
                    block.name,
                    direction.name,
                    safe_alpha,
                    reason,
                    float(np.max(contrib)) if contrib.size else 0.0,
                )

                if early_stop:
                    break

            logger.debug(
                'probe_scales_multidim_stage2: block %d/%d name=%s completed '
                'n_evals_so_far=%d elapsed=%.1fs block_scale_max=%.6g',
                blk_idx + 1,
                len(blocks),
                block.name,
                n_evals,
                _time.monotonic() - t0,
                float(np.max(block_scales)) if block_scales.size else 0.0,
            )
            if early_stop:
                logger.warning(
                    'probe_scales_multidim_stage2: early stop - %s (block %d/%d)',
                    early_stop,
                    blk_idx,
                    len(blocks),
                )
                break

        elapsed_total = _time.monotonic() - t0
        summary = early_stop
        if not summary and block_stop_examples:
            summary = 'stage2_stop_reasons: ' + ' | '.join(block_stop_examples[:8])

        logger.info(
            'probe_scales_multidim_stage2 complete: blocks=%d evals=%d %.1fs '
            'ray_ref=%.6g ray_stop=%.6g block_scale_range=[%.2e, %.2e] stopped_by=%s',
            len(blocks),
            n_evals,
            elapsed_total,
            ray_rmse_ref,
            ray_stop_threshold,
            float(np.min(block_scales)) if block_scales.size else 0.0,
            float(np.max(block_scales)) if block_scales.size else 0.0,
            summary or 'normal',
        )

        return BlockProbeResult(
            block_scales=block_scales,
            n_evals=n_evals,
            wall_seconds=elapsed_total,
            early_stop_reason=summary,
        )

    else:
        # --- Parallel path with BrokenProcessPool retry (Fix 3) ---
        block_scales = np.zeros(n, dtype=np.float64)
        t0 = _time.monotonic()
        n_evals_total = n_evals  # Already counted ref_result

        # Retry loop for robustness against worker crashes
        max_retry_attempts = 3
        retry_backoffs = [1.0, 2.0, 4.0]  # seconds
        remaining_blocks = set(range(len(blocks)))  # Track which blocks haven't completed

        for attempt in range(max_retry_attempts):
            try:
                with ProcessPoolExecutor(
                    max_workers=max_workers,
                    mp_context=mp.get_context('spawn'),
                    initializer=_init_probing_worker
                ) as executor:
                    futures = {
                        executor.submit(
                            _probe_stage2_single_block,
                            shared_setup,
                            blk_idx,
                            block,
                            ray_rmse_ref,
                            ray_stop_threshold,
                            max_evals,
                            max_wall_seconds,
                            min_scale,
                            max_scale,
                            max_alpha_steps,
                            alpha_growth,
                            enable_compensation,
                            max_compensation_iters,
                            base_h,
                        ): blk_idx
                        for blk_idx in remaining_blocks
                        for block in [blocks[blk_idx]]
                    }

                    for future in as_completed(futures):
                        blk_idx = futures[future]
                        result = future.result()
                        # Aggregate block_scales using np.maximum (NOT simple assignment)
                        block_scales = np.maximum(block_scales, result['block_scales'])
                        n_evals_total += result['n_evals']
                        remaining_blocks.discard(blk_idx)

                # Success — all blocks completed
                break

            except BrokenProcessPool as bpp:
                error_msg = f'BrokenProcessPool: {bpp!r}'
                if attempt < max_retry_attempts - 1:
                    backoff_sec = retry_backoffs[attempt]
                    logger.warning(
                        'probe_scales_multidim_stage2: BrokenProcessPool on attempt %d/%d: %s. '
                        'Retrying in %.1fs (blocks_remaining=%d)',
                        attempt + 1, max_retry_attempts, error_msg,
                        backoff_sec, len(remaining_blocks),
                    )
                    _time.sleep(backoff_sec)
                else:
                    # Max retries exhausted — fall back to sequential for remaining blocks
                    logger.error(
                        'probe_scales_multidim_stage2: BrokenProcessPool persists after %d attempts. '
                        'Falling back to sequential execution for %d remaining blocks.',
                        max_retry_attempts, len(remaining_blocks),
                    )
                    # Sequential fallback for remaining blocks
                    for blk_idx in list(remaining_blocks):
                        block = blocks[blk_idx]
                        try:
                            result = _probe_stage2_single_block(
                                shared_setup,
                                blk_idx,
                                block,
                                ray_rmse_ref,
                                ray_stop_threshold,
                                max_evals,
                                max_wall_seconds,
                                min_scale,
                                max_scale,
                                max_alpha_steps,
                                alpha_growth,
                                enable_compensation,
                                max_compensation_iters,
                                base_h,
                            )
                            block_scales = np.maximum(block_scales, result['block_scales'])
                            n_evals_total += result['n_evals']
                            remaining_blocks.discard(blk_idx)
                        except Exception as e:
                            logger.error(
                                'probe_scales_multidim_stage2: Sequential fallback failed for block %d: %s',
                                blk_idx, f'{type(e).__name__}: {e!r}',
                                exc_info=True,
                            )
                    break

        elapsed_total = _time.monotonic() - t0

        logger.info(
            'probe_scales_multidim_stage2 complete (parallel): blocks=%d evals=%d %.1fs '
            'ray_ref=%.6g ray_stop=%.6g block_scale_range=[%.2e, %.2e]',
            len(blocks),
            n_evals_total,
            elapsed_total,
            ray_rmse_ref,
            ray_stop_threshold,
            float(np.min(block_scales)) if block_scales.size else 0.0,
            float(np.max(block_scales)) if block_scales.size else 0.0,
        )

        return BlockProbeResult(
            block_scales=block_scales,
            n_evals=n_evals_total,
            wall_seconds=elapsed_total,
            early_stop_reason='',
        )


# ===========================================================================
#  Task 4: CMA-ES Global Search Driver
# ===========================================================================

# ---------------------------------------------------------------------------
#  Data structures
# ---------------------------------------------------------------------------

@dataclass
class GenerationLog:
    """Per-generation diagnostic snapshot.

    Core fields (always populated):
        gen, best_objective, median_objective, worst_objective,
        feasible_fraction, sigma, cumulative_evals, cumulative_wall_seconds

    Extended fields (populated by Task 6 generation-detail capture):
        best_ray_rmse, best_len_rmse — RMSE metrics for the generation-best candidate.
        best_real_params — real (absolute) parameter values for the generation-best.
        pop_real_min, pop_real_max — per-parameter empirical min/max of real values
            across the sampled population for this generation.
    """
    gen: int
    best_objective: float
    median_objective: float
    worst_objective: float
    feasible_fraction: float   # fraction of candidates with success=True
    sigma: float               # CMA-ES step-size
    cumulative_evals: int
    cumulative_wall_seconds: float

    # --- Extended generation-detail fields (Task 6) ---
    best_ray_rmse: float = field(default=float('nan'))
    best_len_rmse: float = field(default=float('nan'))
    best_real_params: Optional[np.ndarray] = field(default=None)
    pop_real_min: Optional[np.ndarray] = field(default=None)
    pop_real_max: Optional[np.ndarray] = field(default=None)

@dataclass
class CMARunResult:
    """Outcome of a single CMA-ES run."""
    run_id: int
    best_x_norm: np.ndarray          # best solution in normalised space
    best_x_delta: np.ndarray         # best solution as physical delta
    best_objective: float
    best_diagnostics: Dict[str, Any] # full evaluate_candidate result dict
    generation_log: List[GenerationLog]
    n_generations: int
    n_evals: int
    wall_seconds: float
    stop_reason: str


@dataclass
class GlobalSearchResult:
    """Outcome of the full global search (possibly multi-run)."""
    runs: List[CMARunResult]
    best_x_norm: np.ndarray
    best_x_delta: np.ndarray
    best_objective: float
    best_diagnostics: Dict[str, Any]
    ref_objective: float
    probe_result: ProbeResult
    candidates_deduped: List[Dict[str, Any]]  # deduplicated top candidates
    total_evals: int                           # across all runs + probing
    total_wall_seconds: float
    budget_status: Dict[str, Any]              # runtime/eval budget diagnostics


# ---------------------------------------------------------------------------
#  Phase-1 parallel architecture: inter-run parallelism only
#
#  Design constraints (locked by Task 1, enforced across Tasks 2-7):
#    1. Parallelism is INTER-RUN ONLY — each CMA-ES restart is an
#       independent unit of work dispatched to a separate process.
#    2. NO within-generation candidate parallelism inside _run_cma_single().
#       The es.ask() -> evaluate -> es.tell() loop remains sequential.
#    3. NO pickling of native/C++ objects (EvaluationContext, cams_cpp,
#       RefractiveBAOptimizer).  Workers reconstruct these locally.
#    4. NO nested parallelism (outer-run pool + inner candidate pool).
#    5. Sequential path is the default and source-of-truth fallback.
#       enable_parallel=False, n_runs==1, or max_workers<=1 all bypass
#       pool creation and use the direct sequential loop.
# ---------------------------------------------------------------------------


@dataclass
class ParallelConfig:
    """Phase-1 inter-run parallelism configuration.

    Controls whether independent CMA-ES restarts are dispatched to
    separate worker processes.  All defaults preserve current sequential
    behavior — parallel execution is opt-in.

    Architecture constraints (enforced, not configurable):
        - Parallelism is inter-run only; each worker runs one full
          ``_run_cma_single()`` to completion.
        - Within-generation candidate evaluation remains sequential.
        - Workers reconstruct native/C++ state locally; no pickling
          of ``EvaluationContext``, ``cams_cpp``, or optimizer objects.
    """

    enable_parallel: bool = False
    """Enable inter-run process parallelism.  When False (default),
    runs execute sequentially in the parent process."""

    max_workers: int = 1
    """Maximum number of worker processes for parallel runs.
    Effective only when ``enable_parallel=True`` and ``n_runs > 1``.
    Values <= 1 force sequential execution regardless of ``enable_parallel``."""

    worker_timeout_seconds: float = 7200.0
    """Per-worker wall-time timeout (seconds).  Workers exceeding this
    limit are terminated.  Only applies when parallel execution is active."""


# ---------------------------------------------------------------------------
#  Public multidimensional probing wrapper (orchestrates Stage 1 + Stage 2)
# ---------------------------------------------------------------------------

def probe_scales_multidim(
    ctx: EvaluationContext,
    *,
    probe_steps: Optional[Dict[str, float]] = None,
    max_evals: int = 500,
    max_wall_seconds: float = float('inf'),
    min_scale: float = 1e-8,
    max_scale: float = 100.0,
    ray_rmse_stop_factor: float = 1.1,
    enable_compensation: bool = True,
    max_compensation_iters: int = 3,
    alpha_growth: float = 2.0,
    max_alpha_steps: int = 12,
    enable_stage2: bool = True,
    stage2_max_evals: Optional[int] = None,
) -> ProbeResult:
    """Public wrapper orchestrating Stage-1 and optional Stage-2 probing.

    This function performs multidimensional probing to estimate per-parameter
    scales suitable for CMA-ES normalization. It first runs Stage-1 (1-D alpha
    expansion with compensation), then optionally runs Stage-2 (block directional
    probing for coupled parameters).

    Parameters
    ----------
    ctx : EvaluationContext
        Evaluation context with optimizer, dataset, and reference state.
    probe_steps : dict or None
        Per-parameter-type initial step sizes. If None, uses defaults.
    max_evals : int
        Total evaluation budget across both stages (default 500).
    max_wall_seconds : float
        Total wall-clock time budget across both stages (default: no limit).
    min_scale : float
        Minimum allowed scale (default 1e-8).
    max_scale : float
        Maximum allowed scale (default 100.0).
    ray_rmse_stop_factor : float
        Probing stops when compensated_ray_rmse >= factor * ray_rmse_ref.
    enable_compensation : bool
        Enable limited GN/LM compensation during probing (default True).
    max_compensation_iters : int
        Max compensation iterations per probe step (default 3).
    alpha_growth : float
        Alpha expansion growth factor (default 2.0, must be > 1.0).
    max_alpha_steps : int
        Max alpha steps per parameter (default 12, must be > 0).
    enable_stage2 : bool
        Enable Stage-2 block directional probing (default True).
        If False, only Stage-1 is run.
    stage2_max_evals : int or None
        Evaluation budget for Stage-2 only. If None, uses remaining budget
        from max_evals after Stage-1.

    Returns
    -------
    ProbeResult
        Aggregated result containing final scales, sensitivities, and diagnostics
        from both stages.

    Notes
    -----
    - Stage-1 uses the full max_evals budget initially.
    - Stage-2 (if enabled) uses any remaining budget or stage2_max_evals.
    - Scales from both stages are combined; Stage-2 block scales contribute
      additively to the final scales.
    - All defaults preserve backward compatibility with existing 1-D probing.
    """
    import time as _time

    t_start = _time.monotonic()
    max_wall_seconds_f = float(max(max_wall_seconds, 1.0))

    # Stage 1: 1-D alpha expansion with compensation
    stage1_result = probe_scales_multidim_stage1(
        ctx,
        probe_steps=probe_steps,
        max_evals=max_evals,
        max_wall_seconds=max_wall_seconds_f,
        min_scale=min_scale,
        max_scale=max_scale,
        ray_rmse_stop_factor=ray_rmse_stop_factor,
        enable_compensation=enable_compensation,
        max_compensation_iters=max_compensation_iters,
        alpha_growth=alpha_growth,
        max_alpha_steps=max_alpha_steps,
    )

    elapsed_s1 = _time.monotonic() - t_start
    remaining_evals = max_evals - stage1_result.n_evals
    remaining_wall_seconds = max_wall_seconds_f - elapsed_s1

    # Stage 2: optional block directional probing
    if enable_stage2 and remaining_evals > 10 and remaining_wall_seconds > 5.0:
        s2_max_evals = stage2_max_evals if stage2_max_evals is not None else remaining_evals
        s2_max_wall_seconds = remaining_wall_seconds

        try:
            stage2_result = probe_scales_multidim_stage2_blocks(
                ctx,
                layout=stage1_result.param_layout,
                probe_steps=probe_steps,
                max_evals=s2_max_evals,
                max_wall_seconds=s2_max_wall_seconds,
                min_scale=min_scale,
                max_scale=max_scale,
                ray_rmse_stop_factor=ray_rmse_stop_factor,
                enable_compensation=enable_compensation,
                max_compensation_iters=max_compensation_iters,
                alpha_growth=alpha_growth,
                max_alpha_steps=max_alpha_steps,
            )
            # Aggregate Stage-2 results into Stage-1 result
            stage1_result.scales = stage1_result.scales + (stage2_result.scales * 0.5)
            stage1_result.block_scales = stage2_result.scales
            stage1_result.block_probe_summary = stage2_result.summary
            stage1_result.n_evals += stage2_result.n_evals
            stage1_result.wall_seconds = _time.monotonic() - t_start
        except Exception as e:
            logger.warning(
                'Stage-2 block probing failed, keeping Stage-1 scales: %s',
                str(e),
            )

    logger.info(
        'probe_scales_multidim complete: stage1_evals=%d, stage2_evals=%d, '
        'total_evals=%d, total_wall_seconds=%.1f, scale_range=[%.2e, %.2e]',
        stage1_result.n_evals,
        max_evals - stage1_result.n_evals if enable_stage2 else 0,
        stage1_result.n_evals,
        stage1_result.wall_seconds,
        float(np.min(stage1_result.scales)),
        float(np.max(stage1_result.scales)),
    )

    return stage1_result


@dataclass
class ProbingConfig:
    """Configuration for probing mode selection and behavior.

    Controls which probing algorithm is used to estimate per-parameter
    scales for CMA-ES normalization.  All defaults preserve the current
    1-D curvature-based probing behavior (``probing_mode='1d'``).

    Attributes
    ----------
    probing_mode : str
        Probing algorithm to use:
        - ``'1d'`` (default) — existing 1-D central-difference curvature
          probing via :func:`probe_scales`.
        - ``'multidim'`` — multidimensional probing with block directional
          expansion and Ray-RMSE-based stop criteria (requires Tasks 2-5).
    shrink_factor : float
        Conservative multiplier applied to probed basin-width scales
        before feeding them into CMA normalization.  Only effective when
        ``probing_mode='multidim'``.  A value of 1.0 means no shrinkage;
        smaller values produce more conservative (tighter) search bounds.
    enable_block_probing : bool
        Whether to run Stage-2 block directional probing (coupled
        parameter directions) when ``probing_mode='multidim'``.
        Disabling this limits multidimensional probing to Stage-1
        (1-D alpha expansion with compensation) only.
    enable_compensation : bool
        Whether to run limited compensation optimization during each
        probing step.  When True, non-probed parameters are allowed to
        adjust slightly (2-3 GN/LM iterations) while the probed parameter
        is locked.  Only effective when ``probing_mode='multidim'``.
    max_compensation_iters : int
        Maximum number of Gauss-Newton / LM iterations for compensation
        optimization during probing.  Only effective when
        ``enable_compensation=True``.
    ray_rmse_stop_factor : float
        Probing stops when ``compensated_ray_rmse >= factor * ray_rmse_ref``.
        Only effective when ``probing_mode='multidim'``.
    """

    probing_mode: str = '1d'
    """Probing algorithm: '1d' (default) or 'multidim'."""

    shrink_factor: float = 0.5
    """Conservative multiplier for multidim basin-width scales (0 < s <= 1)."""

    enable_block_probing: bool = True
    """Enable Stage-2 block directional probing in multidim mode."""

    enable_compensation: bool = True
    """Enable limited compensation optimization during multidim probing."""

    max_compensation_iters: int = 3
    """Max GN/LM iterations for compensation (2-3 recommended)."""

    ray_rmse_stop_factor: float = 1.1
    """Stop probing when ray_rmse >= factor * ray_rmse_ref."""


@dataclass
class GenerationDetailConfig:
    """Configuration for per-run generation-detail CSV output.

    Controls whether detailed per-generation metrics (best ray/len RMSE,
    sigma, best real parameter values, per-parameter real min/max across
    the sampled population) are written to per-run CSV files.

    All defaults disable detail output, preserving current behavior where
    only the summary generation CSV is produced.

    Attributes
    ----------
    enable : bool
        Whether to write per-run generation-detail CSVs.  Default False
        preserves current behavior (summary CSV only).
    output_dir : str or Path or None
        Directory where per-run detail CSV files are written.  When None
        (default), files are placed alongside the summary diagnostics
        output directory.  Each run produces a separate file named
        ``{prefix}_run{run_id}_detail.csv``.
    prefix : str
        Filename prefix for detail CSV files.
    """

    enable: bool = False
    """Enable per-run generation-detail CSV output."""

    output_dir: Optional[str | Path] = None
    """Directory for per-run detail CSVs.  None = use diagnostics output_dir."""

    prefix: str = 'full_global'
    """Filename prefix for detail CSV files."""

@dataclass

class BudgetConfig:

    """Explicit runtime and evaluation budget configuration."""

    max_total_evals: int = 50000

    max_total_wall_seconds: float = 86400.0

    max_probing_evals: int = 500

    max_probing_wall_seconds: float = float('inf')

    max_per_run_evals: int = 2500

    max_per_run_wall_seconds: float = 3600.0

    enable_probing: bool = True



    def is_reduced_probing(self) -> bool:

        """Return True if probing is disabled or heavily constrained."""

        return (not self.enable_probing) or (self.max_probing_evals < 100)





@dataclass

class BudgetStatus:

    """Per-phase budget exhaustion tracking and diagnostics."""

    probing_evals_used: int = 0

    probing_wall_seconds: float = 0.0

    probing_stopped_by: str = ''

    total_evals_used: int = 0

    total_wall_seconds: float = 0.0

    total_stopped_by: str = ''

    runs_completed: int = 0

    cumulative_by_run: List[Dict[str, Any]] = field(default_factory=list)



    def to_dict(self) -> Dict[str, Any]:

        return {

            'probing_evals_used': self.probing_evals_used,

            'probing_wall_seconds': self.probing_wall_seconds,

            'probing_stopped_by': self.probing_stopped_by,

            'total_evals_used': self.total_evals_used,

            'total_wall_seconds': self.total_wall_seconds,

            'total_stopped_by': self.total_stopped_by,

            'runs_completed': self.runs_completed,

            'cumulative_by_run': self.cumulative_by_run,

        }





def _cma_objective(
    x_norm: np.ndarray,
    ctx: EvaluationContext,
    scales: np.ndarray,
) -> float:
    """Convert normalised vector to physical delta and return scalar cost.

    Parameters
    ----------
    x_norm : ndarray, shape (n_params,)
        Candidate in normalised CMA-ES space.
    ctx : EvaluationContext
        Evaluation context with optimizer and layout.
    scales : ndarray, shape (n_params,)
        Per-parameter scales from :func:`probe_scales`.

    Returns
    -------
    float
        Scalar objective (sentinel value on failure).
    """
    x_delta = scales * np.asarray(x_norm, dtype=np.float64)
    result = evaluate_candidate(x_delta, ctx)
    if result['success']:
        return result['objective']
    return _SENTINEL_OBJECTIVE


def _cma_objective_full(
    x_norm: np.ndarray,
    ctx: EvaluationContext,
    scales: np.ndarray,
) -> Tuple[float, Dict[str, Any]]:
    """Like :func:`_cma_objective` but also returns the full diagnostics dict."""
    x_delta = scales * np.asarray(x_norm, dtype=np.float64)
    result = evaluate_candidate(x_delta, ctx)
    obj = result['objective'] if result['success'] else _SENTINEL_OBJECTIVE
    return obj, result



def _extract_reference_values(ctx: EvaluationContext) -> np.ndarray:
    """Build a vector of reference (absolute) values for each layout parameter.

    For additive parameters (``cam_t``, ``plane_d``), the reference value is the
    stored initial value in the optimizer.  For angular perturbation parameters
    (``cam_r``, ``plane_a``, ``plane_b``) the reference is **0** because the
    delta parameterisation treats the initial orientation as the origin.

    The returned vector has the same length as ``ctx.n_params``.  Adding
    ``x_delta`` (physical delta) to this vector yields "real" parameter values.

    Notes
    -----
    - ``cam_r`` uses Rodrigues-style left-multiplication, so the delta is not
      strictly additive to the stored rvec.  For search-range diagnostics the
      convention ref=0 (initial rotation = identity perturbation) is adequate.
    - ``plane_a`` / ``plane_b`` similarly represent tangent-space angles from the
      initial normal; ref=0 is the natural convention.
    """
    opt = ctx.optimizer
    n = ctx.n_params
    ref_vec = np.zeros(n, dtype=np.float64)

    for idx, (ptype, entity_id, sub_idx) in enumerate(ctx.layout):
        if ptype == 'plane_d':
            # d0 = initial signed distance from anchor along the plane normal
            d0 = opt._plane_d0.get(entity_id, 0.0)
            ref_vec[idx] = float(d0)
        elif ptype in ('plane_a', 'plane_b'):
            # tangent-space angle perturbation — ref is 0
            ref_vec[idx] = 0.0
        elif ptype == 'cam_t':
            # translation component: initial_cam_params[cid][3 + sub_idx]
            cparams = opt.initial_cam_params.get(entity_id)
            if cparams is not None and len(cparams) > 3 + sub_idx:
                ref_vec[idx] = float(cparams[3 + sub_idx])
        elif ptype == 'cam_r':
            # rotation perturbation — ref is 0 (identity rotation delta)
            ref_vec[idx] = 0.0
        elif ptype == 'cam_f':
            cparams = opt.initial_cam_params.get(entity_id)
            if cparams is not None and len(cparams) > 6:
                ref_vec[idx] = float(cparams[6])
        elif ptype == 'cam_k1':
            cparams = opt.initial_cam_params.get(entity_id)
            if cparams is not None and len(cparams) > 9:
                ref_vec[idx] = float(cparams[9])
        elif ptype == 'cam_k2':
            cparams = opt.initial_cam_params.get(entity_id)
            if cparams is not None and len(cparams) > 10:
                ref_vec[idx] = float(cparams[10])
        elif ptype == 'win_t':
            media = opt.initial_media.get(entity_id)
            if media is not None and 'thickness' in media:
                ref_vec[idx] = float(media['thickness'])
        # else: unknown ptype, ref stays 0.0

    return ref_vec


# ---------------------------------------------------------------------------
#  Single CMA-ES run
#
#  ARCHITECTURE CONSTRAINT (Phase 1): this function executes one complete
#  CMA-ES run SEQUENTIALLY.  The es.ask() -> for-loop evaluate -> es.tell()
#  cycle must NOT be parallelised internally.  Inter-run parallelism is
#  handled at the outer level in run_global_search(), not here.
# ---------------------------------------------------------------------------
def _run_cma_single(
    ctx: EvaluationContext,
    scales: np.ndarray,
    run_id: int = 0,
    *,
    sigma0: float = 2.0,
    popsize: Optional[int] = None,
    max_evals: int = 2500,
    max_generations: Optional[int] = None,
    stagnation_gens: int = 25,
    sigma_stop: float = 1e-3,
    seed: Optional[int] = None,
) -> CMARunResult:
    """Execute a single CMA-ES run in normalised space.

    Parameters
    ----------
    ctx : EvaluationContext
        Shared evaluation context.
    scales : ndarray, shape (n_params,)
        Per-parameter scales from probing.
    run_id : int
        Identifier for this run (used in logging).
    sigma0 : float
        Initial step-size in normalised space.
    popsize : int or None
        Population size.  *None* uses CMA-ES default ``4 + int(3*ln(n))``.
    max_evals : int
        Maximum number of evaluations for this run.
    max_generations : int or None
        Maximum number of generations.  *None* means no generation cap
        (only eval budget applies).
    stagnation_gens : int
        Stop if no improvement in this many consecutive generations.
    sigma_stop : float
        Stop if CMA-ES sigma falls below this threshold.
    seed : int or None
        Random seed for CMA-ES.

    Returns
    -------
    CMARunResult
    """
    import time as _time
    import cma  # type: ignore[import-untyped]

    n = ctx.n_params
    x0 = np.zeros(n, dtype=np.float64)

    # --- Compute effective population size based on eval budget ---
    # CMA-ES default: 4 + 3*ln(n)
    default_popsize = max(4, int(4 + 3 * np.log(n)))
    if popsize is None:
        effective_popsize = default_popsize
    else:
        effective_popsize = min(popsize, default_popsize)
    
    # Cap population size to remaining eval budget to avoid exceeding max_evals
    effective_popsize = min(effective_popsize, max(1, max_evals))

    # --- CMA-ES options ---
    opts: Dict[str, Any] = {
        'verbose': -9,           # suppress all console output
        'maxfevals': max_evals,
        'bounds': [[-8.0] * n, [8.0] * n],
        'CMA_active': True,
        'popsize': effective_popsize,
    }
    if seed is not None:
        opts['seed'] = seed

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    # --- tracking ---
    gen_log: List[GenerationLog] = []
    best_obj = _SENTINEL_OBJECTIVE
    best_x_norm: Optional[np.ndarray] = None
    best_diag: Dict[str, Any] = {}
    gens_since_improvement = 0
    total_evals = 0
    gen_idx = 0
    stop_reason = ''
    t0 = _time.monotonic()

    # --- extract reference values once for real-value generation detail ---
    try:
        ref_values = _extract_reference_values(ctx)
    except Exception:
        ref_values = np.zeros(n, dtype=np.float64)

    while not es.stop():
        # --- generation limit ---
        if max_generations is not None and gen_idx >= max_generations:
            stop_reason = f'max_generations_reached ({max_generations})'
            break

        # --- hard eval budget check: stop if we've hit the limit ---
        if total_evals >= max_evals:
            stop_reason = f'max_evals_exhausted ({total_evals} >= {max_evals})'
            break

        # --- ask / evaluate / tell (sequential — see Phase-1 constraint) ---
        X = es.ask()
        fitnesses: List[float] = []
        diags: List[Dict[str, Any]] = []
        n_feasible = 0

        for x_norm_i in X:
            # --- Hard stop: do not evaluate if we've hit budget ---
            if total_evals >= max_evals:
                stop_reason = f'max_evals_exhausted_mid_gen ({total_evals} >= {max_evals})'
                break
            obj_i, diag_i = _cma_objective_full(x_norm_i, ctx, scales)
            fitnesses.append(obj_i)
            diags.append(diag_i)
            total_evals += 1
            if diag_i.get('success', False):
                n_feasible += 1

        # If we broke early from the population loop, salvage any best from partial batch
        if total_evals >= max_evals:
            if fitnesses:  # partial batch has evaluated candidates
                partial_best_idx = int(np.argmin(fitnesses))
                partial_best_obj = fitnesses[partial_best_idx]
                if partial_best_obj < best_obj:
                    best_obj = partial_best_obj
                    best_x_norm = np.array(X[partial_best_idx], dtype=np.float64)
                    best_diag = diags[partial_best_idx]
            break

        es.tell(X, fitnesses)

        # --- track best ---
        gen_best_idx = int(np.argmin(fitnesses))
        gen_best_obj = fitnesses[gen_best_idx]

        if gen_best_obj < best_obj:
            best_obj = gen_best_obj
            best_x_norm = np.array(X[gen_best_idx], dtype=np.float64)
            best_diag = diags[gen_best_idx]
            gens_since_improvement = 0
        else:
            gens_since_improvement += 1

        # --- generation log ---
        sorted_fits = sorted(fitnesses)
        pop_size_actual = len(fitnesses)
        median_obj = sorted_fits[pop_size_actual // 2]
        worst_obj = sorted_fits[-1]
        feasible_frac = n_feasible / max(pop_size_actual, 1)
        elapsed = _time.monotonic() - t0

        # --- extended generation-detail: real-value population stats ---
        gen_best_diag = diags[gen_best_idx] if 0 <= gen_best_idx < len(diags) else {}
        gen_best_ray_rmse = float(gen_best_diag.get('ray_rmse', float('nan')))
        gen_best_len_rmse = float(gen_best_diag.get('len_rmse', float('nan')))

        # Best candidate real params: ref + delta
        gen_best_x_norm = np.asarray(X[gen_best_idx], dtype=np.float64)
        gen_best_x_delta = scales * gen_best_x_norm
        gen_best_real = ref_values + gen_best_x_delta

        # Empirical real-value min/max across the sampled population
        pop_deltas = np.array(
            [scales * np.asarray(xi, dtype=np.float64) for xi in X],
            dtype=np.float64,
        )  # shape (pop_size, n_params)
        pop_real = ref_values[np.newaxis, :] + pop_deltas  # broadcast
        pop_real_min = np.min(pop_real, axis=0)
        pop_real_max = np.max(pop_real, axis=0)

        gen_log.append(GenerationLog(
            gen=gen_idx,
            best_objective=gen_best_obj,
            median_objective=median_obj,
            worst_objective=worst_obj,
            feasible_fraction=feasible_frac,
            sigma=float(es.sigma),
            cumulative_evals=total_evals,
            cumulative_wall_seconds=elapsed,
            best_ray_rmse=gen_best_ray_rmse,
            best_len_rmse=gen_best_len_rmse,
            best_real_params=gen_best_real,
            pop_real_min=pop_real_min,
            pop_real_max=pop_real_max,
        ))

        logger.info(
            '[Run %d][Gen %d] obj=%.4f ray=%.4f len=%.4f '
            'med=%.4f worst=%.4f sigma=%.4f evals=%d feas=%.0f%%',
            run_id + 1,
            gen_idx + 1,
            gen_best_obj,
            gen_best_ray_rmse,
            gen_best_len_rmse,
            median_obj,
            worst_obj,
            float(es.sigma),
            total_evals,
            feasible_frac * 100,
        )

        gen_idx += 1

        # --- custom termination ---
        if gens_since_improvement >= stagnation_gens:
            stop_reason = f'stagnation ({stagnation_gens} gens without improvement)'
            break

        if float(es.sigma) < sigma_stop:
            stop_reason = f'sigma_collapse (sigma={float(es.sigma):.2e} < {sigma_stop})'
            break

    # --- finalise ---
    if not stop_reason:
        # CMA-ES internal stop (maxfevals, etc.)
        cma_stops = es.stop()
        if cma_stops:
            stop_reason = 'cma_internal: ' + ', '.join(
                f'{k}={v}' for k, v in cma_stops.items()
            )
        else:
            stop_reason = 'unknown'

    wall_total = _time.monotonic() - t0

    if best_x_norm is None:
        # No valid evaluation found
        best_x_norm = x0.copy()
        best_diag = evaluate_candidate(None, ctx)
        best_obj = best_diag.get('objective', _SENTINEL_OBJECTIVE)

    best_x_delta = scales * best_x_norm

    logger.info(
        'CMA run %d finished: %d gens, %d evals, %.1fs, '
        'best_obj=%.4f, stop=%s',
        run_id, gen_idx, total_evals, wall_total,
        best_obj, stop_reason,
    )

    return CMARunResult(
        run_id=run_id,
        best_x_norm=best_x_norm,
        best_x_delta=best_x_delta,
        best_objective=best_obj,
        best_diagnostics=best_diag,
        generation_log=gen_log,
        n_generations=gen_idx,
        n_evals=total_evals,
        wall_seconds=wall_total,
        stop_reason=stop_reason,
    )


# ---------------------------------------------------------------------------
#  De-duplication helper
# ---------------------------------------------------------------------------

def _deduplicate_candidates(
    runs: List[CMARunResult],
    threshold: float = 0.1,
) -> List[Dict[str, Any]]:
    """Collect best-per-run candidates and de-duplicate by L2 distance.

    Parameters
    ----------
    runs : list[CMARunResult]
        Completed CMA-ES runs.
    threshold : float
        Minimum L2 distance in normalised space to consider candidates
        distinct.

    Returns
    -------
    list[dict]
        De-duplicated candidate dicts, sorted by objective (ascending).
        Each dict has keys: ``x_norm``, ``x_delta``, ``objective``,
        ``diagnostics``, ``run_id``.
    """
    # Sort runs by best objective
    sorted_runs = sorted(runs, key=lambda r: r.best_objective)

    candidates: List[Dict[str, Any]] = []
    for run in sorted_runs:
        x_n = run.best_x_norm
        # Check against already-accepted candidates
        is_dup = False
        for accepted in candidates:
            dist = float(np.linalg.norm(x_n - accepted['x_norm']))
            if dist < threshold:
                is_dup = True
                break
        if not is_dup:
            candidates.append({
                'x_norm': x_n,
                'x_delta': run.best_x_delta,
                'objective': run.best_objective,
                'diagnostics': run.best_diagnostics,
                'run_id': run.run_id,
            })

    return candidates


# ---------------------------------------------------------------------------
#  Public entry-point: multi-run global search
# ---------------------------------------------------------------------------

def run_global_search(
    camfile_dir: str | Path,
    obs_csv_path: str | Path,
    wand_length: float,
    *,
    n_runs: int = 5,
    max_evals_per_run: int = 2500,
    max_generations: Optional[int] = None,
    sigma0: float = 0.6,
    popsize: Optional[int] = None,
    stagnation_gens: int = 25,
    sigma_stop: float = 1e-3,
    dedup_threshold: float = 0.1,
    probe_result: Optional[ProbeResult] = None,
    probe_max_evals: int = 500,
    probe_max_wall: float = float('inf'),
    lambda_base_per_cam: float = 2.0,
    max_frames: int = 50000,
    dist_coeff_num: int = 0,
    seed_base: Optional[int] = None,
    max_total_evals: int = 50000,
    max_total_wall_seconds: float = 86400.0,
    budget_config: Optional[BudgetConfig] = None,
    enable_parallel: bool = False,
    max_workers: int = 1,
    worker_timeout: float = 7200.0,
    parallel_config: Optional[ParallelConfig] = None,
    probing_config: Optional[ProbingConfig] = None,
    generation_detail_config: Optional[GenerationDetailConfig] = None,
    ) -> GlobalSearchResult:
    """Run multi-start CMA-ES global search over camera/plane parameters.

    This is the top-level entry point for Task 4.  It:

    1. Loads the reference state and observations.
    2. Builds the evaluation context.
    3. Runs 1-D scale probing (or reuses a prior result).
    4. Executes *n_runs* independent CMA-ES runs (sequentially by default,
       or via inter-run process parallelism when enabled).
    5. De-duplicates the best-per-run candidates.
    6. Returns a :class:`GlobalSearchResult` with all diagnostics.

    **Phase-1 parallel architecture**: when ``enable_parallel=True`` and
    ``max_workers > 1`` and ``n_runs > 1``, independent CMA-ES restarts
    are dispatched to separate worker processes.  Each worker reconstructs
    its own evaluation context and native C++ cameras locally — no
    native/C++ objects are pickled.  Within-generation candidate evaluation
    inside each run remains sequential.  See :class:`ParallelConfig`.

    Parameters
    ----------
    camfile_dir : path-like
        Directory containing ``cam{id}.txt`` PINPLATE files.
    obs_csv_path : path-like
        Path to the wand-point CSV.
    wand_length : float
        Physical wand length in mm.
    n_runs : int
        Number of independent CMA-ES restarts.
    max_evals_per_run : int
        Per-run evaluation budget.
    max_generations : int or None
        Per-run generation cap.  *None* means no cap (budget only).
    sigma0 : float
        Initial CMA-ES step-size in normalised space.
    popsize : int or None
        CMA-ES population size.  *None* uses default.
    stagnation_gens : int
        Generations without improvement before stopping a run.
    sigma_stop : float
        Sigma threshold for early stopping.
    dedup_threshold : float
        L2 distance threshold for de-duplication in normalised space.
    probe_result : ProbeResult or None
        Pre-computed probe result.  If *None*, probing is run first.
    probe_max_evals : int
        Max evaluations for probing (if run).
    probe_max_wall : float
        Max wall-time for probing (if run).
    lambda_base_per_cam : float
        Per-camera weight for wand-length residuals.
    max_frames : int
        Maximum observation frames to use.
    dist_coeff_num : int
        Number of distortion coefficients (0, 1, or 2).
    seed_base : int or None
        Base seed; each run uses ``seed_base + run_id``.
    max_total_evals : int
        Global evaluation budget across all runs + probing.
    max_total_wall_seconds : float
        Global wall-time budget.
    budget_config : BudgetConfig or None
        Explicit budget configuration.  Overrides scalar budget params.
    enable_parallel : bool
        Enable inter-run process parallelism.  Default ``False``
        preserves current sequential behavior.
    max_workers : int
        Maximum worker processes when parallel is enabled.  Default ``1``
        forces sequential execution even if ``enable_parallel=True``.
    worker_timeout : float
        Per-worker timeout in seconds (parallel mode only).
    parallel_config : ParallelConfig or None
        Structured parallel configuration.  When provided, overrides
        ``enable_parallel``, ``max_workers``, and ``worker_timeout``.
    probing_config : ProbingConfig or None
        Structured probing configuration.  When provided, controls which
        probing algorithm is used and its behavioral parameters.  Default
        ``None`` preserves current 1-D curvature probing behavior.
    generation_detail_config : GenerationDetailConfig or None
        Configuration for per-run generation-detail CSV output.  Default
        ``None`` disables detail output, preserving current behavior.

    Returns
    -------
    GlobalSearchResult
    """
    import time as _time

    # --- Resolve parallel configuration ---
    # ParallelConfig takes precedence over individual scalar params.
    if parallel_config is not None:
        _par = parallel_config
    else:
        _par = ParallelConfig(
            enable_parallel=enable_parallel,
            max_workers=max_workers,
            worker_timeout_seconds=worker_timeout,
        )

    # Task 6: Control oversubscription and worker resource usage

    # Phase-1 gate: determine whether to use parallel dispatch.
    # Sequential path is used when ANY of these hold:
    #   - enable_parallel is False
    #   - max_workers <= 1
    #   - n_runs <= 1  (no benefit from parallelism)
    
    # Compute effective worker count with guardrails:
    # Default max_workers to min(n_runs, cpu_count()-1, 4) if caller did not set it explicitly.
    # This prevents oversubscription while bounding CPU usage to sane defaults.
    if _par.max_workers <= 1:
        # Caller explicitly requested serial or did not set (default=1)
        effective_workers = 1
    else:
        # Caller set explicit max_workers > 1; still cap at n_runs (no more workers than runs).
        effective_workers = min(_par.max_workers, n_runs)
    
    _use_parallel = (
        _par.enable_parallel
        and effective_workers > 1
        and n_runs > 1
    )

    if _use_parallel:
        logger.info(
            'Task 6: Phase-1 parallel mode with bounded workers '
            '(requested_max_workers=%d, n_runs=%d, effective_workers=%d, timeout=%.0fs)',
            _par.max_workers, n_runs, effective_workers, _par.worker_timeout_seconds,
        )
    else:
        if _par.enable_parallel and (effective_workers <= 1 or n_runs <= 1):
            logger.info(
                'Parallel requested but bypassed (requested_max_workers=%d, n_runs=%d, effective_workers=%d); '
                'using sequential path',
                _par.max_workers, n_runs, effective_workers,
            )
        elif not _par.enable_parallel:
            logger.debug(
                'Parallel disabled; using sequential path (n_runs=%d)',
                n_runs,
            )

    # Create or use default budget config

    if budget_config is None:

        budget_config = BudgetConfig(

            max_total_evals=max_total_evals,

            max_total_wall_seconds=max_total_wall_seconds,

            max_probing_evals=probe_max_evals,

            max_probing_wall_seconds=probe_max_wall,

            max_per_run_evals=max_evals_per_run,

        )


    # --- Resolve probing configuration ---
    # ProbingConfig defaults preserve current 1-D curvature probing.
    if probing_config is None:
        probing_config = ProbingConfig()
    logger.debug(
        'Probing config: mode=%s, shrink_factor=%.3f, block_probing=%s, '
        'compensation=%s (max_iters=%d), ray_rmse_stop_factor=%.3f',
        probing_config.probing_mode, probing_config.shrink_factor,
        probing_config.enable_block_probing, probing_config.enable_compensation,
        probing_config.max_compensation_iters, probing_config.ray_rmse_stop_factor,
    )

    # --- Resolve generation detail configuration ---
    # GenerationDetailConfig defaults disable per-run detail output.
    if generation_detail_config is None:
        generation_detail_config = GenerationDetailConfig()
    if generation_detail_config.enable:
        logger.info(
            'Generation detail output enabled: dir=%s, prefix=%s',
            generation_detail_config.output_dir, generation_detail_config.prefix,
        )

    # Track budget exhaustion across all phases

    budget_status = BudgetStatus()



    t_global = _time.monotonic()

    total_evals = 0


    # --- 1. Load reference state ---
    logger.info('=== Global Search: loading reference state ===')
    ref_state = load_reference_state(camfile_dir)

    # --- 2. Load observations ---
    logger.info('=== Global Search: loading observations ===')
    dataset = load_observations_csv(obs_csv_path, wand_length,
                                     dist_coeff_num=dist_coeff_num)

    # --- 3. Build evaluation context ---
    logger.info('=== Global Search: building evaluation context ===')
    ctx = build_evaluation_context(
        ref_state, dataset, wand_length,
        lambda_base_per_cam=lambda_base_per_cam,
        max_frames=max_frames,
    )

    # --- Build shared_setup for worker processes ---
    shared_setup = build_shared_setup(
        ref_state,
        dataset,
        np.ones(ctx.n_params, dtype=np.float64),
        wand_length=wand_length,
        lambda_base_per_cam=lambda_base_per_cam,
        max_frames=max_frames,
        dist_coeff_num=dist_coeff_num,
    )

    # --- 4. Probe scales (or reuse) ---

    t_probe_start = _time.monotonic()

    if probe_result is None:

        if budget_config.enable_probing:

            logger.info('=== Global Search: probing scales (mode=%s, budget: %d evals, %.0fs) ===',
                       probing_config.probing_mode,
                       budget_config.max_probing_evals, budget_config.max_probing_wall_seconds)

            if probing_config.probing_mode == 'multidim':
                probe_result = probe_scales_multidim_stage1(
                    ctx,
                    max_evals=budget_config.max_probing_evals,
                    max_wall_seconds=budget_config.max_probing_wall_seconds,
                    ray_rmse_stop_factor=probing_config.ray_rmse_stop_factor,
                    enable_compensation=probing_config.enable_compensation,
                    max_compensation_iters=probing_config.max_compensation_iters,
                    shared_setup=shared_setup,
                )
                if probing_config.enable_block_probing:
                    remaining_evals = max(0, budget_config.max_probing_evals - probe_result.n_evals)
                    remaining_wall = max(0.0, budget_config.max_probing_wall_seconds - probe_result.wall_seconds)
                    if remaining_evals <= 0 or remaining_wall <= 0.0:
                        probe_result.block_probe_summary = (
                            f'stage2_skipped_no_budget (remaining_evals={remaining_evals}, '
                            f'remaining_wall={remaining_wall:.3f}s)'
                        )
                        logger.info(
                            'Multidimensional probing Stage-2 skipped: %s',
                            probe_result.block_probe_summary,
                        )
                    else:
                        block_result = probe_scales_multidim_stage2_blocks(
                            ctx,
                            layout=probe_result.param_layout,
                            max_evals=remaining_evals,
                            max_wall_seconds=remaining_wall,
                            ray_rmse_stop_factor=probing_config.ray_rmse_stop_factor,
                            enable_compensation=probing_config.enable_compensation,
                            max_compensation_iters=probing_config.max_compensation_iters,
                            shared_setup=shared_setup,
                        )
                        probe_result.block_scales = block_result.block_scales
                        probe_result.block_probe_summary = block_result.early_stop_reason
                        probe_result.n_evals += block_result.n_evals
                        probe_result.wall_seconds += block_result.wall_seconds
                else:
                    probe_result.block_probe_summary = 'stage2_disabled_by_config'
            else:
                if probing_config.probing_mode != '1d':
                    logger.warning(
                        "Unknown probing_mode=%r; falling back to '1d'",
                        probing_config.probing_mode,
                    )
                probe_result = probe_scales(
                    ctx,
                    max_evals=budget_config.max_probing_evals,
                    max_wall_seconds=budget_config.max_probing_wall_seconds,
                )

        else:

            logger.info('=== Global Search: probing disabled, using default scales ===')

            # Create minimal probe result with uniform scales

            n = ctx.n_params

            probe_result = ProbeResult(

                scales=np.ones(n, dtype=np.float64) * 0.3,

                sensitivities=np.zeros(n, dtype=np.float64),

                ref_objective=0.0,

                labels=build_search_parameter_layout(ctx).labels(),

                n_evals=0,

                wall_seconds=0.0,

                early_stop_reason='probing_disabled',

                param_layout=build_search_parameter_layout(ctx),

            )

        total_evals += probe_result.n_evals

        budget_status.probing_evals_used = probe_result.n_evals

        budget_status.probing_wall_seconds = _time.monotonic() - t_probe_start

        budget_status.probing_stopped_by = probe_result.early_stop_reason

    else:

        logger.info('=== Global Search: reusing pre-computed probe result ===')


    # --- Scale merge and shrink-factor mapping (Task 5) ---
    raw_scales_1d = probe_result.scales.copy()
    raw_block_scales = probe_result.block_scales
    ref_obj = probe_result.ref_objective

    if probing_config.probing_mode == 'multidim':
        # Merge: effective_scale(i) = max(scale_1d(i), scale_block(i))
        if (raw_block_scales.size == raw_scales_1d.size
                and raw_block_scales.size > 0):
            effective_scales = np.maximum(raw_scales_1d, raw_block_scales)
            n_block_dominated = int(np.sum(raw_block_scales > raw_scales_1d))
            logger.info(
                'Scale merge: %d/%d params have block_scale > scale_1d '
                '(block-dominated)',
                n_block_dominated, effective_scales.size,
            )
        else:
            effective_scales = raw_scales_1d.copy()
            logger.info(
                'Scale merge: no block scales available (size=%d vs 1d=%d); '
                'using Stage-1 scales only',
                raw_block_scales.size, raw_scales_1d.size,
            )

        # Apply conservative shrink factor
        shrink = float(probing_config.shrink_factor)
        if shrink <= 0.0 or shrink > 1.0:
            logger.warning(
                'shrink_factor=%.4f out of (0,1]; clamping to 0.5', shrink,
            )
            shrink = 0.5
        scales = effective_scales * shrink

        # Post-probe sanity guardrail: if any scale is non-finite or
        # unreasonably large (> 1000), shrink all scales globally to
        # prevent CMA-ES from diverging.
        _SCALE_SANITY_CEIL = 1000.0
        if not np.all(np.isfinite(scales)) or np.any(scales > _SCALE_SANITY_CEIL):
            n_bad = int(np.sum(~np.isfinite(scales)) + np.sum(scales > _SCALE_SANITY_CEIL))
            logger.warning(
                'Post-probe sanity guardrail: %d/%d scales are non-finite or '
                '> %.0f; applying global clamp',
                n_bad, scales.size, _SCALE_SANITY_CEIL,
            )
            scales = np.where(
                np.isfinite(scales), np.minimum(scales, _SCALE_SANITY_CEIL), 0.3
            )

        # Ensure all scales are positive (CMA requires positive scales)
        _SCALE_FLOOR = 1e-8
        scales = np.maximum(scales, _SCALE_FLOOR)

        logger.info(
            'Multidim scale mapping: shrink_factor=%.3f, '
            'effective range [%.2e, %.2e] -> CMA range [%.2e, %.2e]',
            shrink,
            float(np.min(effective_scales)), float(np.max(effective_scales)),
            float(np.min(scales)), float(np.max(scales)),
        )
    else:
        # 1-D probing mode: use raw scales directly (no merge, no shrink)
        scales = raw_scales_1d

    # Build serializable shared setup for worker reconstruction (Task 2).
    # This payload intentionally excludes EvaluationContext / optimizer / C++
    # camera objects and uses the final (possibly merged+shrunk) scales.
    _shared_setup = build_shared_setup(
        ref_state,
        dataset,
        scales,
        wand_length=wand_length,
        lambda_base_per_cam=lambda_base_per_cam,
        max_frames=max_frames,
        dist_coeff_num=dist_coeff_num,
    )

    logger.info(
        'Probe: ref_obj=%.4f, %d params, CMA scales range [%.2e, %.2e], '
        'stopped_by: %s',
        ref_obj, len(scales), float(np.min(scales)), float(np.max(scales)),
        probe_result.early_stop_reason or 'normal',
    )
    logger.info(
        'Shared setup prepared for workers: key=%s, n_params=%d',
        _shared_setup.get('setup_key', ''),
        int(_shared_setup.get('n_params_expected', 0)),
    )

    logger.info('=== Probe scales (raw 1-D / Stage-1) ===')
    for label, scale in zip(probe_result.labels, raw_scales_1d):
        logger.info('  %-12s scale_1d=%.6g', label, float(scale))
    if raw_block_scales.size == raw_scales_1d.size and raw_block_scales.size > 0:
        logger.info('=== Probe block scales (Stage-2) ===')
        for label, scale in zip(probe_result.labels, raw_block_scales):
            logger.info('  %-12s block_scale=%.6g', label, float(scale))
        logger.info('Stage-2 block probing summary: %s', probe_result.block_probe_summary or 'normal')
    if probing_config.probing_mode == 'multidim':
        logger.info('=== Final CMA scales (after merge + shrink_factor=%.3f) ===', float(probing_config.shrink_factor))
        for label, s in zip(probe_result.labels, scales):
            logger.info('  %-12s cma_scale=%.6g', label, float(s))

    # --- 5. CMA-ES runs ---
    # Phase-1 architecture: inter-run parallelism only.
    # _run_cma_single() is NEVER parallelised internally.
    runs: List[CMARunResult] = []

    if _use_parallel:
        # ---- Parallel dispatch via ProcessPoolExecutor (Task 3/5) ----
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from concurrent.futures.process import BrokenProcessPool

        mp_ctx = mp.get_context('spawn')
        logger.info(
            '=== Global Search: executing %d CMA-ES runs on %d worker processes '
            '(spawn context, timeout=%.0fs) ===',
            n_runs, effective_workers, _par.worker_timeout_seconds,
        )

        # Pre-compute per-run params (seed, max_evals)
        run_params: List[Dict[str, Any]] = []
        for run_id in range(n_runs):
            remaining_evals = budget_config.max_total_evals - total_evals
            run_max_evals = min(budget_config.max_per_run_evals, remaining_evals)
            seed = (seed_base + run_id) if seed_base is not None else None
            run_params.append({
                'run_id': run_id,
                'max_evals': run_max_evals,
                'seed': seed,
            })

        # --- Task 5: failure isolation and serial fallback state ---
        completed_runs_by_id: Dict[int, CMARunResult] = {}
        failed_run_ids: Dict[int, str] = {}  # run_id -> error description
        fallback_run_ids: List[int] = []      # runs recovered via serial fallback
        pool_broken = False

        with ProcessPoolExecutor(
            max_workers=effective_workers,
            mp_context=mp_ctx,
        ) as executor:
            # Submit one future per CMA run
            future_to_run_id = {}
            for rp in run_params:
                logger.info(
                    '=== Global Search: submitting CMA-ES run %d/%d '
                    '(budget=%d evals) ===',
                    rp['run_id'] + 1, n_runs, rp['max_evals'],
                )
                future = executor.submit(
                    _run_cma_worker,
                    _shared_setup,
                    rp['run_id'],
                    sigma0,
                    popsize,
                    rp['max_evals'],
                    max_generations,
                    stagnation_gens,
                    sigma_stop,
                    rp['seed'],
                )
                future_to_run_id[future] = rp['run_id']

            # Collect results as they complete (Task 5: with failure isolation).
            try:
                for future in as_completed(future_to_run_id):
                    rid = future_to_run_id[future]
                    try:
                        run_result = future.result(
                            timeout=_par.worker_timeout_seconds,
                        )
                    except BrokenProcessPool as bpp:
                        # Pool is dead — record this run as failed and break
                        # out to trigger serial fallback for remaining runs.
                        error_msg = f'BrokenProcessPool: {bpp!r}'
                        logger.error(
                            'CMA run %d: pool broken during result retrieval: %s',
                            rid, error_msg,
                        )
                        failed_run_ids[rid] = error_msg
                        pool_broken = True
                        break
                    except Exception as exc:
                        error_msg = f'{type(exc).__name__}: {exc!r}'
                        logger.error(
                            'CMA run %d failed in worker: %s', rid, error_msg,
                            exc_info=True,
                        )
                        failed_run_ids[rid] = error_msg
                        continue

                    if run_result.run_id != rid:
                        logger.warning(
                            'Worker returned run_id=%d for submitted run_id=%d; '
                            'using worker-provided id for aggregation',
                            run_result.run_id, rid,
                        )

                    completed_runs_by_id[run_result.run_id] = run_result

                    logger.info(
                        'Run %d result: obj=%.4f (%+.4f vs ref), '
                        '%d gens, %d evals, %.1fs, stopped_by: %s',
                        run_result.run_id, run_result.best_objective,
                        run_result.best_objective - ref_obj,
                        run_result.n_generations, run_result.n_evals,
                        run_result.wall_seconds, run_result.stop_reason,
                    )
            except BrokenProcessPool as bpp:
                # Pool broke during as_completed() iteration itself
                logger.error(
                    'Process pool broke during result collection: %s', bpp,
                )
                pool_broken = True

        # --- Task 5: serial fallback for pending runs after pool failure ---
        if pool_broken:
            # Identify runs that were NOT successfully completed
            all_run_ids = {rp['run_id'] for rp in run_params}
            completed_ids = set(completed_runs_by_id.keys())
            pending_ids = sorted(all_run_ids - completed_ids - set(failed_run_ids.keys()))

            if pending_ids:
                logger.warning(
                    'Pool broken — falling back to sequential execution for '
                    '%d pending runs: %s',
                    len(pending_ids), pending_ids,
                )
                rp_by_id = {rp['run_id']: rp for rp in run_params}
                for pid in pending_ids:
                    rp = rp_by_id[pid]
                    logger.info(
                        '=== Global Search: serial fallback for run %d/%d '
                        '(budget=%d evals) ===',
                        pid + 1, n_runs, rp['max_evals'],
                    )
                    try:
                        fallback_result = _run_cma_single(
                            ctx, scales, run_id=pid,
                            sigma0=sigma0,
                            popsize=popsize,
                            max_evals=rp['max_evals'],
                            max_generations=max_generations,
                            stagnation_gens=stagnation_gens,
                            sigma_stop=sigma_stop,
                            seed=rp['seed'],
                        )
                        completed_runs_by_id[fallback_result.run_id] = fallback_result
                        fallback_run_ids.append(pid)
                        logger.info(
                            'Fallback run %d result: obj=%.4f (%+.4f vs ref), '
                            '%d gens, %d evals, %.1fs, stopped_by: %s',
                            fallback_result.run_id, fallback_result.best_objective,
                            fallback_result.best_objective - ref_obj,
                            fallback_result.n_generations, fallback_result.n_evals,
                            fallback_result.wall_seconds, fallback_result.stop_reason,
                        )
                    except Exception as fallback_exc:
                        error_msg = f'serial_fallback_{type(fallback_exc).__name__}: {fallback_exc!r}'
                        logger.error(
                            'Serial fallback for run %d also failed: %s',
                            pid, error_msg, exc_info=True,
                        )
                        failed_run_ids[pid] = error_msg
            else:
                logger.info('Pool broken but all runs already completed or failed.')

        # Log failure/fallback summary
        if failed_run_ids:
            logger.warning(
                'Failed runs (%d): %s',
                len(failed_run_ids),
                {rid: reason for rid, reason in sorted(failed_run_ids.items())},
            )
        if fallback_run_ids:
            logger.info(
                'Runs recovered via serial fallback (%d): %s',
                len(fallback_run_ids), fallback_run_ids,
            )

        # Aggregate in deterministic run_id order before dedup/ranking/
        # diagnostics so parallel mode matches sequential merge semantics.
        for run_id in sorted(completed_runs_by_id):
            run_result = completed_runs_by_id[run_id]
            runs.append(run_result)
            total_evals += run_result.n_evals
            _dispatch_mode = (
                'serial_fallback' if run_id in fallback_run_ids else 'parallel'
            )
            budget_status.cumulative_by_run.append({
                'run_id': run_id,
                'n_evals': run_result.n_evals,
                'wall_seconds': run_result.wall_seconds,
                'stop_reason': run_result.stop_reason,
                'best_objective': run_result.best_objective,
                'dispatch_mode': _dispatch_mode,
            })

        # Record failed runs in budget diagnostics
        for rid in sorted(failed_run_ids):
            budget_status.cumulative_by_run.append({
                'run_id': rid,
                'n_evals': 0,
                'wall_seconds': 0.0,
                'stop_reason': f'failed: {failed_run_ids[rid]}',
                'best_objective': float('inf'),
                'dispatch_mode': 'failed',
            })

        budget_status.runs_completed = len(runs)
        if failed_run_ids:
            budget_status.total_stopped_by = (
                f'parallel_failures ({len(failed_run_ids)} runs failed: '
                f'{sorted(failed_run_ids.keys())})'
            )
        if pool_broken:
            budget_status.total_stopped_by = (
                (budget_status.total_stopped_by + '; ' if budget_status.total_stopped_by else '')
                + f'pool_broken (fallback_recovered={len(fallback_run_ids)})'
            )

    else:
        # ---- Sequential dispatch (default, source-of-truth) ----
        for run_id in range(n_runs):

            # Guardrail: total eval budget
            if total_evals >= budget_config.max_total_evals:
                budget_status.total_stopped_by = (
                    f'total_evals_exhausted (run {run_id}/{n_runs})'
                )
                logger.warning(
                    'Global eval budget exhausted (%d >= %d) before run %d',
                    total_evals, budget_config.max_total_evals, run_id,
                )
                break

            # Guardrail: total wall-time budget
            elapsed_global = _time.monotonic() - t_global
            if elapsed_global >= budget_config.max_total_wall_seconds:
                budget_status.total_stopped_by = (
                    f'total_wall_time_exhausted (run {run_id}/{n_runs})'
                )
                logger.warning(
                    'Global wall-time budget exhausted (%.1fs >= %.1fs) before run %d',
                    elapsed_global, budget_config.max_total_wall_seconds, run_id,
                )
                break

            # Remaining budget for this run
            remaining_evals = budget_config.max_total_evals - total_evals
            run_max_evals = min(budget_config.max_per_run_evals, remaining_evals)

            seed = (seed_base + run_id) if seed_base is not None else None

            logger.info(
                '=== Global Search: starting CMA-ES run %d/%d '
                '(budget=%d evals, %.0fs wall) ===',
                run_id + 1, n_runs, run_max_evals,
                budget_config.max_per_run_wall_seconds,
            )

            run_result = _run_cma_single(
                ctx, scales, run_id=run_id,
                sigma0=sigma0,
                popsize=popsize,
                max_evals=run_max_evals,
                max_generations=max_generations,
                stagnation_gens=stagnation_gens,
                sigma_stop=sigma_stop,
                seed=seed,
            )

            runs.append(run_result)
            total_evals += run_result.n_evals
            budget_status.runs_completed = run_id + 1
            budget_status.cumulative_by_run.append({
                'run_id': run_id,
                'n_evals': run_result.n_evals,
                'wall_seconds': run_result.wall_seconds,
                'stop_reason': run_result.stop_reason,
                'best_objective': run_result.best_objective,
            })

            logger.info(
                'Run %d result: obj=%.4f (%+.4f vs ref), '
                '%d gens, %d evals, %.1fs, stopped_by: %s',
                run_id, run_result.best_objective,
                run_result.best_objective - ref_obj,
                run_result.n_generations, run_result.n_evals,
                run_result.wall_seconds, run_result.stop_reason,
            )

    # Normalize run ordering before any downstream consumer (dedup/ranking/
    # diagnostics) to keep sequential and parallel outputs deterministic.
    runs.sort(key=lambda r: r.run_id)
    budget_status.cumulative_by_run.sort(key=lambda row: int(row.get('run_id', -1)))
    budget_status.runs_completed = len(runs)

    # --- 6. De-duplicate and find overall best ---
    candidates_deduped = _deduplicate_candidates(runs, threshold=dedup_threshold)

    if candidates_deduped:
        overall_best = candidates_deduped[0]  # sorted by objective
        best_x_norm = overall_best['x_norm']
        best_x_delta = overall_best['x_delta']
        best_objective = overall_best['objective']
        best_diagnostics = overall_best['diagnostics']
    else:
        # Fallback: no valid runs
        best_x_norm = np.zeros(ctx.n_params, dtype=np.float64)
        best_x_delta = np.zeros(ctx.n_params, dtype=np.float64)
        best_objective = ref_obj
        best_diagnostics = evaluate_candidate(None, ctx)

    total_wall = _time.monotonic() - t_global

    # Update budget_status with final totals
    budget_status.total_evals_used = total_evals
    budget_status.total_wall_seconds = total_wall

    logger.info(
        '=== Global Search complete: %d runs, %d total evals, %.1fs, '
        'best_obj=%.4f (ref=%.4f, delta=%+.4f), %d unique candidates ===',
        len(runs), total_evals, total_wall,
        best_objective, ref_obj, best_objective - ref_obj,
        len(candidates_deduped),
    )

    return GlobalSearchResult(
        runs=runs,
        best_x_norm=best_x_norm,
        best_x_delta=best_x_delta,
        best_objective=best_objective,
        best_diagnostics=best_diagnostics,
        ref_objective=ref_obj,
        probe_result=probe_result,
        candidates_deduped=candidates_deduped,
        total_evals=total_evals,
        total_wall_seconds=total_wall,
        budget_status=budget_status.to_dict(),
    )


# ---------------------------------------------------------------------------
#  Task 5 — Top-K selection and BA refinement handoff
# ---------------------------------------------------------------------------

import time as _time_mod


@dataclass
class RefinementResult:
    """Outcome of BA refinement for a single candidate."""
    candidate_rank: int                 # 0-based rank in top-K
    pre_objective: float                # objective BEFORE BA refinement
    post_objective: float               # objective AFTER BA refinement
    pre_diagnostics: Dict[str, Any]     # diagnostics before BA
    post_diagnostics: Dict[str, Any]    # diagnostics after BA (ray_rmse, len_rmse, etc.)
    refined_planes: Dict[int, Dict]     # window_planes after BA
    refined_cam_params: Dict[int, Any]  # cam_params after BA
    x_delta: np.ndarray                 # the physical delta that seeded this candidate
    wall_seconds: float                 # wall time for this BA run
    success: bool                       # True if BA completed without error
    error: Optional[str] = None         # error message if success=False


def select_top_k_candidates(
    search_result: GlobalSearchResult,
    k: int = 5,
    *,
    include_reference: bool = True,
) -> List[Dict[str, Any]]:
    """Select the top-K candidates from a global search result.

    Parameters
    ----------
    search_result : GlobalSearchResult
        Output of :func:`run_global_search`.
    k : int
        Maximum number of candidates to return.
    include_reference : bool
        If True and the reference state (zero delta) is not already among
        the candidates, prepend it as rank 0 so it competes in refinement.

    Returns
    -------
    list of dict
        Each dict has keys ``x_norm``, ``x_delta``, ``objective``,
        ``diagnostics``, ``run_id``.  Sorted ascending by ``objective``
        (best first).  At most *k* entries.
    """
    candidates = list(search_result.candidates_deduped)

    # Optionally inject the reference (zero-delta) if missing
    if include_reference:
        has_ref = any(
            np.allclose(c['x_delta'], 0.0, atol=1e-12)
            for c in candidates
        )
        if not has_ref:
            n_params = search_result.best_x_norm.shape[0]
            ref_candidate: Dict[str, Any] = {
                'x_norm': np.zeros(n_params, dtype=np.float64),
                'x_delta': np.zeros(n_params, dtype=np.float64),
                'objective': search_result.ref_objective,
                'diagnostics': search_result.best_diagnostics
                    if np.allclose(search_result.best_x_delta, 0.0, atol=1e-12)
                    else {'objective': search_result.ref_objective, 'success': True},
                'run_id': -1,  # sentinel: reference state
            }
            candidates.insert(0, ref_candidate)

    # Sort by objective (ascending = best first) and trim to k
    candidates.sort(key=lambda c: c['objective'])
    top_k = candidates[:k]

    logger.info(
        'Selected top-%d candidates (from %d). Objectives: %s',
        len(top_k), len(candidates),
        ', '.join(f"{c['objective']:.4f}" for c in top_k),
    )
    return top_k


def refine_candidates_ba(
    candidates: List[Dict[str, Any]],
    ref_state: Dict[str, Any],
    dataset: Dict[str, Any],
    wand_length: float,
    *,
    ba_config_overrides: Optional[Dict[str, Any]] = None,
    lambda_base_per_cam: float = 2.0,
    max_frames: int = 50000,
    verbosity: int = 1,
) -> List[RefinementResult]:
    """Run BA refinement on each candidate from a global search.

    For every candidate in *candidates*, this function:

    1. Decodes the candidate's ``x_delta`` into concrete planes / cam_params
       / media using the same layout as the global-search evaluator.
    2. Creates a **fresh** :class:`RefractiveBAOptimizer` seeded with that
       decoded state (complete isolation between candidates).
    3. Calls ``optimizer.optimize()`` and collects before/after metrics.

    Parameters
    ----------
    candidates : list of dict
        As returned by :func:`select_top_k_candidates`.
    ref_state : dict
        Reference state from :func:`load_reference_state`.
    dataset : dict
        Observation dataset from :func:`load_observations_csv`.
    wand_length : float
        Physical wand length in mm.
    ba_config_overrides : dict, optional
        Keys to override on the default :class:`RefractiveBAConfig`.
        Useful for controlling ``stage``, ``max_frames``,
        ``verbosity``, etc.
    lambda_base_per_cam : float
        Per-camera weight for wand-length residuals.
    max_frames : int
        Max frames for evaluation (passed to both context and BA).
    verbosity : int
        BA verbosity level.

    Returns
    -------
    list of :class:`RefinementResult`
        One per candidate, in the same order as *candidates*.
    """
    from .refraction_calibration_BA import RefractiveBAConfig, RefractiveBAOptimizer
    from .refraction_wand_calibrator import CppCameraFactory

    results: List[RefinementResult] = []

    # Build a temporary evaluation context to decode deltas via _unpack_params_delta
    decode_ctx = build_evaluation_context(
        ref_state, dataset, wand_length,
        lambda_base_per_cam=lambda_base_per_cam,
        max_frames=max_frames,
    )

    for rank, cand in enumerate(candidates):
        t0 = _time_mod.monotonic()
        x_delta = np.asarray(cand['x_delta'], dtype=np.float64)
        pre_obj = float(cand['objective'])
        pre_diag = dict(cand.get('diagnostics', {}))

        logger.info(
            '--- Refining candidate rank=%d (run_id=%s, pre_obj=%.4f) ---',
            rank, cand.get('run_id', '?'), pre_obj,
        )

        try:
            # --- 1. Decode delta into concrete state ---
            opt_decode = decode_ctx.optimizer
            curr_planes, curr_cams, curr_media = opt_decode._unpack_params_delta(
                x_delta, decode_ctx.layout,
            )

            # --- 2. Create fresh C++ cameras for this candidate ---
            image_sizes = ref_state['metadata']['image_sizes']
            base_stub = _CameraSettingsStub(image_sizes, curr_cams)
            cam_to_window = ref_state['cam_to_window']
            cams_cpp = CppCameraFactory.init_cams_cpp_in_memory(
                base_stub, curr_cams, curr_media,
                cam_to_window, curr_planes,
            )

            # --- 3. BA config for refinement ---
            cfg_kwargs: Dict[str, Any] = {
                'use_regularization': False,
                'max_frames': max_frames,
                'lambda_base_per_cam': lambda_base_per_cam,
                'dist_coeff_num': dataset.get('dist_coeff_num', 0),
                'use_proj_residuals': False,
                'skip_optimization': False,
                'verbosity': verbosity,
            }
            if ba_config_overrides:
                cfg_kwargs.update(ba_config_overrides)
            cfg = RefractiveBAConfig(**cfg_kwargs)

            # --- 4. Create fresh optimizer with candidate's state ---
            ba_opt = RefractiveBAOptimizer(
                dataset=dataset,
                cam_params=curr_cams,
                cams_cpp=cams_cpp,
                cam_to_window=cam_to_window,
                window_media=curr_media,
                window_planes=curr_planes,
                wand_length=wand_length,
                config=cfg,
            )

            # --- 5. Run BA refinement ---
            refined_planes, refined_cams = ba_opt.optimize()

            # --- 6. Evaluate post-refinement metrics ---
            ba_opt._compute_physical_sigmas()
            n_cams = len(refined_cams)
            lambda_eff = lambda_base_per_cam * n_cams
            layout_post = ba_opt._get_param_layout(
                enable_planes=True,
                enable_cam_t=True,
                enable_cam_r=True,
                enable_cam_f=False,
                enable_win_t=False,
                enable_cam_k1=False,
                enable_cam_k2=False,
            )
            # Zero-delta evaluates the current (refined) state
            x_zero = np.zeros(len(layout_post), dtype=np.float64)
            post_planes, post_cams, post_media = ba_opt._unpack_params_delta(
                x_zero, layout_post,
            )
            (_, S_ray, S_len, N_ray, N_len,
             S_proj, N_proj) = ba_opt.evaluate_residuals(
                post_planes, post_cams, lambda_eff,
                window_media=post_media,
            )

            post_obj = float(S_ray + lambda_eff * S_len)
            ray_rmse = float(np.sqrt(S_ray / max(N_ray, 1)))
            len_rmse = float(np.sqrt(S_len / max(N_len, 1)))
            proj_rmse = float(np.sqrt(S_proj / max(N_proj, 1))) if N_proj > 0 else 0.0

            post_diag: Dict[str, Any] = {
                'objective': post_obj,
                'ray_rmse': ray_rmse,
                'len_rmse': len_rmse,
                'proj_rmse': proj_rmse,
                'success': True,
                'error': None,
                'n_ray': int(N_ray),
                'n_len': int(N_len),
                'n_proj': int(N_proj),
                's_ray': float(S_ray),
                's_len': float(S_len),
            }

            wall = _time_mod.monotonic() - t0
            logger.info(
                '  Candidate rank=%d refined: pre_obj=%.4f -> post_obj=%.4f '
                '(delta=%+.4f), ray_rmse=%.4fmm, len_rmse=%.4fmm, %.1fs',
                rank, pre_obj, post_obj, post_obj - pre_obj,
                ray_rmse, len_rmse, wall,
            )

            results.append(RefinementResult(
                candidate_rank=rank,
                pre_objective=pre_obj,
                post_objective=post_obj,
                pre_diagnostics=pre_diag,
                post_diagnostics=post_diag,
                refined_planes=refined_planes,
                refined_cam_params=refined_cams,
                x_delta=x_delta,
                wall_seconds=wall,
                success=True,
            ))

        except Exception as exc:
            wall = _time_mod.monotonic() - t0
            err_msg = f'{type(exc).__name__}: {exc}'
            logger.error(
                '  Candidate rank=%d FAILED: %s (%.1fs)', rank, err_msg, wall,
            )
            results.append(RefinementResult(
                candidate_rank=rank,
                pre_objective=pre_obj,
                post_objective=_SENTINEL_OBJECTIVE,
                pre_diagnostics=pre_diag,
                post_diagnostics={'success': False, 'error': err_msg},
                refined_planes={},
                refined_cam_params={},
                x_delta=x_delta,
                wall_seconds=wall,
                success=False,
                error=err_msg,
            ))

    # Summary
    ok_results = [r for r in results if r.success]
    if ok_results:
        best = min(ok_results, key=lambda r: r.post_objective)
        logger.info(
            '=== BA refinement complete: %d/%d succeeded, best rank=%d '
            'post_obj=%.4f (pre=%.4f, improvement=%+.4f) ===',
            len(ok_results), len(results), best.candidate_rank,
            best.post_objective, best.pre_objective,
            best.post_objective - best.pre_objective,
        )
    else:
        logger.warning('=== BA refinement: ALL %d candidates failed ===', len(results))

    return results


# ---------------------------------------------------------------------------
#  Task 6 — Diagnostics and logging (per-eval + per-generation)
# ---------------------------------------------------------------------------


def _serialize_np(value: Any) -> Any:
    """Recursively convert numpy types to JSON-serialisable Python types.

    Mirrors the helper in ``pretest_global_search.py`` so that diagnostics
    written from either module have identical dtype handling.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {str(k): _serialize_np(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_np(v) for v in value]
    return value


# -- Flattening helpers for CSV rows -----------------------------------------

def _flatten_candidate_for_csv(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a de-duplicated candidate dict into a flat CSV-friendly row.

    Parameters
    ----------
    candidate : dict
        Dict with keys ``x_norm``, ``x_delta``, ``objective``,
        ``diagnostics``, ``run_id`` (as produced by
        :func:`_deduplicate_candidates`).

    Returns
    -------
    dict
        Flat dict with scalar values suitable for :class:`csv.DictWriter`.
    """
    diag = candidate.get('diagnostics', {})
    row: Dict[str, Any] = {
        'run_id': candidate.get('run_id', -1),
        'objective': candidate.get('objective', None),
        'ray_rmse': diag.get('ray_rmse', None),
        'len_rmse': diag.get('len_rmse', None),
        'proj_rmse': diag.get('proj_rmse', None),
        'n_ray': diag.get('n_ray', None),
        'n_len': diag.get('n_len', None),
        'n_proj': diag.get('n_proj', None),
        's_ray': diag.get('s_ray', None),
        's_len': diag.get('s_len', None),
        'success': diag.get('success', None),
        'error': diag.get('error', None),
    }
    # Append x_delta components as separate columns
    x_delta = candidate.get('x_delta', None)
    if x_delta is not None:
        x_delta = np.asarray(x_delta)
        for i, val in enumerate(x_delta.flat):
            row[f'x_delta_{i}'] = float(val)
    return row


def _flatten_generation_log(gl: GenerationLog, run_id: int) -> Dict[str, Any]:
    """Flatten a :class:`GenerationLog` entry into a CSV-friendly row.

    Parameters
    ----------
    gl : GenerationLog
        Single generation snapshot.
    run_id : int
        CMA-ES run identifier.

    Returns
    -------
    dict
        Flat dict with scalar values.
    """
    return {
        'run_id': run_id,
        'gen': gl.gen,
        'best_objective': float(gl.best_objective),
        'median_objective': float(gl.median_objective),
        'worst_objective': float(gl.worst_objective),
        'feasible_fraction': float(gl.feasible_fraction),
        'sigma': float(gl.sigma),
        'cumulative_evals': gl.cumulative_evals,
        'cumulative_wall_seconds': float(gl.cumulative_wall_seconds),
    }


# -- CSV writers -------------------------------------------------------------

def write_eval_csv(
    csv_path: str | Path,
    result: GlobalSearchResult,
) -> Path:
    """Write per-candidate evaluation diagnostics to a CSV file.

    Each row corresponds to one de-duplicated candidate from the global
    search, with columns for objective components, residual counts, and
    the physical delta vector.

    Parameters
    ----------
    csv_path : path-like
        Destination CSV path (created/overwritten).
    result : GlobalSearchResult
        Completed global search result.

    Returns
    -------
    Path
        The resolved output path.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [_flatten_candidate_for_csv(c) for c in result.candidates_deduped]
    if not rows:
        logger.warning('write_eval_csv: no candidates to write')
        return csv_path

    # Determine fieldnames from first row (preserves insertion order)
    fieldnames = list(rows[0].keys())
    # Ensure all rows contribute columns (x_delta may differ in length)
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    for k in sorted(all_keys - set(fieldnames)):
        fieldnames.append(k)

    with csv_path.open('w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    logger.info('Wrote %d candidate rows to %s', len(rows), csv_path)
    return csv_path


def write_generation_csv(
    csv_path: str | Path,
    result: GlobalSearchResult,
) -> Path:
    """Write per-generation diagnostics across all CMA-ES runs to CSV.

    Each row is one generation from one CMA-ES run, with columns for
    best/median/worst objective, feasible fraction, sigma, cumulative
    evals, and wall time.

    Parameters
    ----------
    csv_path : path-like
        Destination CSV path (created/overwritten).
    result : GlobalSearchResult
        Completed global search result.

    Returns
    -------
    Path
        The resolved output path.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for run in sorted(result.runs, key=lambda r: r.run_id):
        for gl in run.generation_log:
            rows.append(_flatten_generation_log(gl, run.run_id))

    if not rows:
        logger.warning('write_generation_csv: no generation records to write')
        return csv_path

    fieldnames = [
        'run_id', 'gen', 'best_objective', 'median_objective',
        'worst_objective', 'feasible_fraction', 'sigma',
        'cumulative_evals', 'cumulative_wall_seconds',
    ]

    with csv_path.open('w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info('Wrote %d generation rows to %s', len(rows), csv_path)
    return csv_path


def write_generation_detail_csv(
    csv_path: str | Path,
    run: CMARunResult,
) -> Path:
    """Write per-run generation-detail CSV with extended metrics and real-value ranges.

    Produces one CSV per run containing best ray/len RMSE, sigma, best real
    parameter values, and empirical real-value min/max for every parameter
    across the sampled population in each generation.

    Parameters
    ----------
    csv_path : path-like
        Destination CSV path (created/overwritten).
    run : CMARunResult
        Single CMA-ES run result with generation_log.

    Returns
    -------
    Path
        The resolved output path.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    param_names = []  # inferred from first row

    for gl in run.generation_log:
        row = {
            'run_id': run.run_id,
            'gen': gl.gen,
            'best_objective': float(gl.best_objective),
            'median_objective': float(gl.median_objective),
            'worst_objective': float(gl.worst_objective),
            'feasible_fraction': float(gl.feasible_fraction),
            'sigma': float(gl.sigma),
            'cumulative_evals': gl.cumulative_evals,
            'cumulative_wall_seconds': float(gl.cumulative_wall_seconds),
            'best_ray_rmse': float(gl.best_ray_rmse),
            'best_len_rmse': float(gl.best_len_rmse),
        }

        # Infer parameter names from first generation that has any real params
        if not param_names:
            src = gl.best_real_params if gl.best_real_params is not None else (
                gl.pop_real_min if gl.pop_real_min is not None else gl.pop_real_max
            )
            if src is not None:
                n_params = len(src)
                param_names = [f'param_{i:02d}' for i in range(n_params)]

        # Add best real parameter values
        if gl.best_real_params is not None and param_names:
            for i, val in enumerate(gl.best_real_params):
                if i < len(param_names):
                    row[f'best_real_{param_names[i]}'] = float(val)

        # Add per-parameter real min/max
        if gl.pop_real_min is not None and param_names:
            for i, val in enumerate(gl.pop_real_min):
                if i < len(param_names):
                    row[f'real_min_{param_names[i]}'] = float(val)

        if gl.pop_real_max is not None and param_names:
            for i, val in enumerate(gl.pop_real_max):
                if i < len(param_names):
                    row[f'real_max_{param_names[i]}'] = float(val)

        rows.append(row)

    if not rows:
        logger.warning('write_generation_detail_csv: no generation records to write')
        return csv_path

    # Build column order: summary cols first, then detail cols
    base_cols = [
        'run_id', 'gen', 'best_objective', 'median_objective',
        'worst_objective', 'feasible_fraction', 'sigma',
        'cumulative_evals', 'cumulative_wall_seconds',
        'best_ray_rmse', 'best_len_rmse',
    ]

    # Collect all parameter-related columns from rows
    detail_cols = set()
    for row in rows:
        detail_cols.update(
            k for k in row.keys()
            if k.startswith('best_real_') or k.startswith('real_min_') or k.startswith('real_max_')
        )

    # Sort detail columns to group by parameter: param_0 best/min/max, param_1 best/min/max, etc.
    detail_cols_sorted = sorted(detail_cols)

    fieldnames = base_cols + detail_cols_sorted

    with csv_path.open('w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, restval='')
        writer.writeheader()
        writer.writerows(rows)

    logger.info(
        'Wrote %d generation-detail rows to %s (run_id=%d)',
        len(rows), csv_path, run.run_id
    )
    return csv_path

# -- JSON writer -------------------------------------------------------------

def write_diagnostics_json(
    json_path: str | Path,
    result: GlobalSearchResult,
    *,
    include_all_runs: bool = True,
) -> Path:
    """Serialise a :class:`GlobalSearchResult` to a human-readable JSON file.

    The output includes:

    * **summary** — best objective, reference objective, total evals/wall-time.
    * **probe** — probe scales and reference objective.
    * **candidates** — de-duplicated candidate list with full diagnostics.
    * **runs** (optional) — per-run metadata and generation logs.

    Parameters
    ----------
    json_path : path-like
        Destination JSON path (created/overwritten).
    result : GlobalSearchResult
        Completed global search result.
    include_all_runs : bool
        If True (default), include the full generation log for every run.
        Set False to reduce file size when only the summary/candidates
        are needed.

    Returns
    -------
    Path
        The resolved output path.
    """
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    data: Dict[str, Any] = {
        'summary': {
            'best_objective': result.best_objective,
            'ref_objective': result.ref_objective,
            'improvement': result.best_objective - result.ref_objective,
            'n_runs': len(result.runs),
            'n_candidates_deduped': len(result.candidates_deduped),
            'total_evals': result.total_evals,
            'total_wall_seconds': result.total_wall_seconds,
        },
        'probe': {
            'ref_objective': result.probe_result.ref_objective,
            'n_evals': result.probe_result.n_evals,
            'scales': result.probe_result.scales,
        },
        'best_diagnostics': result.best_diagnostics,
        'best_x_delta': result.best_x_delta,
        'candidates': [
            {
                'rank': i,
                'run_id': c.get('run_id', -1),
                'objective': c.get('objective'),
                'x_delta': c.get('x_delta'),
                'diagnostics': c.get('diagnostics', {}),
            }
            for i, c in enumerate(result.candidates_deduped)
        ],
    }

    if include_all_runs:
        data['runs'] = []
        for run in sorted(result.runs, key=lambda r: r.run_id):
            run_dict: Dict[str, Any] = {
                'run_id': run.run_id,
                'best_objective': run.best_objective,
                'n_generations': run.n_generations,
                'n_evals': run.n_evals,
                'wall_seconds': run.wall_seconds,
                'stop_reason': run.stop_reason,
                'best_diagnostics': run.best_diagnostics,
                'best_x_delta': run.best_x_delta,
                'generation_log': [
                    {
                        'gen': gl.gen,
                        'best_objective': gl.best_objective,
                        'median_objective': gl.median_objective,
                        'worst_objective': gl.worst_objective,
                        'feasible_fraction': gl.feasible_fraction,
                        'sigma': gl.sigma,
                        'cumulative_evals': gl.cumulative_evals,
                        'cumulative_wall_seconds': gl.cumulative_wall_seconds,
                    }
                    for gl in run.generation_log
                ],
            }
            data['runs'].append(run_dict)

    # Recursively convert numpy types, then write
    data = _serialize_np(data)

    with json_path.open('w', encoding='utf-8') as fp:
        json.dump(data, fp, indent=2)

    logger.info('Wrote diagnostics JSON to %s', json_path)
    return json_path


# -- Convenience: emit all diagnostics at once --------------------------------

def emit_diagnostics(
    result: GlobalSearchResult,
    output_dir: str | Path,
    *,
    prefix: str = 'full_global',
    include_all_runs: bool = True,
    generation_detail_config: Optional[GenerationDetailConfig] = None,
) -> Dict[str, Path]:
    """Write all diagnostic artifacts for a completed global search.

    :func:`write_eval_csv`, :func:`write_generation_csv`, and
    :func:`write_generation_detail_csv` for each run in one shot.
    :func:`write_eval_csv`, and :func:`write_generation_csv` in one shot.
    :func:`write_eval_csv`, and :func:`write_generation_csv` in one shot.

    Parameters
    ----------
    result : GlobalSearchResult
        Completed global search result.
    output_dir : path-like
        Directory where diagnostic files are written.  Created if it
        does not exist.
    prefix : str
        Filename prefix (default ``'full_global'``).  Produces files
        named ``{prefix}_diagnostics.json``, ``{prefix}_eval.csv``,
        and ``{prefix}_generation.csv``.
    include_all_runs : bool
        Passed through to :func:`write_diagnostics_json`.
    generation_detail_config : GenerationDetailConfig or None
        Configuration for per-run generation-detail CSV output.  If None,
        defaults to disabled detail output (preserving current behavior).
        When ``enable=True``, per-run detail CSV files are written; when
        ``enable=False`` (default), detail output is skipped.

    Returns
    -------
    dict[str, Path]
        Mapping from artifact name (``'json'``, ``'eval_csv'``,
        ``'generation_csv'``, ``'detail_csvs'``) to file path(s). The
        ``'detail_csvs'`` entry is a dict mapping run_id to per-run detail CSV path.
        ``'generation_csv'``) to the written file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {}

    paths['json'] = write_diagnostics_json(
        output_dir / f'{prefix}_diagnostics.json',
        result,
        include_all_runs=include_all_runs,
    )

    paths['eval_csv'] = write_eval_csv(
        output_dir / f'{prefix}_eval.csv',
        result,
    )

    paths['generation_csv'] = write_generation_csv(
        output_dir / f'{prefix}_generation.csv',
        result,
    )

    # Write per-run generation-detail CSVs (only if enabled)
    detail_csvs: Dict[int, Path] = {}
    if generation_detail_config is None:
        generation_detail_config = GenerationDetailConfig()  # defaults: enable=False
    
    if generation_detail_config.enable:
        for run in sorted(result.runs, key=lambda r: r.run_id):
            detail_csv_path = write_generation_detail_csv(
                output_dir / f'{generation_detail_config.prefix}_run{run.run_id:03d}_detail.csv',
                run,
            )
            detail_csvs[run.run_id] = detail_csv_path

    # Add detail_csvs to paths only if detail CSVs were written
    if detail_csvs:
        paths['detail_csvs'] = detail_csvs
    logger.info(
        'Emitted diagnostics to %s: %s',
        output_dir,
        {k: v.name if isinstance(v, Path) else {rid: p.name for rid, p in v.items()} for k, v in paths.items()},
    )

    return paths
