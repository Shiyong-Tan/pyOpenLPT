"""Read-only, out-of-sample quality gate for OpenLPT VSC camera files."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np

from . import vsc_service
from .camera_io import parse_camera_file
from .dataset_identity import (
    compute_dataset_fingerprint,
    compute_frame_range_fingerprint,
)
from .refraction_optimizer import RefractionVSCOptimizer


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _natural_index(path):
    match = re.search(r"(\d+)(?=\.txt$|\.csv$)", os.path.basename(path))
    return int(match.group(1)) if match else -1


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {key: None for key in ("mean", "rmse", "median", "p90", "p95", "max")} | {"n": 0}
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "rmse": float(np.sqrt(np.mean(values * values))),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _read_output_dir(project_dir):
    config = os.path.join(project_dir, "config.txt")
    output = os.path.join(project_dir, "Results")
    if not os.path.isfile(config):
        return output
    with open(config, "r", encoding="utf-8") as stream:
        lines = stream.readlines()
    for index, line in enumerate(lines):
        if "Output Folder Path" not in line:
            continue
        for value in lines[index + 1:]:
            value = value.strip()
            if not value or value.startswith("#"):
                continue
            return os.path.normpath(value if os.path.isabs(value)
                                    else os.path.join(project_dir, value))
    return output


def _read_frame_range(project_dir):
    """Read the inclusive LPT frame range from a project's config.txt."""
    config = os.path.join(project_dir, "config.txt")
    if not os.path.isfile(config):
        raise RuntimeError(f"Configuration file not found: {config}")
    with open(config, "r", encoding="utf-8") as stream:
        lines = stream.readlines()
    for index, line in enumerate(lines):
        if "Frame Range" not in line:
            continue
        for value in lines[index + 1:]:
            value = value.strip()
            if not value or value.startswith("#"):
                continue
            parts = [part.strip() for part in value.split(",")]
            if len(parts) < 2:
                break
            start, end = int(parts[0]), int(parts[1])
            if end < start:
                raise RuntimeError(
                    f"Invalid frame range in {config}: {start}..{end}"
                )
            return start, end
    raise RuntimeError(f"Could not read '# Frame Range' from {config}")


def _provenance_frame_range(provenance):
    """Return the conservative inclusive range that contributed to VSC."""
    actual = provenance.get("actual_correspondence_frame_range", {})
    requested = provenance.get("requested_frame_range", {})
    start = actual.get("start")
    end = actual.get("end")
    if start is None:
        start = requested.get("start")
    if end is None:
        end = requested.get("end")
    if start is None or end is None:
        return None
    return int(start), int(end)


def _subtract_inclusive_range(whole, excluded):
    """Subtract one inclusive interval, returning zero, one, or two intervals."""
    start, end = whole
    if excluded is None:
        return []
    exclude_start, exclude_end = excluded
    if exclude_end < start or exclude_start > end:
        return [(start, end)]
    result = []
    if start < exclude_start:
        result.append((start, min(end, exclude_start - 1)))
    if end > exclude_end:
        result.append((max(start, exclude_end + 1), end))
    return result


def _frame_in_ranges(frame_id, ranges):
    return any(start <= frame_id <= end for start, end in ranges)


def vsc_source_frame_count(project_dir):
    """Return (count, source) from provenance or a legacy VSC log."""
    project_dir = os.path.abspath(project_dir)
    provenance_path = os.path.join(
        project_dir, "camFile_VSC", "vsc_provenance.json"
    )
    if os.path.isfile(provenance_path):
        try:
            with open(provenance_path, "r", encoding="utf-8") as stream:
                provenance = json.load(stream)
            count = provenance.get(
                "actual_correspondence_frame_range", {}
            ).get("unique_frames")
            if count is not None:
                return int(count), "provenance"
        except Exception:
            pass
    log_path = os.path.join(project_dir, "VSC_log.txt")
    if os.path.isfile(log_path):
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as stream:
                text = stream.read()
            matches = re.findall(r"Processing\s+([\d,]+)\s+unique frames", text)
            if matches:
                return int(matches[-1].replace(",", "")), "legacy VSC log"
        except Exception:
            pass
    return None, "unknown"


def assess_validation_scope(source_project, validation_project=None):
    """Describe whether a trial contains frame ranges independent of a tagged VSC.

    A different trial is independent regardless of matching frame numbers. When the
    VSC and validation data come from the same trial, only frames outside the
    conservative VSC fitting interval are eligible.
    """
    source_project = os.path.abspath(source_project)
    validation_project = os.path.abspath(validation_project or source_project)
    provenance_path = os.path.join(
        source_project, "camFile_VSC", "vsc_provenance.json"
    )
    if not os.path.isfile(provenance_path):
        return {
            "can_validate_by_range": False,
            "reason": "The VSC has no provenance tag, so its fitting range is unknown.",
            "source_project_path": source_project,
            "validation_project_path": validation_project,
            "validation_frame_range": _read_frame_range(validation_project),
            "vsc_frame_range": None,
            "eligible_frame_ranges": [],
            "same_trial": None,
        }
    with open(provenance_path, "r", encoding="utf-8") as stream:
        provenance = json.load(stream)
    tagged_source = provenance.get("source_project_path") or source_project
    vsc_range = _provenance_frame_range(provenance)
    source_range_fingerprint = provenance.get("source_frame_range_fingerprint", {})
    validation_range_fingerprint = (
        compute_frame_range_fingerprint(validation_project, vsc_range)
        if vsc_range else {"available": False}
    )
    source_fingerprint = provenance.get("source_dataset_fingerprint", {})
    validation_fingerprint = compute_dataset_fingerprint(validation_project)
    range_fingerprints_available = (
        source_range_fingerprint.get("available")
        and validation_range_fingerprint.get("available")
        and source_range_fingerprint.get("digest")
        and validation_range_fingerprint.get("digest")
    )
    full_fingerprints_available = (
        source_fingerprint.get("available")
        and validation_fingerprint.get("available")
        and source_fingerprint.get("digest")
        and validation_fingerprint.get("digest")
    )
    if range_fingerprints_available:
        same_trial = (
            source_range_fingerprint["digest"]
            == validation_range_fingerprint["digest"]
        )
        identity_basis = "VSC-source raw-image fingerprint"
    elif full_fingerprints_available:
        same_trial = (
            source_fingerprint["digest"] == validation_fingerprint["digest"]
        )
        identity_basis = "whole-acquisition raw-image fingerprint"
    else:
        # Backward-compatible conservative fallback for older provenance files.
        same_trial = (
            os.path.normcase(os.path.abspath(tagged_source))
            == os.path.normcase(validation_project)
        )
        identity_basis = "project path (image fingerprint unavailable)"
    validation_range = _read_frame_range(validation_project)
    if same_trial:
        eligible = _subtract_inclusive_range(validation_range, vsc_range)
        reason = (
            "Held-out frames exist outside the VSC fitting range."
            if eligible else
            "The configured range contains no frames outside the VSC fitting range."
        )
    else:
        eligible = [validation_range]
        reason = "The validation data come from a different trial."
    return {
        "can_validate_by_range": bool(eligible),
        "reason": reason,
        "source_project_path": source_project,
        "tagged_source_project_path": tagged_source,
        "source_trial": provenance.get("source_trial") or os.path.basename(tagged_source),
        "validation_project_path": validation_project,
        "validation_frame_range": validation_range,
        "vsc_frame_range": vsc_range,
        "eligible_frame_ranges": eligible,
        "same_trial": same_trial,
        "identity_basis": identity_basis,
        "source_dataset_digest": source_fingerprint.get("digest"),
        "validation_dataset_digest": validation_fingerprint.get("digest"),
        "source_range_digest": source_range_fingerprint.get("digest"),
        "validation_range_digest": validation_range_fingerprint.get("digest"),
    }


def _parse_track_row(row, n_cams):
    if len(row) < 6 + 3 * n_cams:
        return None
    try:
        item = {
            "frame_id": int(row[1]),
            "pt3d": np.asarray([float(row[2]), float(row[3]), float(row[4])], dtype=np.float64),
            "r3d_mm": float(row[5]),
            "csv_cam_2d": {},
        }
        for cam in range(n_cams):
            offset = 6 + 3 * cam
            x, y, radius = map(float, row[offset:offset + 3])
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(radius)
                    and x > 0.0 and y > 0.0 and radius > 0.0):
                return None
            item["csv_cam_2d"][cam] = (x, y, radius)
        return item
    except (ValueError, IndexError):
        return None


def _evenly_spaced(items, count):
    if len(items) <= count:
        return items
    indices = np.linspace(0, len(items) - 1, count, dtype=int)
    return [items[int(index)] for index in indices]


def _sample_track_candidates(project_dir, frame_count, max_points_per_frame, n_cams,
                             allowed_ranges=None, random_seed=None):
    track_dir = os.path.join(_read_output_dir(project_dir), "ConvergeTrack")
    if not os.path.isdir(track_dir):
        fallback = os.path.join(project_dir, "Results", "ConvergeTrack")
        track_dir = fallback if os.path.isdir(fallback) else track_dir
    files = [
        os.path.join(track_dir, name) for name in os.listdir(track_dir)
        if name.startswith("LongTrackInactive_") and name.endswith(".csv")
    ] if os.path.isdir(track_dir) else []
    if not files and os.path.isdir(track_dir):
        # Short or sparse validation runs may not have produced an inactive
        # long-track snapshot yet. Fall back to the current active and exited
        # collections rather than declaring the dataset unusable.
        files = [
            os.path.join(track_dir, name) for name in os.listdir(track_dir)
            if name.endswith(".csv") and name.startswith(
                ("LongTrackActive_", "ExitTrack_")
            )
        ]
    files.sort(key=_natural_index)
    if not files:
        raise RuntimeError(f"No long-track CSV files found in {track_dir}")

    # For same-trial validation we must first filter every candidate by frame,
    # then distribute the requested sample over files that actually contain
    # held-out data. Selecting files before filtering can silently miss a narrow
    # eligible interval.
    available = []
    for path in files:
        by_frame = defaultdict(list)
        with open(path, "r", newline="") as stream:
            reader = csv.reader(stream)
            next(reader, None)
            for row in reader:
                item = _parse_track_row(row, n_cams)
                if item is None:
                    continue
                if allowed_ranges and not _frame_in_ranges(item["frame_id"], allowed_ranges):
                    continue
                by_frame[item["frame_id"]].append(item)
        if by_frame:
            ranked = sorted(
                by_frame.items(), key=lambda pair: (-len(pair[1]), pair[0])
            )
            compact = []
            # A few alternatives preserve the duplicate-frame fallback without
            # retaining every parsed CSV row from every checkpoint in memory.
            for frame_id, rows in ranked[:8]:
                if len(rows) > max_points_per_frame:
                    indices = np.linspace(
                        0, len(rows) - 1, max_points_per_frame, dtype=int
                    )
                    rows = [rows[int(index)] for index in indices]
                compact.append((frame_id, rows))
            available.append((path, compact))

    if random_seed is None:
        selected = _evenly_spaced(available, frame_count)
    else:
        rng = random.Random(int(random_seed))
        selected = rng.sample(available, min(int(frame_count), len(available)))
    observations = []
    frame_ids = []
    source_files = []
    seen_frames = set()
    for path, ranked in selected:
        picked = next(((fid, rows) for fid, rows in ranked if fid not in seen_frames), None)
        if picked is None:
            continue
        frame_id, rows = picked
        seen_frames.add(frame_id)
        observations.extend(rows)
        frame_ids.append(frame_id)
        source_files.append(os.path.basename(path))
    ordered_sources = sorted(zip(frame_ids, source_files))
    frame_ids = [frame_id for frame_id, _ in ordered_sources]
    source_files = [source for _, source in ordered_sources]
    return observations, frame_ids, source_files


def _redetect(project_dir, candidates, n_cams, image_size, log):
    params = {
        "obj_type": "Bubble",
        "obj_radius": 5.0,
        "margin_factor": 4.0,
        "search_radius": 4.0,
        "isolation_margin": 2.0,
        "tolerance_mode": "custom",
        "tolerance_value": 20.0,
    }
    vsc_service.init_worker(project_dir, params, list(range(n_cams)))
    grouped = defaultdict(list)
    for item in candidates:
        xyz = item["pt3d"]
        grouped[item["frame_id"]].append((
            float(xyz[0]), float(xyz[1]), float(xyz[2]), item["frame_id"],
            item["r3d_mm"], item["csv_cam_2d"],
        ))

    detected = []
    aggregate = Counter()
    errors = []
    for position, frame_id in enumerate(sorted(grouped), start=1):
        rows, stats = vsc_service.process_frame_task(
            frame_id, grouped[frame_id], image_size, params
        )
        detected.extend(rows)
        for key, value in stats.items():
            if key == "error":
                if value:
                    errors.append(str(value))
            elif isinstance(value, (int, np.integer)):
                aggregate[key] += int(value)
        log(f"  Validation frame {position}/{len(grouped)}: {frame_id}, clean observations={len(rows)}")
    for corr_id, item in enumerate(detected):
        item["corr_id"] = corr_id
    return detected, {
        "attempted": len(candidates),
        "accepted": len(detected),
        "stats": dict(aggregate),
        "errors": errors,
        "selection_tolerance_px": params["tolerance_value"],
    }


def _make_point3d(lpt, xyz):
    point = lpt.Pt3D()
    point[0], point[1], point[2] = map(float, xyz)
    return point


def _load_problem(camera_files, correspondences):
    import pyopenlpt as lpt

    models, states, sizes = {}, {}, {}
    cam_to_window, window_planes = {}, {}
    for cam_id, path in enumerate(camera_files):
        parsed = parse_camera_file(path)
        camera = lpt.Camera(path)
        if str(camera._type).split(".")[-1].upper() != "PINPLATE":
            raise RuntimeError("External VSC validation currently requires PINPLATE cameras")
        models[cam_id] = camera
        sizes[cam_id] = tuple(parsed["img_size"])
        pin = camera._pinplate_param
        rvec = camera.rmtxTorvec(pin.r_mtx)
        states[cam_id] = {
            "rvec": np.asarray([rvec[0], rvec[1], rvec[2]], dtype=np.float64),
            "tvec": np.asarray([pin.t_vec[0], pin.t_vec[1], pin.t_vec[2]], dtype=np.float64),
            "is_active": bool(camera._is_active),
            "max_intensity": float(camera._max_intensity),
        }
        meta = parsed.get("ref_meta", {}) or {}
        window_id = int(meta.get("window_id", cam_id))
        cam_to_window[cam_id] = window_id
        plane = pin.plane
        window_planes[window_id] = {
            "plane_pt": np.asarray([plane.pt[0], plane.pt[1], plane.pt[2]], dtype=np.float64),
            "plane_n": np.asarray(
                [plane.norm_vector[0], plane.norm_vector[1], plane.norm_vector[2]],
                dtype=np.float64,
            ),
        }

    optimizer = RefractionVSCOptimizer()
    optimizer._setup_problem(models, states, correspondences, cam_to_window, window_planes)
    packed = optimizer._pack_params(states)
    optimizer._apply_params(packed)
    return lpt, optimizer, models, sizes, packed


def _evaluate(camera_files, correspondences):
    lpt, optimizer, models, sizes, packed = _load_problem(camera_files, correspondences)
    all_camera = optimizer._compute_metrics(packed)
    rays_by_layout = optimizer._compute_rays_batch(lpt)
    heldout, heldout_ray = defaultdict(list), defaultdict(list)
    coverage, failures = defaultdict(Counter), Counter()

    for layout_index, (corr, cam_ids, _) in enumerate(optimizer._layouts):
        rays_by_cam = rays_by_layout[layout_index]
        observations = corr["2d_per_cam"]
        for holdout in cam_ids:
            training = [ray for cam, ray in rays_by_cam.items() if cam != holdout]
            if len(training) < 2 or holdout not in rays_by_cam:
                failures[holdout] += 1
                continue
            xyz = optimizer._triangulate_from_lines(training)
            if xyz is None:
                failures[holdout] += 1
                continue
            ok, predicted, _ = models[holdout].projectStatus(_make_point3d(lpt, xyz), False)
            if not ok:
                failures[holdout] += 1
                continue
            observed = observations[holdout]
            heldout[holdout].append(float(np.hypot(
                predicted[0] - observed[0], predicted[1] - observed[1]
            )))
            origin, direction = rays_by_cam[holdout]
            direction = direction / np.linalg.norm(direction)
            delta = xyz - origin
            perpendicular = delta - np.dot(delta, direction) * direction
            heldout_ray[holdout].append(float(np.linalg.norm(perpendicular)))
            nrow, ncol = map(float, sizes[holdout])
            bx = min(3, max(0, int(4.0 * observed[0] / ncol)))
            by = min(3, max(0, int(4.0 * observed[1] / nrow)))
            coverage[holdout][f"{bx},{by}"] += 1

    pooled = [value for values in heldout.values() for value in values]
    pooled_ray = [value for values in heldout_ray.values() for value in values]
    return {
        "all_camera_fit": all_camera,
        "leave_one_camera_out_px": {
            "pooled": _summary(pooled),
            "per_camera": {str(cam): _summary(heldout[cam]) for cam in sorted(models)},
        },
        "leave_one_camera_out_ray_mm": {
            "pooled": _summary(pooled_ray),
            "per_camera": {str(cam): _summary(heldout_ray[cam]) for cam in sorted(models)},
        },
        "spatial_coverage_4x4": {
            str(cam): {"occupied_bins": len(coverage[cam]), "total_bins": 16,
                       "counts": dict(sorted(coverage[cam].items()))}
            for cam in sorted(models)
        },
        "failures": {str(cam): int(failures[cam]) for cam in sorted(models)},
    }


class VSCValidationService:
    def __init__(self, source_project, validation_project, log_callback=None):
        self.source_project = os.path.abspath(source_project)
        self.validation_project = os.path.abspath(validation_project)
        self.log_callback = log_callback

    def _log(self, message):
        if self.log_callback:
            self.log_callback(str(message))
        print(message)

    @staticmethod
    def _camera_files(folder, prefix):
        paths = [
            os.path.join(folder, name) for name in os.listdir(folder)
            if name.lower().startswith(prefix) and name.lower().endswith(".txt")
        ] if os.path.isdir(folder) else []
        paths.sort(key=_natural_index)
        if len(paths) < 2:
            raise RuntimeError(f"Not enough camera files in {folder}")
        return paths

    @staticmethod
    def _create_legacy_provenance(source_project, vsc_files):
        """Bind an untagged legacy VSC without inventing its fitting range."""
        vsc_dir = os.path.join(source_project, "camFile_VSC")
        camera_hashes = {
            os.path.basename(path): _sha256(path) for path in vsc_files
        }
        frame_count, frame_count_source = vsc_source_frame_count(source_project)
        manifest = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "provenance_kind": "legacy_unresolved_fit_range",
            "source_project_path": os.path.abspath(source_project),
            "source_trial": os.path.basename(os.path.normpath(source_project)),
            "source_label": os.path.basename(os.path.normpath(source_project)),
            "requested_frame_range": {"start": None, "end": None},
            "actual_correspondence_frame_range": {
                "start": None,
                "end": None,
                "unique_frames": frame_count,
            },
            "legacy_frame_count_source": frame_count_source,
            "source_dataset_fingerprint": compute_dataset_fingerprint(source_project),
            "source_frame_range_fingerprint": {
                "available": False,
                "reason": "Legacy VSC did not record its contributing frame IDs.",
            },
            "vsc_camera_sha256": camera_hashes,
        }
        path = os.path.join(vsc_dir, "vsc_provenance.json")
        temp = path + ".tmp"
        with open(temp, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temp, path)
        return manifest

    def run(self, frame_count=20, max_points_per_frame=200,
            report_path=None, require_provenance=True,
            allow_current_data_self_check=False, random_sampling=False):
        vsc_dir = os.path.join(self.source_project, "camFile_VSC")
        provenance_path = os.path.join(vsc_dir, "vsc_provenance.json")
        vsc_files = self._camera_files(vsc_dir, "vsc_cam")
        provenance = None
        if os.path.isfile(provenance_path):
            with open(provenance_path, "r", encoding="utf-8") as stream:
                provenance = json.load(stream)
        elif allow_current_data_self_check:
            provenance = self._create_legacy_provenance(
                self.source_project, vsc_files
            )

        original_files = self._camera_files(
            os.path.join(self.validation_project, "camFile"), "cam"
        )
        if len(vsc_files) != len(original_files):
            raise RuntimeError("Source VSC and validation trial camera counts differ")
        camera_hashes = {os.path.basename(path): _sha256(path) for path in vsc_files}

        if require_provenance and provenance is None:
            raise RuntimeError("VSC provenance tag is missing")
        scope = assess_validation_scope(self.source_project, self.validation_project)
        validation_mode = (
            "different_acquisition" if not scope.get("same_trial", True)
            else "held_out_frames" if scope["can_validate_by_range"]
            else "current_data_self_check"
        )
        if (not scope["can_validate_by_range"]
                and not allow_current_data_self_check):
            raise RuntimeError(scope["reason"])
        if validation_mode == "current_data_self_check":
            scope = dict(scope)
            scope["eligible_frame_ranges"] = [
                _read_frame_range(self.validation_project)
            ]
        if provenance is not None:
            expected = provenance.get("vsc_camera_sha256", {})
            if not expected or expected != camera_hashes:
                raise RuntimeError(
                    "VSC camera files do not match their provenance tag"
                )

        random_seed = None
        if random_sampling:
            seed_material = "|".join(
                camera_hashes[name] for name in sorted(camera_hashes)
            )
            random_seed = int(
                hashlib.sha256(seed_material.encode("ascii")).hexdigest()[:16], 16
            )

        parsed = parse_camera_file(original_files[0])
        image_size = tuple(parsed["img_size"])
        self._log(
            f"External VSC validation: {os.path.basename(self.source_project)} -> "
            f"{os.path.basename(self.validation_project)}"
        )
        self._log(
            "  VSC fitting range: "
            + (f"{scope['vsc_frame_range'][0]}..{scope['vsc_frame_range'][1]}"
               if scope["vsc_frame_range"] else "unknown")
        )
        self._log(
            "  Eligible validation range(s): "
            + ", ".join(
                f"{start}..{end}" for start, end in scope["eligible_frame_ranges"]
            )
        )
        self._log(
            "  Dataset relationship: "
            + ("same acquisition; current-data self-check (not independent)"
               if validation_mode == "current_data_self_check"
               else "same acquisition; held-out frames"
               if scope["same_trial"] else "different acquisition")
            + f" ({scope['identity_basis']})"
        )
        candidates, frame_ids, source_files = _sample_track_candidates(
            self.validation_project, int(frame_count), int(max_points_per_frame),
            len(vsc_files), scope["eligible_frame_ranges"], random_seed
        )
        if not frame_ids:
            eligible_text = ", ".join(
                f"{start}..{end}" for start, end in scope["eligible_frame_ranges"]
            )
            raise RuntimeError(
                "No original-camera track observations were found in the eligible "
                f"validation range(s): {eligible_text}. Run those held-out frames "
                "with the original cameras before validating the VSC."
            )
        observations, detection = _redetect(
            self.validation_project, candidates, len(vsc_files), image_size, self._log
        )
        if len(observations) < 10:
            raise RuntimeError(
                f"Only {len(observations)} clean image observations survived validation detection"
            )
        self._log(f"  Scoring original cameras on {len(observations)} observations...")
        original = _evaluate(original_files, observations)
        self._log(f"  Scoring VSC cameras on {len(observations)} observations...")
        vsc = _evaluate(vsc_files, observations)

        checks = {}
        checks["provenance_present"] = {
            "required": bool(require_provenance),
            "passed": provenance is not None or not require_provenance,
        }
        source_path = (self.source_project if provenance is None and not require_provenance
                       else None if provenance is None
                       else provenance.get("source_project_path"))
        frames_are_held_out = bool(frame_ids) and all(
            _frame_in_ranges(frame_id, scope["eligible_frame_ranges"])
            for frame_id in frame_ids
        )
        checks["independent_validation_data"] = {
            "required": validation_mode != "current_data_self_check",
            "passed": (
                frames_are_held_out and validation_mode != "current_data_self_check"
            ),
            "source_project_path": source_path,
            "validation_project_path": self.validation_project,
            "same_trial": scope["same_trial"],
            "vsc_frame_range": scope["vsc_frame_range"],
            "eligible_frame_ranges": scope["eligible_frame_ranges"],
            "frames_used": frame_ids,
        }
        expected_hashes = {} if provenance is None else provenance.get("vsc_camera_sha256", {})
        hash_match = ((not require_provenance and provenance is None)
                      or (bool(expected_hashes) and expected_hashes == camera_hashes))
        checks["camera_hashes_match_provenance"] = {
            "required": True,
            "passed": hash_match,
        }
        checks["minimum_clean_observations"] = {
            "required": True,
            "passed": len(observations) >= 50,
            "value": len(observations),
            "minimum": 50,
        }
        coverage_min = min(
            entry["occupied_bins"] for entry in vsc["spatial_coverage_4x4"].values()
        )
        checks["spatial_coverage"] = {
            "required": True,
            "passed": coverage_min >= 12,
            "minimum_occupied_bins_per_camera": coverage_min,
            "required_bins": 12,
            "total_bins": 16,
        }
        orig_pool = original["leave_one_camera_out_px"]["pooled"]
        vsc_pool = vsc["leave_one_camera_out_px"]["pooled"]
        rmse_improves = (
            orig_pool["rmse"] is not None
            and vsc_pool["rmse"] is not None
            and vsc_pool["rmse"] < orig_pool["rmse"]
        )
        checks["pooled_rmse_improves"] = {
            "required": True,
            "passed": rmse_improves,
            "original": orig_pool["rmse"], "vsc": vsc_pool["rmse"],
        }
        p95_improves = (
            orig_pool["p95"] is not None
            and vsc_pool["p95"] is not None
            and vsc_pool["p95"] < orig_pool["p95"]
        )
        checks["pooled_p95_improves"] = {
            "required": True,
            "passed": p95_improves,
            "original": orig_pool["p95"], "vsc": vsc_pool["p95"],
        }
        per_cam_ratios = {}
        all_camera_ratios_available = True
        for cam, vsc_stats in vsc["leave_one_camera_out_px"]["per_camera"].items():
            original_p95 = original["leave_one_camera_out_px"]["per_camera"][cam]["p95"]
            vsc_p95 = vsc_stats["p95"]
            if original_p95 is None or vsc_p95 is None:
                per_cam_ratios[cam] = None
                all_camera_ratios_available = False
            elif original_p95 == 0.0:
                per_cam_ratios[cam] = 1.0 if vsc_p95 == 0.0 else None
                all_camera_ratios_available = (
                    all_camera_ratios_available and vsc_p95 == 0.0
                )
            else:
                per_cam_ratios[cam] = vsc_p95 / original_p95
        finite_ratios = [
            value for value in per_cam_ratios.values() if value is not None
        ]
        worst_ratio = max(finite_ratios) if finite_ratios else None
        checks["no_camera_tail_regression"] = {
            "required": True,
            "passed": (
                all_camera_ratios_available
                and worst_ratio is not None
                and worst_ratio <= 1.10
            ),
            "worst_vsc_to_original_p95_ratio": worst_ratio,
            "maximum_ratio": 1.10,
            "per_camera_ratios": per_cam_ratios,
        }
        checks["valid_refraction_geometry"] = {
            "required": True,
            "passed": int(vsc["all_camera_fit"]["barrier_violations"]) == 0,
            "barrier_violations": int(vsc["all_camera_fit"]["barrier_violations"]),
        }
        passed = all(item["passed"] for item in checks.values() if item["required"])
        report = {
            "schema_version": 2,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "passed": bool(passed),
            "validation_mode": validation_mode,
            "source_project_path": self.source_project,
            "validation_project_path": self.validation_project,
            "vsc_camera_sha256": camera_hashes,
            "vsc_provenance_sha256": (
                _sha256(provenance_path) if os.path.isfile(provenance_path) else None
            ),
            "source_provenance": provenance,
            "sample": {
                "frames_requested": int(frame_count),
                "frames_used": frame_ids,
                "sampling": (
                    "reproducible_random" if random_sampling else "distributed"
                ),
                "random_seed": random_seed,
                "vsc_frame_range": scope["vsc_frame_range"],
                "eligible_frame_ranges": scope["eligible_frame_ranges"],
                "source_track_files": source_files,
                "image_redetection": detection,
            },
            "checks": checks,
            "metrics": {"validation_original": original, "source_vsc": vsc},
        }
        if report_path:
            report_dir = os.path.dirname(report_path)
            if report_dir:
                os.makedirs(report_dir, exist_ok=True)
            temp = report_path + ".tmp"
            with open(temp, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(report, stream, indent=2, sort_keys=True)
                stream.write("\n")
            os.replace(temp, report_path)
        status = "PASSED" if passed else "FAILED"
        check_name = (
            "VSC current-data self-check"
            if validation_mode == "current_data_self_check" else "VSC validation"
        )
        def metric_text(value):
            return "unavailable" if value is None else f"{value:.4f}"

        self._log(
            f"{check_name} {status}: RMSE {metric_text(orig_pool['rmse'])} -> "
            f"{metric_text(vsc_pool['rmse'])} px, P95 "
            f"{metric_text(orig_pool['p95'])} -> {metric_text(vsc_pool['p95'])} px"
        )
        return bool(passed), f"{check_name} {status}", report


def verify_saved_validation(project_dir):
    """Return (ok, message) for the hash-bound report used by LPT preflight."""
    vsc_dir = os.path.join(project_dir, "camFile_VSC")
    report_path = os.path.join(vsc_dir, "vsc_validation.json")
    if not os.path.isfile(report_path):
        return False, "No VSC quality report was found. Run the 30-frame current-data check first."
    try:
        with open(report_path, "r", encoding="utf-8") as stream:
            report = json.load(stream)
        if not report.get("passed", False):
            return False, "The saved VSC validation report did not pass."
        current = {
            os.path.basename(path): _sha256(path)
            for path in VSCValidationService._camera_files(vsc_dir, "vsc_cam")
        }
        if current != report.get("vsc_camera_sha256", {}):
            return False, "VSC camera files changed after validation; validate them again."
        provenance_path = os.path.join(vsc_dir, "vsc_provenance.json")
        expected_provenance = report.get("vsc_provenance_sha256")
        if (not expected_provenance or not os.path.isfile(provenance_path)
                or _sha256(provenance_path) != expected_provenance):
            return False, "VSC provenance changed after validation; validate it again."
        if report.get("validation_mode") == "current_data_self_check":
            return True, "VSC 30-frame current-data self-check passed (not independent)."
        return True, (
            "VSC validation passed against "
            f"{os.path.basename(report.get('validation_project_path', 'unknown'))}."
        )
    except Exception as exc:
        return False, f"Could not verify the VSC validation report: {exc}"


def saved_validation_audit_lines(project_dir):
    """Return concise, persistent audit lines for a production LPT log."""
    vsc_dir = os.path.join(project_dir, "camFile_VSC")
    report_path = os.path.join(vsc_dir, "vsc_validation.json")
    with open(report_path, "r", encoding="utf-8") as stream:
        report = json.load(stream)
    provenance = report.get("source_provenance", {}) or {}
    source_range = _provenance_frame_range(provenance)
    sample = report.get("sample", {})
    eligible = sample.get("eligible_frame_ranges", [])
    frames = sample.get("frames_used", [])
    independent = report.get("checks", {}).get(
        "independent_validation_data", {}
    )
    if report.get("validation_mode") == "current_data_self_check":
        relationship = "same acquisition, current-data self-check; not independent"
    else:
        relationship = (
            "same acquisition, non-overlapping held-out frames"
            if independent.get("same_trial") else "different acquisition"
        )
    source_label = (
        provenance.get("source_label") or provenance.get("source_trial")
        or os.path.basename(provenance.get("source_project_path", "")) or "unknown"
    )
    validation_label = os.path.basename(
        os.path.normpath(report.get("validation_project_path", "unknown"))
    )
    source_range_text = (
        f"{source_range[0]}..{source_range[1]}" if source_range else "unknown"
    )
    source_frame_count = provenance.get(
        "actual_correspondence_frame_range", {}
    ).get("unique_frames")
    eligible_text = ", ".join(
        f"{start}..{end}" for start, end in eligible
    ) or "unknown"
    frames_text = ", ".join(str(frame) for frame in frames) or "none"
    digest = (
        provenance.get("source_frame_range_fingerprint", {}).get("digest")
        or provenance.get("source_dataset_fingerprint", {}).get("digest")
    )
    range_label = (
        "Self-check sampling range" if report.get("validation_mode")
        == "current_data_self_check" else "Eligible validation frames"
    )
    return [
        f"VSC source label: {source_label}",
        "VSC contributing-frame count: "
        + (f"{int(source_frame_count):,}" if source_frame_count is not None else "unknown"),
        f"VSC fitting frames: {source_range_text}",
        f"Validation dataset: {validation_label} ({relationship})",
        f"{range_label}: {eligible_text}",
        f"Validation frames actually used ({len(frames)}): {frames_text}",
        f"Validation sampling: {sample.get('sampling', 'unknown')}",
        "VSC data fingerprint: " + (digest[:16] + "..." if digest else "unavailable"),
        f"VSC validation report: {report_path}",
    ]
