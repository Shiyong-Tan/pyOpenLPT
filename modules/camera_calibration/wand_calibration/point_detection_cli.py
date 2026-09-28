"""
HZ_fix: Command-line equivalent of the Wand Calibration "Process All Frames" button.

The GUI's "Generate CLI" button writes a single-line command into
``Point_Detection_CLI.txt`` (in the T0 folder). Copy that line into a terminal
and it runs **exactly** what "Process All Frames / Resume" does — the same
detection (`WandCalibrator.detect_wand_points_generator`), over all frames in the
``cam<N>`` subfolders of T0, with the detection settings chosen in the UI, and
writes the same CSV (``--output``). The only difference is that it runs headless,
without the GUI.

This module is a thin driver: it builds the per-camera image dict the way the GUI
loads it (each ``cam<N>`` folder's images, sorted), then consumes the generator
to completion (which autosaves the CSV with Raw + Filtered data at the end).

Run (one line, as written into ``Point_Detection_CLI.txt``)::

    python "<repo>/modules/camera_calibration/wand_calibration/point_detection_cli.py" \\
        --t0 "<T0>" --wand-type dark --min-radius 20 --max-radius 200 \\
        --sensitivity 0.850 --detect-mode fast \\
        --output "<T0>/wand_points.csv"
"""

import sys

import re
import argparse
from pathlib import Path

try:
    # Normal package import (python -m ...).
    from .wand_calibrator import WandCalibrator
except ImportError:  # pragma: no cover - fallback when run as a loose script.
    _REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from modules.camera_calibration.wand_calibration.wand_calibrator import WandCalibrator


IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
CAM_RE = re.compile(r"^cam(\d+)$", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Folder / frame discovery (mirrors the GUI's Root-folder loader)             #
# --------------------------------------------------------------------------- #
def discover_cam_folders(t0_dir):
    """Return [(cam_number, folder), ...] for every ``cam<N>`` subfolder of T0,
    sorted ascending by N — the same order the GUI assigns to camera slots."""
    p = Path(t0_dir)
    cams = []
    if not p.is_dir():
        return cams
    for child in p.iterdir():
        if child.is_dir():
            m = CAM_RE.match(child.name)
            if m:
                cams.append((int(m.group(1)), str(child)))
    cams.sort(key=lambda x: x[0])
    return cams


def list_cam_images(folder):
    """Sorted list of image file paths in a camera folder."""
    p = Path(folder)
    return sorted(str(f) for f in p.iterdir()
                  if f.is_file() and f.suffix.lower() in IMAGE_EXTS)


def build_image_paths_dict(cam_folders):
    """Map slot index (0..N-1, ascending by cam number) -> sorted image paths.

    This matches how the GUI populates ``wand_images`` from the Root folder, so
    the "Camera" column in the output CSV matches the GUI.
    """
    image_paths = {}
    for slot, (_num, folder) in enumerate(cam_folders):
        files = list_cam_images(folder)
        if files:
            image_paths[slot] = files
    return image_paths


# --------------------------------------------------------------------------- #
# One-line command string (used by the GUI to write the .txt)                 #
# --------------------------------------------------------------------------- #
def build_cli_command(t0_dir, wand_type, min_radius, max_radius, sensitivity,
                      detect_mode, output_csv, python_exe=None):
    """Return the single-line command that reproduces 'Process All Frames'."""
    python_exe = python_exe or sys.executable or "python"
    script = Path(__file__).resolve()
    parts = [
        f'"{python_exe}"',
        f'"{script}"',
        f'--t0 "{t0_dir}"',
        f'--wand-type {wand_type}',
        f'--min-radius {min_radius}',
        f'--max-radius {max_radius}',
        f'--sensitivity {sensitivity:.3f}',
        f'--detect-mode {detect_mode}',
        f'--output "{output_csv}"',
    ]
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Driver: run the exact "Process All" generator to completion                 #
# --------------------------------------------------------------------------- #
def run_process_all(image_paths_dict, wand_type, min_radius, max_radius,
                    sensitivity, detect_mode="fast", output_csv=None,
                    resume=False, progress_cb=None):
    """Run `detect_wand_points_generator` to completion (writes the CSV via
    autosave). Returns the WandCalibrator instance."""
    calib = WandCalibrator()
    gen = calib.detect_wand_points_generator(
        image_paths_dict, wand_type, float(min_radius), float(max_radius),
        float(sensitivity), autosave_path=output_csv, resume=resume,
        detect_mode=detect_mode,
    )
    for current, total in gen:
        if progress_cb is not None:
            progress_cb(current, total)
    return calib


# --------------------------------------------------------------------------- #
# CLI entry point                                                             #
# --------------------------------------------------------------------------- #
def main(argv=None):
    parser = argparse.ArgumentParser(
        description="OpenLPT Wand Point Detection (CLI equivalent of the GUI "
                    "'Process All Frames' button)."
    )
    parser.add_argument("--t0", required=True,
                        help="T0 folder containing cam0, cam1, ... subfolders.")
    parser.add_argument("--wand-type", default="bright", choices=["dark", "bright"])
    parser.add_argument("--min-radius", type=float, default=20.0)
    parser.add_argument("--max-radius", type=float, default=200.0)
    parser.add_argument("--sensitivity", type=float, default=0.85)
    parser.add_argument("--detect-mode", default="fast", choices=["fast", "reliable"])
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: <t0>/wand_points.csv).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from an existing output CSV (like the GUI Resume).")
    args = parser.parse_args(argv)

    t0 = Path(args.t0)
    cam_folders = discover_cam_folders(t0)
    if not cam_folders:
        print(f"Error: no 'cam<N>' folders found in: {t0}")
        return 1

    image_paths = build_image_paths_dict(cam_folders)
    if not image_paths:
        print(f"Error: no images found in the camera folders under: {t0}")
        return 1

    out_csv = args.output or str(t0 / "wand_points.csv")
    n_frames = max(len(v) for v in image_paths.values())

    print(f"T0         : {t0}")
    print(f"Cameras    : {[n for n, _ in cam_folders]} -> slots {sorted(image_paths.keys())}")
    print(f"Frames     : up to {n_frames} per camera")
    print(f"Settings   : wand_type={args.wand_type}, radius=[{args.min_radius},"
          f"{args.max_radius}], sensitivity={args.sensitivity}, "
          f"detect_mode={args.detect_mode}, resume={args.resume}")
    print(f"Output CSV : {out_csv}")

    def _progress(cur, total):
        print(f"  processing... {cur}/{total} frames", end="\r", flush=True)

    run_process_all(
        image_paths, args.wand_type, args.min_radius, args.max_radius,
        args.sensitivity, args.detect_mode, output_csv=out_csv,
        resume=args.resume, progress_cb=_progress,
    )
    print()
    print(f"Done. Detection results written to: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
