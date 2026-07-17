"""
CLI entry point for OpenLPT image preprocessing.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .io import (
    DetectedRootInput,
    detect_input_root,
    ImagePreprocessingIOError,
    compute_mean_background_from_cine,
    compute_mean_background_from_tiff,
    plan_cine_tasks,
    plan_tiff_tasks,
)
from .runner import BatchProcessingResult, run_batch_processing


def _get_detected_input_root(args: argparse.Namespace) -> DetectedRootInput | None:
    detected = getattr(args, "_detected_input_root", None)
    if detected is None and args.input_root:
        detected = detect_input_root(args.input_root)
        args._detected_input_root = detected
    return detected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openlpt preprocess",
        description="Preprocess TIFF image lists, direct TIFF images, or CINE files into OpenLPT-ready TIFF outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  openlpt preprocess --input-root ./dataset --frames 0 99\n"
            "  openlpt preprocess --input-list cam0.txt --input-list cam1.txt --output-dir processed\n"
            "  openlpt preprocess --image cam0_0001.tif --image cam1_0001.tif --output-dir processed\n"
            "  openlpt preprocess --image img1.tif --image img2.tif --camera-index 0 --output-dir processed\n"
            "  openlpt preprocess --cine cam0.cine --cine cam1.cine --frames 0 99 --output-dir processed --background mean\n"
        ),
    )

    parser.add_argument("--input-root", help="Root directory containing camera inputs such as cam0/, cam1/, ...")
    parser.add_argument("--input-list", action="append", default=[], help="TIFF image list text file. Repeat once per camera.")
    parser.add_argument("--image", action="append", default=[], help="Direct TIFF image path. Repeat for additional images.")
    parser.add_argument("--cine", action="append", default=[], help="CINE file path. Repeat once per camera.")
    parser.add_argument("--output-dir", help="Directory where processed TIFFs and image lists will be written.")
    parser.add_argument(
        "--camera-index",
        type=int,
        help="0-based camera index for direct --image inputs when all images belong to one camera.",
    )
    parser.add_argument(
        "--frames",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Inclusive frame range for CINE input mode.",
    )
    parser.add_argument("--invert", action="store_true", help="Invert image intensities.")
    parser.add_argument(
        "--background",
        choices=("none", "mean"),
        default="none",
        help="Background subtraction mode.",
    )
    parser.add_argument("--bg-start", type=int, default=0, help="Background sampling start index/frame number.")
    parser.add_argument("--bg-count", type=int, default=None, help="Number of frames/images to sample for mean background.")
    parser.add_argument("--bg-stride", type=int, default=1, help="Stride between sampled frames/images for background.")
    parser.add_argument("--low-in", type=float, default=0, help="Lower input intensity for contrast adjustment.")
    parser.add_argument("--high-in", type=float, default=255, help="Upper input intensity for contrast adjustment.")
    parser.add_argument("--denoise", action="store_true", help="Enable denoise processing.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers. Use 0 for all available CPU cores (default=1 for sequential processing).",
    )
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    source_groups = [bool(args.input_root), bool(args.input_list), bool(args.image), bool(args.cine)]
    if sum(source_groups) != 1:
        parser.error("choose exactly one input mode: --input-root, --input-list, --image, or --cine")

    if args.input_root and args.camera_index is not None:
        parser.error("--camera-index is only supported with direct --image inputs")

    if args.camera_index is not None and args.camera_index < 0:
        parser.error("--camera-index must be >= 0")

    if args.input_list and args.camera_index is not None:
        parser.error("--camera-index is only supported with direct --image inputs")

    if args.cine and args.frames is None:
        parser.error("--frames START END is required with --cine")
    if args.frames is not None and not (args.cine or args.input_root):
        parser.error("--frames is only supported with --cine or --input-root")

    if args.input_root:
        try:
            detected = _get_detected_input_root(args)
        except ImagePreprocessingIOError as exc:
            parser.error(str(exc))
        if detected is None:
            parser.error("--input-root detection is unavailable")
        if detected.input_kind == "cine" and args.frames is None:
            parser.error("--frames START END is required with CINE inputs")
        if detected.input_kind != "cine" and args.frames is not None:
            parser.error("--frames is only supported with CINE inputs")

    if args.bg_stride <= 0:
        parser.error("--bg-stride must be >= 1")

    if args.bg_count is not None and args.bg_count <= 0:
        parser.error("--bg-count must be >= 1 when provided")

    if args.high_in < args.low_in:
        parser.error("--high-in must be >= --low-in")

    if args.workers < 0:
        parser.error("--workers must be >= 0")


def _strip_wrapping_quotes(value: str) -> str:
    text = str(value).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    return text


def _normalize_path_args(args: argparse.Namespace) -> None:
    args.input_root = _strip_wrapping_quotes(args.input_root) if args.input_root is not None else None
    args.input_list = [_strip_wrapping_quotes(path) for path in args.input_list]
    args.image = [_strip_wrapping_quotes(path) for path in args.image]
    args.cine = [_strip_wrapping_quotes(path) for path in args.cine]
    args.output_dir = _strip_wrapping_quotes(args.output_dir) if args.output_dir is not None else None


def _build_tiff_tasks_from_lists(input_lists: Sequence[str], output_dir: str | Path):
    tasks = []
    for cam_idx, input_list in enumerate(input_lists, start=0):
        tasks.extend(plan_tiff_tasks(cam_idx=cam_idx, output_dir=output_dir, image_list_files=[input_list]))
    return tasks


def _build_tiff_tasks_from_images(images: Sequence[str], output_dir: str | Path, camera_index: int | None):
    tasks = []
    if camera_index is not None:
        tasks.extend(plan_tiff_tasks(cam_idx=camera_index, output_dir=output_dir, image_paths=list(images)))
        return tasks

    for cam_idx, image_path in enumerate(images, start=0):
        tasks.extend(plan_tiff_tasks(cam_idx=cam_idx, output_dir=output_dir, image_paths=[image_path]))
    return tasks


def _build_cine_tasks(cines: Sequence[str], output_dir: str | Path, frames: Sequence[int]):
    start_frame, end_frame = frames
    tasks = []
    for cam_idx, cine_path in enumerate(cines, start=0):
        tasks.extend(
            plan_cine_tasks(
                cam_idx=cam_idx,
                cine_path=cine_path,
                output_dir=output_dir,
                start_frame=start_frame,
                end_frame=end_frame,
            )
        )
    return tasks


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()
    if args.input_root:
        return Path(args.input_root).expanduser().resolve() / "imgFile"
    raise ImagePreprocessingIOError("--output-dir is required unless --input-root is used")


def _build_tasks_from_input_root(args: argparse.Namespace, output_dir: Path):
    detected = _get_detected_input_root(args)
    if detected is None:
        raise ImagePreprocessingIOError("--input-root detection is unavailable")
    tasks = []
    if detected.input_kind == "cine":
        if args.frames is None:
            raise ImagePreprocessingIOError("--frames START END is required with CINE inputs")
        start_frame, end_frame = args.frames
        for cam_idx, camera in enumerate(detected.cameras, start=0):
            if camera.cine_path is None:
                raise ImagePreprocessingIOError(f"Detected CINE camera has no CINE path: {camera.camera_name}")
            tasks.extend(
                plan_cine_tasks(
                    cam_idx=cam_idx,
                    cine_path=camera.cine_path,
                    output_dir=output_dir,
                    start_frame=start_frame,
                    end_frame=end_frame,
                )
            )
        return tasks, detected

    if args.frames is not None:
        parser_message = "--frames is only supported with CINE inputs"
        raise ImagePreprocessingIOError(parser_message)

    for cam_idx, camera in enumerate(detected.cameras, start=0):
        tasks.extend(plan_tiff_tasks(cam_idx=cam_idx, output_dir=output_dir, image_paths=list(camera.image_paths)))
    return tasks, detected


def _build_backgrounds(args: argparse.Namespace) -> dict[int, object]:
    if args.background != "mean":
        return {}

    backgrounds: dict[int, object] = {}
    common_kwargs = {
        "start": args.bg_start,
        "count": args.bg_count,
        "stride": args.bg_stride,
    }

    if args.input_root:
        detected = _get_detected_input_root(args)
        if detected is None:
            raise ImagePreprocessingIOError("--input-root detection is unavailable")
        if detected.input_kind == "cine":
            for cam_idx, camera in enumerate(detected.cameras, start=0):
                if camera.cine_path is None:
                    raise ImagePreprocessingIOError(f"Detected CINE camera has no CINE path: {camera.camera_name}")
                backgrounds[cam_idx] = compute_mean_background_from_cine(cine_path=camera.cine_path, **common_kwargs)
            return backgrounds

        for cam_idx, camera in enumerate(detected.cameras, start=0):
            backgrounds[cam_idx] = compute_mean_background_from_tiff(image_paths=list(camera.image_paths), **common_kwargs)
        return backgrounds

    if args.input_list:
        for cam_idx, input_list in enumerate(args.input_list, start=0):
            backgrounds[cam_idx] = compute_mean_background_from_tiff(image_list_files=[input_list], **common_kwargs)
        return backgrounds

    if args.image:
        if args.camera_index is not None:
            backgrounds[args.camera_index] = compute_mean_background_from_tiff(image_paths=args.image, **common_kwargs)
            return backgrounds

        for cam_idx, image_path in enumerate(args.image, start=0):
            backgrounds[cam_idx] = compute_mean_background_from_tiff(image_paths=[image_path], **common_kwargs)
        return backgrounds

    for cam_idx, cine_path in enumerate(args.cine, start=0):
        backgrounds[cam_idx] = compute_mean_background_from_cine(cine_path=cine_path, **common_kwargs)
    return backgrounds


def _build_settings(args: argparse.Namespace) -> dict[str, object]:
    return {
        "bg_enabled": args.background == "mean",
        "invert": args.invert,
        "low_in": args.low_in,
        "high_in": args.high_in,
        "denoise": args.denoise,
    }


def _print_summary(result: BatchProcessingResult) -> None:
    print(
        f"Preprocessing complete: {result.processed_count} processed, "
        f"{result.failed_count} failed, output={result.output_dir}"
    )
    for cam_idx in sorted(result.image_list_files):
        print(f"  cam{cam_idx}: {result.image_list_files[cam_idx]}")
    if result.failures:
        preview = result.failures[:3]
        for failure in preview:
            print(f"  failure: {failure}")
        if len(result.failures) > len(preview):
            print(f"  ... {len(result.failures) - len(preview)} more failure(s)")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    _normalize_path_args(args)
    _validate_args(args, parser)

    try:
        output_dir = _resolve_output_dir(args)
        if args.input_root:
            tasks, detected = _build_tasks_from_input_root(args, output_dir)
            if detected.input_kind != "cine" and args.frames is not None:
                raise ImagePreprocessingIOError("--frames is only supported with CINE inputs")
        elif args.input_list:
            tasks = _build_tiff_tasks_from_lists(args.input_list, output_dir)
        elif args.image:
            tasks = _build_tiff_tasks_from_images(args.image, output_dir, args.camera_index)
        else:
            tasks = _build_cine_tasks(args.cine, output_dir, args.frames)

        backgrounds = _build_backgrounds(args)
        result = run_batch_processing(
            tasks,
            output_dir=output_dir,
            settings=_build_settings(args),
            backgrounds=backgrounds,
            workers=args.workers,
        )
    except (ImagePreprocessingIOError, ValueError, OSError) as exc:
        print(f"Error: {exc}")
        return 1

    _print_summary(result)
    return 0 if result.failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
