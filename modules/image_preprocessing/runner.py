# pyright: reportMissingImports=false, reportAttributeAccessIssue=false
"""
GUI-independent batch runner for image preprocessing.
"""

from __future__ import annotations

import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import cv2
import numpy as np

from .core import apply_processing_pipeline_with_settings, normalize_processing_settings
from .io import PlannedFrameTask, load_task_image, write_image_list_file


ProgressCallback = Callable[[str, int, int, str], None]


@dataclass
class BatchProcessingResult:
    """Summary of one batch preprocessing run."""

    output_dir: Path
    processed_count: int
    failed_count: int
    image_list_files: dict[int, Path]
    image_relpaths_by_camera: dict[int, list[str]]
    failures: list[str] = field(default_factory=list)


def _emit_progress(callback: ProgressCallback | None, stage: str, current: int, total: int, message: str) -> None:
    if callback is not None:
        callback(stage, current, total, message)


def _ensure_output_dirs(tasks: Sequence[PlannedFrameTask]) -> None:
    for task in tasks:
        task.output_path.parent.mkdir(parents=True, exist_ok=True)


def _process_single_task(
    task: PlannedFrameTask,
    task_index: int,
    normalized_settings: dict,
    backgrounds: dict[int, np.ndarray],
) -> tuple[int, PlannedFrameTask, str | None]:
    """
    Process a single task and return (task_index, task, error_message).
    Returns error_message=None on success.
    """
    try:
        raw_img = load_task_image(task)
        bg = backgrounds.get(task.cam_idx)
        processed_img = apply_processing_pipeline_with_settings(raw_img, bg, task.cam_idx, normalized_settings)

        ok = cv2.imwrite(str(task.output_path), processed_img)
        if not ok:
            raise RuntimeError(f"Failed to write processed image: {task.output_path}")

        return (task_index, task, None)
    except Exception as exc:
        return (task_index, task, str(exc))


def run_batch_processing(
    tasks: Sequence[PlannedFrameTask],
    *,
    output_dir: str | Path,
    settings: Mapping,
    backgrounds: Mapping[int, np.ndarray] | None = None,
    progress_callback: ProgressCallback | None = None,
    continue_on_error: bool = True,
    workers: int = 1,
) -> BatchProcessingResult:
    """
    Run preprocessing for planned TIFF/CINE tasks and write per-camera image lists.

    Args:
        tasks: Sequence of preprocessing tasks to execute.
        output_dir: Directory where processed TIFFs and image lists will be written.
        settings: Processing settings (background, invert, denoise, etc.).
        backgrounds: Optional per-camera background images for subtraction.
        progress_callback: Optional callback for progress updates.
        continue_on_error: If True, collect failures and continue; if False, raise on first failure.
        workers: Number of parallel workers (default=1 for sequential). Use 0 to use all available CPU cores.

    Returns:
        BatchProcessingResult with processed counts, image lists, and failures.
    """
    normalized_settings = normalize_processing_settings(dict(settings))
    output_dir = Path(output_dir).expanduser().resolve()
    tasks = list(tasks)
    backgrounds = dict(backgrounds or {})

    if workers < 0:
        raise ValueError("workers must be >= 0")
    if workers == 0:
        workers = max(1, int(os.cpu_count() or 1))

    if not tasks:
        return BatchProcessingResult(
            output_dir=output_dir,
            processed_count=0,
            failed_count=0,
            image_list_files={},
            image_relpaths_by_camera={},
            failures=[],
        )

    _ensure_output_dirs(tasks)

    image_relpaths_by_camera: dict[int, list[str]] = defaultdict(list)
    failures: list[str] = []
    processed_count = 0
    failed_count = 0
    total = len(tasks)

    # Track results by original task index to preserve order
    results_by_index: dict[int, tuple[PlannedFrameTask, str | None]] = {}

    if workers <= 1:
        # Sequential processing (original behavior)
        for current, task in enumerate(tasks, start=1):
            task_index, task, error = _process_single_task(task, current - 1, normalized_settings, backgrounds)
            results_by_index[task_index] = (task, error)

            if error is None:
                processed_count += 1
                _emit_progress(progress_callback, "process", current, total, f"Processed {current}/{total}: {task.output_relpath}")
            else:
                failed_count += 1
                failures.append(f"{task.output_relpath}: {error}")
                _emit_progress(progress_callback, "process", current, total, f"Failed {current}/{total}: {task.output_relpath}")
                if not continue_on_error:
                    raise RuntimeError(f"{task.output_relpath}: {error}")
    else:
        # Parallel processing with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(_process_single_task, task, idx, normalized_settings, backgrounds): idx
                for idx, task in enumerate(tasks)
            }

            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_index):
                task_index, task, error = future.result()
                results_by_index[task_index] = (task, error)
                completed += 1

                if error is None:
                    processed_count += 1
                    _emit_progress(progress_callback, "process", completed, total, f"Processed {completed}/{total}: {task.output_relpath}")
                else:
                    failed_count += 1
                    failures.append(f"{task.output_relpath}: {error}")
                    _emit_progress(progress_callback, "process", completed, total, f"Failed {completed}/{total}: {task.output_relpath}")
                    if not continue_on_error:
                        # Cancel remaining tasks
                        for f in future_to_index:
                            f.cancel()
                        raise RuntimeError(f"{task.output_relpath}: {error}")

    # Reconstruct image lists in original task order
    for task_index in sorted(results_by_index.keys()):
        task, error = results_by_index[task_index]
        if error is None:
            image_relpaths_by_camera[task.cam_idx].append(task.output_relpath)

    image_list_files: dict[int, Path] = {}
    cameras = sorted(image_relpaths_by_camera)
    list_total = len(cameras)
    for current, cam_idx in enumerate(cameras, start=1):
        image_list_path = output_dir / f"cam{cam_idx}ImageNames.txt"
        image_list_files[cam_idx] = write_image_list_file(image_list_path, image_relpaths_by_camera[cam_idx])
        _emit_progress(progress_callback, "write_lists", current, list_total, f"Wrote image list for cam{cam_idx}")

    return BatchProcessingResult(
        output_dir=output_dir,
        processed_count=processed_count,
        failed_count=failed_count,
        image_list_files=image_list_files,
        image_relpaths_by_camera={cam_idx: list(paths) for cam_idx, paths in image_relpaths_by_camera.items()},
        failures=failures,
    )
