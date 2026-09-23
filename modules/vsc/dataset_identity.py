"""Name-independent identity for an OpenLPT image acquisition."""

from __future__ import annotations

import hashlib
import json
import os
import re


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _natural_key(path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", os.path.basename(path))
    ]


def _resolve_image(project_dir, list_path, value):
    value = value.strip().strip('"')
    if os.path.isabs(value):
        return os.path.normpath(value)
    project_relative = os.path.normpath(os.path.join(project_dir, value))
    if os.path.isfile(project_relative):
        return project_relative
    return os.path.normpath(os.path.join(os.path.dirname(list_path), value))


def _sample_indices(count, sample_count):
    sample_count = max(1, int(sample_count))
    if count <= sample_count:
        return list(range(count))
    if sample_count == 1:
        return [count // 2]
    return sorted({
        int(round(index * (count - 1) / float(sample_count - 1)))
        for index in range(sample_count)
    })


def compute_dataset_fingerprint(project_dir, samples_per_camera=5):
    """Fingerprint image content without relying on folder or trial names.

    Distributed raw-image samples and per-camera list lengths contribute to the
    digest. Paths and filenames are deliberately excluded, so moving or
    renaming a dataset does not change its identity.
    """
    project_dir = os.path.abspath(project_dir)
    image_list_dir = os.path.join(project_dir, "imgFile")
    lists = [
        os.path.join(image_list_dir, name)
        for name in os.listdir(image_list_dir)
        if name.lower().endswith("imagenames.txt")
    ] if os.path.isdir(image_list_dir) else []
    lists.sort(key=_natural_key)
    if not lists:
        return {
            "available": False,
            "algorithm": "distributed-image-sha256-v1",
            "reason": f"No image-name lists found in {image_list_dir}",
        }

    cameras = []
    try:
        for camera_index, list_path in enumerate(lists):
            with open(list_path, "r", encoding="utf-8-sig") as stream:
                entries = [line.strip() for line in stream if line.strip()]
            if not entries:
                raise RuntimeError(f"Image list is empty: {list_path}")
            samples = []
            for frame_index in _sample_indices(len(entries), samples_per_camera):
                image_path = _resolve_image(project_dir, list_path, entries[frame_index])
                if not os.path.isfile(image_path):
                    raise RuntimeError(f"Raw image not found: {image_path}")
                samples.append({
                    "frame_index": frame_index,
                    "bytes": os.path.getsize(image_path),
                    "sha256": _sha256(image_path),
                })
            cameras.append({
                "camera_index": camera_index,
                "image_count": len(entries),
                "samples": samples,
            })
    except Exception as exc:
        return {
            "available": False,
            "algorithm": "distributed-image-sha256-v1",
            "reason": str(exc),
        }

    canonical = json.dumps(cameras, sort_keys=True, separators=(",", ":"))
    return {
        "available": True,
        "algorithm": "distributed-image-sha256-v1",
        "camera_count": len(cameras),
        "samples_per_camera": int(samples_per_camera),
        "digest": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "cameras": cameras,
    }


def compute_frame_range_fingerprint(project_dir, frame_range,
                                    samples_per_camera=5):
    """Fingerprint fixed list positions from an inclusive frame interval."""
    project_dir = os.path.abspath(project_dir)
    image_list_dir = os.path.join(project_dir, "imgFile")
    lists = [
        os.path.join(image_list_dir, name)
        for name in os.listdir(image_list_dir)
        if name.lower().endswith("imagenames.txt")
    ] if os.path.isdir(image_list_dir) else []
    lists.sort(key=_natural_key)
    start, end = map(int, frame_range)
    if not lists or start < 0 or end < start:
        return {
            "available": False,
            "algorithm": "frame-range-image-sha256-v1",
            "reason": "Invalid frame range or no image-name lists were found.",
        }

    cameras = []
    try:
        for camera_index, list_path in enumerate(lists):
            with open(list_path, "r", encoding="utf-8-sig") as stream:
                entries = [line.strip() for line in stream if line.strip()]
            if end >= len(entries):
                raise RuntimeError(
                    f"Frame range {start}..{end} exceeds {list_path} "
                    f"({len(entries)} images)"
                )
            relative_indices = _sample_indices(
                end - start + 1, samples_per_camera
            )
            samples = []
            for relative_index in relative_indices:
                frame_index = start + relative_index
                image_path = _resolve_image(project_dir, list_path, entries[frame_index])
                if not os.path.isfile(image_path):
                    raise RuntimeError(f"Raw image not found: {image_path}")
                samples.append({
                    "frame_index": frame_index,
                    "bytes": os.path.getsize(image_path),
                    "sha256": _sha256(image_path),
                })
            cameras.append({"camera_index": camera_index, "samples": samples})
    except Exception as exc:
        return {
            "available": False,
            "algorithm": "frame-range-image-sha256-v1",
            "reason": str(exc),
        }

    payload = {"frame_range": [start, end], "cameras": cameras}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {
        "available": True,
        "algorithm": "frame-range-image-sha256-v1",
        "camera_count": len(cameras),
        "samples_per_camera": int(samples_per_camera),
        "frame_range": [start, end],
        "digest": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "cameras": cameras,
    }
