# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false
import builtins
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from modules.image_preprocessing.runner import run_batch_processing


def test_read_image_list_file_resolves_relative_entries_against_list_location(tmp_path):
    from modules.image_preprocessing import io

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    rel_image = images_dir / "cam1_a.tif"
    rel_image.write_bytes(b"fake")
    abs_image = tmp_path / "cam1_b.tif"
    abs_image.write_bytes(b"fake")

    image_list = tmp_path / "cam1_list.txt"
    image_list.write_text(f"images/cam1_a.tif\n{abs_image}\n\n", encoding="utf-8")

    result = io.read_image_list_file(image_list)

    assert result == [rel_image.resolve(), abs_image.resolve()]


def test_resolve_tiff_input_paths_uses_base_dir_for_direct_paths(tmp_path):
    from modules.image_preprocessing import io

    base_dir = tmp_path / "inputs"
    base_dir.mkdir()
    image_path = base_dir / "frame_0001.tif"
    image_path.write_bytes(b"fake")

    result = io.resolve_tiff_input_paths(image_paths=["frame_0001.tif"], base_dir=base_dir)

    assert result == [image_path.resolve()]


def test_plan_tiff_tasks_uses_source_naming_and_disambiguates_duplicates(tmp_path):
    from modules.image_preprocessing import io

    camera_a = tmp_path / "cam_a"
    camera_b = tmp_path / "cam_b"
    camera_a.mkdir()
    camera_b.mkdir()
    first = camera_a / "frame01.tif"
    second = camera_b / "frame01.tif"
    first.write_bytes(b"a")
    second.write_bytes(b"b")

    tasks = io.plan_tiff_tasks(
        cam_idx=2,
        output_dir=tmp_path / "processed",
        image_paths=[first, second],
        naming="source",
        start_index=10,
    )

    assert [task.output_path.name for task in tasks] == ["frame01.tif", "frame01_000011.tif"]
    assert [task.output_relpath for task in tasks] == [
        "cam2/frame01.tif",
        "cam2/frame01_000011.tif",
    ]


def test_detect_input_root_prefers_natural_sorted_cine_camera_subdirs(tmp_path):
    from modules.image_preprocessing import io

    root = tmp_path / "dataset"
    for name in ["Cam10", "cam2", "CAM1"]:
        folder = root / name
        folder.mkdir(parents=True)
        (folder / f"{name}.cine").write_bytes(b"fake")

    detected = io.detect_input_root(root)

    assert detected.root_dir == root.resolve()
    assert detected.input_kind == "cine"
    assert [camera.camera_name for camera in detected.cameras] == ["CAM1", "cam2", "Cam10"]
    assert [camera.cine_path.name for camera in detected.cameras] == ["CAM1.cine", "cam2.cine", "Cam10.cine"]


def test_detect_input_root_supports_tiff_camera_subdirs(tmp_path):
    from modules.image_preprocessing import io

    root = tmp_path / "dataset"
    for name in ["cam2", "Cam1"]:
        folder = root / name
        folder.mkdir(parents=True)
        cv2.imwrite(str(folder / "frame_0001.tif"), np.array([[1, 2], [3, 4]], dtype=np.uint8))

    detected = io.detect_input_root(root)

    assert detected.input_kind == "tiff"
    assert [camera.camera_name for camera in detected.cameras] == ["Cam1", "cam2"]
    assert [path.name for path in detected.cameras[0].image_paths] == ["frame_0001.tif"]


def test_import_pycine_raises_clear_error_when_dependency_missing(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("pycine"):
            raise ModuleNotFoundError("No module named 'pycine'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from modules.image_preprocessing import io

    monkeypatch.delitem(sys.modules, "pycine", raising=False)
    monkeypatch.delitem(sys.modules, "pycine.file", raising=False)
    monkeypatch.delitem(sys.modules, "pycine.raw", raising=False)

    with pytest.raises(io.ImagePreprocessingIOError) as excinfo:
        io._import_pycine()

    message = str(excinfo.value)
    assert "CINE input requested" in message
    assert "pycine" in message
    assert "unavailable" in message


def test_plan_cine_tasks_uses_inclusive_bounds(monkeypatch, tmp_path):
    from modules.image_preprocessing import io

    def fake_read_header(_path):
        return {"cinefileheader": SimpleNamespace(FirstImageNo=0, ImageCount=3)}

    monkeypatch.setattr(io, "_import_pycine", lambda: (None, fake_read_header))

    tasks = io.plan_cine_tasks(
        cam_idx=1,
        cine_path=tmp_path / "cam1.cine",
        output_dir=tmp_path / "processed",
        start_frame=0,
        end_frame=2,
    )

    assert [task.cine_frame for task in tasks] == [0, 1, 2]
    assert [task.output_path.name for task in tasks] == [
        "frame_000000.tif",
        "frame_000001.tif",
        "frame_000002.tif",
    ]


def test_plan_cine_tasks_respects_nonzero_first_image_number(monkeypatch, tmp_path):
    from modules.image_preprocessing import io

    def fake_read_header(_path):
        return {"cinefileheader": SimpleNamespace(FirstImageNo=5, ImageCount=4)}

    monkeypatch.setattr(io, "_import_pycine", lambda: (None, fake_read_header))

    tasks = io.plan_cine_tasks(
        cam_idx=2,
        cine_path=tmp_path / "cam2.cine",
        output_dir=tmp_path / "processed",
        start_frame=6,
        end_frame=8,
        naming="source",
        start_index=10,
    )

    assert [task.cine_frame for task in tasks] == [6, 7, 8]
    assert [task.output_index for task in tasks] == [10, 11, 12]
    assert [task.output_relpath for task in tasks] == [
        "cam2/cam2_frame_000006.tif",
        "cam2/cam2_frame_000007.tif",
        "cam2/cam2_frame_000008.tif",
    ]


def test_run_batch_processing_with_parallel_workers_preserves_order(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    image_paths = []
    for i in range(6):
        img_path = input_dir / f"img_{i:04d}.tif"
        cv2.imwrite(str(img_path), np.array([[i * 10, i * 20], [i * 30, i * 40]], dtype=np.uint8))
        image_paths.append(img_path)

    from modules.image_preprocessing import io

    tasks = io.plan_tiff_tasks(cam_idx=1, output_dir=output_dir, image_paths=image_paths)

    result = run_batch_processing(
        tasks,
        output_dir=output_dir,
        settings={"invert": False, "low_in": 0, "high_in": 255},
        workers=2,
    )

    assert result.processed_count == 6
    assert result.failed_count == 0
    assert len(result.image_relpaths_by_camera[1]) == 6
    assert result.image_relpaths_by_camera[1] == [
        "cam1/frame_000000.tif",
        "cam1/frame_000001.tif",
        "cam1/frame_000002.tif",
        "cam1/frame_000003.tif",
        "cam1/frame_000004.tif",
        "cam1/frame_000005.tif",
    ]

    image_list = output_dir / "cam1ImageNames.txt"
    assert image_list.is_file()
    lines = image_list.read_text(encoding="utf-8").splitlines()
    assert lines == result.image_relpaths_by_camera[1]
