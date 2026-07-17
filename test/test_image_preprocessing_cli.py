# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false
import builtins
import contextlib
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_preprocessing_modules_import_without_pyside6(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("PySide6"):
            raise ModuleNotFoundError("No module named 'PySide6'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    for module_name in [
        "modules.image_preprocessing.core",
        "modules.image_preprocessing.io",
        "modules.image_preprocessing.runner",
        "modules.image_preprocessing.cli",
    ]:
        sys.modules.pop(module_name, None)
        importlib.import_module(module_name)


def test_cli_main_processes_tiny_tiffs_and_writes_image_list(tmp_path):
    from modules.image_preprocessing.cli import main

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    first = input_dir / "img_0001.tif"
    second = input_dir / "img_0002.tif"
    cv2.imwrite(str(first), np.array([[10, 20], [30, 40]], dtype=np.uint8))
    cv2.imwrite(str(second), np.array([[50, 60], [70, 80]], dtype=np.uint8))

    exit_code = main(
        [
            "--image",
            str(first),
            "--image",
            str(second),
            "--camera-index",
            "1",
            "--output-dir",
            str(output_dir),
            "--background",
            "mean",
            "--bg-count",
            "2",
            "--invert",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "cam1" / "frame_000000.tif").is_file()
    assert (output_dir / "cam1" / "frame_000001.tif").is_file()
    image_list = output_dir / "cam1ImageNames.txt"
    assert image_list.is_file()
    assert image_list.read_text(encoding="utf-8").splitlines() == [
        "cam1/frame_000000.tif",
        "cam1/frame_000001.tif",
    ]


def test_cli_main_help_prints_usage(capsys):
    from modules.image_preprocessing.cli import main

    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])

    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "openlpt preprocess" in captured.out
    assert "--input-root" in captured.out


def test_cli_main_parallel_workers_processes_in_order(tmp_path):
    from modules.image_preprocessing.cli import main

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    images = []
    for i in range(4):
        img_path = input_dir / f"img_{i:04d}.tif"
        cv2.imwrite(str(img_path), np.array([[i * 10, i * 20], [i * 30, i * 40]], dtype=np.uint8))
        images.append(str(img_path))

    exit_code = main(
        [
            "--image",
            images[0],
            "--image",
            images[1],
            "--image",
            images[2],
            "--image",
            images[3],
            "--camera-index",
            "1",
            "--output-dir",
            str(output_dir),
            "--workers",
            "2",
        ]
    )

    assert exit_code == 0
    image_list = output_dir / "cam1ImageNames.txt"
    assert image_list.is_file()
    lines = image_list.read_text(encoding="utf-8").splitlines()
    assert lines == [
        "cam1/frame_000000.tif",
        "cam1/frame_000001.tif",
        "cam1/frame_000002.tif",
        "cam1/frame_000003.tif",
    ]
    for i in range(4):
        assert (output_dir / "cam1" / f"frame_{i:06d}.tif").is_file()


def test_cli_main_workers_zero_uses_auto_worker_count(tmp_path):
    from modules.image_preprocessing.cli import main

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    image_path = input_dir / "img_0000.tif"
    cv2.imwrite(str(image_path), np.array([[10, 20], [30, 40]], dtype=np.uint8))

    exit_code = main(["--image", str(image_path), "--camera-index", "1", "--output-dir", str(output_dir), "--workers", "0"])

    assert exit_code == 0
    assert (output_dir / "cam1" / "frame_000000.tif").is_file()


def test_cli_main_strips_windows_cmd_single_quoted_paths(tmp_path):
    from modules.image_preprocessing.cli import main

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    image_path = input_dir / "img_quoted.tif"
    cv2.imwrite(str(image_path), np.array([[1, 2], [3, 4]], dtype=np.uint8))

    exit_code = main(
        [
            "--image",
            f"'{image_path}'",
            "--camera-index",
            "1",
            "--output-dir",
            f"'{output_dir}'",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "cam1" / "frame_000000.tif").is_file()


def test_cli_main_input_root_defaults_output_dir_and_detects_relative_cines(monkeypatch, tmp_path):
    from modules.image_preprocessing import io
    from modules.image_preprocessing import cli

    root = tmp_path / "dataset"
    for name in ["Cam2", "cam1"]:
        folder = root / name
        folder.mkdir(parents=True)
        (folder / f"{name}.cine").write_bytes(b"fake")

    def fake_read_header(path):
        return {"cinefileheader": SimpleNamespace(FirstImageNo=0, ImageCount=3)}

    observed = {}

    def fake_run_batch_processing(tasks, *, output_dir, settings, backgrounds, workers):
        observed["tasks"] = list(tasks)
        observed["output_dir"] = output_dir
        observed["workers"] = workers
        return SimpleNamespace(
            processed_count=len(tasks),
            failed_count=0,
            output_dir=output_dir,
            image_list_files={0: output_dir / "cam0ImageNames.txt", 1: output_dir / "cam1ImageNames.txt"},
            failures=[],
        )

    monkeypatch.setattr(io, "_import_pycine", lambda: (None, fake_read_header))
    monkeypatch.setattr(cli, "run_batch_processing", fake_run_batch_processing)

    cwd = Path.cwd()
    try:
        monkeypatch.chdir(tmp_path)
        exit_code = cli.main(["--input-root", "dataset", "--frames", "0", "1"])
    finally:
        monkeypatch.chdir(cwd)

    assert exit_code == 0
    output_dir = root / "imgFile"
    assert observed["output_dir"] == output_dir.resolve()
    assert [task.cam_idx for task in observed["tasks"]] == [0, 0, 1, 1]
    assert [task.source_path.name for task in observed["tasks"][::2]] == ["cam1.cine", "Cam2.cine"]
    assert observed["tasks"][0].source_path is not None


def test_cli_main_input_root_requires_frames_for_cine(capsys, tmp_path):
    from modules.image_preprocessing.cli import main

    root = tmp_path / "dataset"
    folder = root / "Cam1"
    folder.mkdir(parents=True)
    (folder / "cam1.cine").write_bytes(b"fake")

    with pytest.raises(SystemExit) as excinfo:
        main(["--input-root", str(root)])

    assert excinfo.value.code != 0
    captured = capsys.readouterr()
    assert "--frames START END is required with CINE inputs" in captured.err


def test_cli_main_workers_validation_rejects_negative(capsys):
    from modules.image_preprocessing.cli import main

    with pytest.raises(SystemExit) as excinfo:
        main(["--image", "dummy.tif", "--output-dir", "out", "--workers", "-1"])

    assert excinfo.value.code != 0
    captured = capsys.readouterr()
    assert "--workers must be >= 0" in captured.err


def test_openlpt_preprocess_route_dispatches_without_gui_dependencies(monkeypatch):
    import openlpt

    observed = {}

    def fake_preprocess_main(argv):
        observed["argv"] = list(argv)
        return 0

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "modules.image_preprocessing.cli":
            return SimpleNamespace(main=fake_preprocess_main)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(sys, "argv", ["openlpt.py", "preprocess", "--help"])

    with pytest.raises(SystemExit) as excinfo, contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
        openlpt.main()

    assert excinfo.value.code == 0
    assert observed["argv"] == ["--help"]
