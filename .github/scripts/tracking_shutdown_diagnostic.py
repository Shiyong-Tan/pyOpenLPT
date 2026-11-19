"""Task-only Linux shutdown diagnostics; never an evaluator or production fix.

Entry points: prepare -> pip build/install -> collect -> diagnose. Only diagnose
and its GDB children import the fresh extension. No variant or retry is run.
"""

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / ".opencode/diagnostics/tracking_shutdown"
EVIDENCE = OUT / "evidence"
BASE = "d45cbccf7430300937df911700beafdc0eddc829"
PROTECTED = ["src", "inc", "openlpt.py", "setup.py", "CMakeLists.txt",
             "cmake", "_version.py", "pyproject.toml"]
CASES = [("import-exit", 90), ("empty-stb", 180),
         ("reduced-0-0", 300), ("full-0-99", 1200)]


def contained(root, path):
    """Return a resolved path inside root, rejecting escapes and root itself."""
    root, path = Path(root).resolve(), Path(path).resolve()
    if path == root or root not in path.parents:
        raise ValueError(f"Path outside allowed output root: {path}")
    return path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = contained(OUT, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def command(argv, cwd=ROOT):
    return subprocess.check_output(argv, cwd=cwd, text=True, stderr=subprocess.STDOUT)


def records(path):
    """Match BasicSetting's nonempty, comment-stripped positional records."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return lines, [(i, value) for i, line in enumerate(lines)
                   if (value := line.split("#", 1)[0].strip())]


def rewrite_config(fixture, destination, end, *, workspace=ROOT, output_root=OUT):
    """Copy only config/list text; preserve input identity and camera/frame order.

    Args:
        fixture: Original test_STB directory, treated read-only.
        destination: Fresh case directory under output_root.
        end: 0 for reduced or 99 for the original frame range.
        workspace: Containment boundary (also used by manufactured local tests).
        output_root: Allowed task-specific derived output location.
    Returns:
        Manifest of copied config, exact input hashes and resolved list order.
    """
    fixture = contained(workspace, fixture)
    destination = contained(output_root, destination)
    contained(workspace, destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to reuse case outputs: {destination}")
    if end not in (0, 99):
        raise ValueError("Only authorized frame bounds 0..0 and 0..99")
    lines, values = records(fixture / "config.txt")
    data = [v for _, v in values]
    if (len(data) != 18 or data[:4] != ["0,99", "1", "0", "4"]
            or data[15:] != ["Tracer", "tracerConfig.txt", "0,-1"]):
        raise ValueError("Unexpected fixture schema; do not guess path rewrites")
    inputs = {}

    def original(raw):
        path = contained(workspace, fixture / raw)
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.read_bytes()[:80].startswith(b"version https://git-lfs.github.com/"):
            raise ValueError(f"Unresolved LFS pointer: {path}")
        inputs[str(path)] = digest(path)
        return path

    original("config.txt")
    cameras = []
    for k in range(4, 8):
        raw, intensity = data[k].rsplit(",", 1)
        camera = original(raw)
        cameras.append(camera.as_posix())
        lines[values[k][0]] = camera.as_posix() + "," + intensity
    lists = []
    # Validate every input before creating any native output/copy.
    for k in range(8, 12):
        path = original(data[k])
        rows = path.read_text(encoding="utf-8").splitlines()
        images = [original(row.strip()).as_posix() for row in rows if row.strip()]
        if len(images) != 100:
            raise ValueError("Expected all 100 original frames per camera")
        lists.append(images)
    tracer = original(data[16])
    native_output = contained(output_root, destination / "native-output")
    lines[values[0][0]] = f"0,{end}"
    lines[values[14][0]] = native_output.as_posix() + "/"
    lines[values[16][0]] = (destination / "tracerConfig.txt").as_posix()
    destination.mkdir(parents=True)
    native_output.mkdir()
    # Byte-identical tracer config; all numerical parameters unchanged.
    shutil.copyfile(tracer, destination / "tracerConfig.txt")
    for i, images in enumerate(lists):
        target = destination / f"cam{i + 1}ImageNames.txt"
        target.write_text("\n".join(images) + "\n", encoding="utf-8")
        lines[values[8 + i][0]] = target.as_posix()
    config = destination / "config.txt"
    config.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {"config": str(config), "output": str(native_output),
                "frames": [0, end], "threads": 0, "cameras": cameras,
                "ordered_images": lists, "original_sha256": inputs,
                "tracer_sha256": digest(destination / "tracerConfig.txt"),
                "derived_config_sha256": digest(config)}
    verify_case(manifest, output_root=output_root)
    return manifest


def verify_case(manifest, *, output_root=OUT):
    """Check all writable native paths and immutable inputs before construction."""
    config = contained(output_root, manifest["config"])
    if digest(config) != manifest["derived_config_sha256"]:
        raise ValueError("Derived configuration changed after verified rewrite")
    _, values = records(config)
    data = [v for _, v in values]
    if len(data) != 18:
        raise ValueError("Unexpected derived config")
    output = contained(output_root, data[14])
    if output != Path(manifest["output"]).resolve():
        raise ValueError("Output redirect mismatch")
    if data[0] != ",".join(map(str, manifest["frames"])) or data[2] != "0":
        raise ValueError("Frame bounds or configured threads changed")
    if data[15] != "Tracer" or data[17] != "0,-1":
        raise ValueError("Unauthorized object/restart config")
    actual_cameras = [v.rsplit(",", 1)[0] for v in data[4:8]]
    if actual_cameras != manifest["cameras"]:
        raise ValueError("Camera order changed")
    for i, raw in enumerate(data[8:12]):
        path = contained(output_root, raw)
        if path.read_text(encoding="utf-8").splitlines() != manifest["ordered_images"][i]:
            raise ValueError("Frame/list order changed")
    tracer = contained(output_root, data[16])
    if digest(tracer) != manifest["tracer_sha256"]:
        raise ValueError("Numerical tracer configuration changed")
    for raw, sha in manifest["original_sha256"].items():
        if digest(raw) != sha:
            raise ValueError(f"Original input changed: {raw}")
    return config, output, tracer


def source_check():
    command(["git", "diff", "--exit-code", BASE, "--", *PROTECTED])
    command(["git", "diff", "--exit-code", "--cached"])
    if command(["git", "rev-parse", "HEAD^"]).strip() != BASE:
        raise ValueError("Expected one diagnostic child commit of exact d45")
    changed = command(["git", "diff", "--name-only", BASE, "HEAD"]).splitlines()
    if sorted(changed) != sorted([
            ".github/scripts/tracking_shutdown_diagnostic.py",
            ".github/workflows/tracking_shutdown_diagnostic.yml"]):
        raise ValueError("Diagnostic commit must contain only the two leased files")


def prepare():
    source_check()
    if (OUT / "build-source").exists():
        raise FileExistsError("Refusing stale build-source / rerun in this workspace")
    target = OUT / "build-source"
    # Copy tracked build/package inputs, not fixtures, notebooks or Git metadata.
    files = command(["git", "ls-files", "-z"]).rstrip("\0").split("\0")
    selected = [p for p in files if Path(p).parts[0] in
                {"src", "inc", "cmake", "modules", "gui", "utils"}
                or len(Path(p).parts) == 1]
    hashes = {}
    for raw in selected:
        source = contained(ROOT, ROOT / raw)
        dest = contained(OUT, target / raw)
        if source.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
            hashes[raw] = digest(source)
            if digest(dest) != hashes[raw]:
                raise ValueError("Build source copy mismatch")
    # pybind11 strips non-Debug builds. Disable ONLY that symbol-removal command;
    # keep Release, LTO, optimization and native algorithms unchanged.
    true_exe = shutil.which("true")
    if not true_exe:
        raise RuntimeError("Missing preinstalled true for symbol retention")
    (OUT / "symbols-toolchain.cmake").write_text(
        f'set(CMAKE_STRIP "{true_exe}" CACHE FILEPATH "Diagnostic: retain symbols" FORCE)\n',
        encoding="utf-8")
    write_json(EVIDENCE / "build-source.json", {
        "head": command(["git", "rev-parse", "HEAD"]).strip(),
        "base": BASE, "source_sha256": hashes, "symbol_only_strip_override": true_exe})


def collect():
    source_check()
    if Path(sys.prefix).resolve() != (OUT / "envs/OpenLPT").resolve():
        raise RuntimeError("Not the workspace-contained OpenLPT environment")
    spec = importlib.util.find_spec("pyopenlpt")  # Find only; import is a GDB case.
    if spec is None or spec.origin is None:
        raise RuntimeError("Fresh extension not found")
    binary = contained(OUT / "envs/OpenLPT", spec.origin)
    built = list((OUT / "build-source/build").glob("lib*/pyopenlpt*.so"))
    if not built or not any(digest(p) == digest(binary) for p in built):
        raise RuntimeError("Installed extension does not match this checkout build")
    sections = command(["readelf", "--sections", "--wide", str(binary)])
    (EVIDENCE / "binary-sections.txt").write_text(sections, encoding="utf-8")
    if ".debug_info" not in sections or ".debug_line" not in sections:
        raise RuntimeError("Missing debug symbols; do not spend native-run budget")
    provenance = json.loads((EVIDENCE / "build-source.json").read_text())
    for raw, sha in provenance["source_sha256"].items():
        if digest(ROOT / raw) != sha or digest(OUT / "build-source" / raw) != sha:
            raise ValueError("Source changed during build")
    databases = list((OUT / "build-source/build").rglob("compile_commands.json"))
    if not databases:
        raise RuntimeError("Missing actual compile commands")
    commands = [c for p in databases for c in json.loads(p.read_text())]
    for name in ("main.cpp", "IPR.cpp", "Config.cpp", "pyOpenLPT.cpp"):
        matches = [c for c in commands if Path(c["file"]).name == name]
        if not matches or any(not all(flag in c["command"] for flag in
                                    ("-g", "-fno-omit-frame-pointer", "-O3", "-fopenmp"))
                              for c in matches):
            raise RuntimeError(f"Missing Release/symbol/OpenMP flags: {name}")
    write_json(EVIDENCE / "compile-commands.json", commands)
    caches = list((OUT / "build-source/build").rglob("CMakeCache.txt"))
    cache_text = "\n".join(p.read_text(errors="replace") for p in caches)
    if "CMAKE_BUILD_TYPE:STRING=Release" not in cache_text:
        raise RuntimeError("Not a Release build")
    (EVIDENCE / "cmake-cache.txt").write_text(cache_text, encoding="utf-8")
    import pybind11  # Pure Python package, not the project extension.
    if pybind11.__version__ != "3.1.0":
        raise RuntimeError("Unexpected pybind11 version")
    ldd = command(["ldd", str(binary)])
    (EVIDENCE / "native-dependencies.txt").write_text(ldd, encoding="utf-8")
    omp_libraries = {}
    for raw in re.findall(r"=>\s+(/\S+)", ldd):
        if "gomp" in raw or "libomp" in raw:
            library = Path(raw).resolve()
            omp_libraries[str(library)] = digest(library)
    if not omp_libraries:
        raise RuntimeError("No loaded OpenMP runtime found")
    packages = (command(["dpkg-query", "-W", "libgomp1", "libc6"])
                if shutil.which("dpkg-query") else "dpkg-query unavailable")
    header = Path(pybind11.get_include()) / "pybind11/iostream.h"
    (EVIDENCE / "pybind11-iostream-header.txt").write_text(
        "\n".join(header.read_text().splitlines()[:40]) + "\n", encoding="utf-8")
    write_json(EVIDENCE / "provenance.json", {
        "head": command(["git", "rev-parse", "HEAD"]).strip(), "base": BASE,
        "python": sys.version, "executable": sys.executable, "prefix": sys.prefix,
        "binary": str(binary), "binary_sha256": digest(binary),
        "pybind11": pybind11.__version__, "pybind11_include": pybind11.get_include(),
        "pybind11_iostream_sha256": digest(header), "openmp_runtime_sha256": omp_libraries,
        "system_runtime_packages": packages, "uname": list(os.uname()),
        "compiler": command(["g++", "--version"]), "gdb": command(["gdb", "--version"]),
        "cmake": command(["cmake", "--version"]), "cpu_count": os.cpu_count(),
        "compiler_jobs": 4, "address_space_limit_per_process_bytes": 8589934592,
        "environment": {k: v for k, v in os.environ.items()
                        if k.startswith("OMP_") or k in
                        ("HOME", "TMP", "TEMP", "TMPDIR", "CONDA_ENVS_PATH",
                         "CONDA_PKGS_DIRS", "CONDARC", "PIP_CACHE_DIR",
                         "CFLAGS", "CXXFLAGS", "CMAKE_TOOLCHAIN_FILE",
                         "CMAKE_BUILD_PARALLEL_LEVEL", "DEBUGINFOD_CACHE_PATH")}})


def process_info(pid):
    """Linux PID identity/state/parent; tolerate already-reaped processes."""
    try:
        tail = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "state": tail[0], "ppid": int(tail[1]), "start": tail[19]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def descendants(owned):
    """Refresh descendants without moving any process to a new session/group."""
    table = [v for p in Path("/proc").iterdir() if p.name.isdigit()
             if (v := process_info(int(p.name))) is not None]
    parents = {p for p, start in owned.items()
               if (v := process_info(p)) and v["start"] == start}
    while True:
        found = {v["pid"] for v in table if v["ppid"] in parents}
        if found <= parents:
            break
        parents |= found
    for v in table:
        if v["pid"] in parents:
            owned[v["pid"]] = v["start"]


def live_owned(owned):
    return [v for pid, start in owned.items()
            if (v := process_info(pid)) and v["start"] == start
            and v["state"] not in ("Z", "X")]


def classify(log, gdb_exit, timed_out, stopped):
    fault = re.search(r"received signal (SIG[A-Z0-9]+)", log)
    normal = bool(re.search(r"^DIAG_EXIT_CODE=0(?:\r?\n|$)", log, re.MULTILINE))
    stack = bool(re.search(r"^#0\s+", log, re.MULTILINE))
    markers = re.findall(r"^DIAG_PHASE=([A-Z_]+)\r?$", log, re.MULTILINE)
    status = ("timeout" if timed_out else "processes_still_active" if not stopped
              else "fault_captured" if fault and stack
              else "fault_without_stack" if fault else "normal_exit"
              if normal and gdb_exit == 0 else "failed_or_unknown_exit")
    return {"status": status, "fault": fault.group(1) if fault else None,
            "stack_present": stack, "normal_exit": normal, "gdb_exit": gdb_exit,
            "timeout": timed_out, "owned_processes_stopped": stopped,
            "last_phase": markers[-1] if markers else None}


def gdb_case(name, seconds):
    manifest_path = EVIDENCE / f"{name}-inputs.json"
    if name != "import-exit":
        manifest = rewrite_config(ROOT / "test/inputs/test_STB", OUT / "cases" / name,
                                  0 if name == "reduced-0-0" else 99)
        write_json(manifest_path, manifest)
    argv = ["gdb", "-nx", "-nh", "--batch", "--return-child-result"]
    for cmd in ["set pagination off", "set confirm off", "set history save off",
                "set debuginfod enabled off", "set print elements 12",
                "set print repeats 4", "set print max-depth 4", "set backtrace limit 40",
                "handle SIGSEGV stop print nopass", "handle SIGABRT stop print nopass",
                "run", "info program", "thread apply all bt full 40",
                "info sharedlibrary", 'printf "DIAG_EXIT_CODE=%d\\n", $_exitcode']:
        argv += ["-ex", cmd]
    argv += ["--args", sys.executable, "-I", "-B", "-X", "utf8", "-X",
             "faulthandler", str(Path(__file__).resolve()), "child", "--case", name]
    log_path = EVIDENCE / f"{name}-gdb.log"
    owned = {}
    start = time.monotonic()
    timed_out = False
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        info = process_info(proc.pid)
        if info:
            owned[proc.pid] = info["start"]
        while proc.poll() is None:
            descendants(owned)
            if time.monotonic() - start > seconds:
                timed_out = True
                # Explicit case deadline, no retry. Kill verified owned identities,
                # including inferior, without setsid or killing the parent's group.
                for item in reversed(live_owned(owned)):
                    try:
                        os.kill(item["pid"], signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                break
            time.sleep(0.25)
        proc.wait(timeout=15)
    for _ in range(20):
        descendants(owned)
        if not live_owned(owned):
            break
        time.sleep(0.1)
    result = classify(log_path.read_text(errors="replace"), proc.returncode,
                      timed_out, not live_owned(owned))
    result.update(case=name, argv=argv, deadline_seconds=seconds,
                  elapsed_seconds=round(time.monotonic() - start, 2),
                  owned_pid_start_ticks=owned, remaining=live_owned(owned))
    write_json(EVIDENCE / f"{name}-result.json", result)
    if name != "import-exit":
        verify_case(manifest)  # Original byte identity after the native case too.
    return result


def diagnose():
    if sys.platform != "linux":
        raise RuntimeError("Linux native debugger required")
    source_check()
    results = []
    summary = {"purpose": "Diagnostic collection only; NOT tracking acceptance",
               "cases": results, "state": "collecting"}
    write_json(EVIDENCE / "summary.json", summary)
    for name, seconds in CASES:
        print(f"DIAGNOSTIC_CASE={name} deadline_seconds={seconds}", flush=True)
        result = gdb_case(name, seconds)
        results.append(result)
        summary["state"] = result["status"]
        write_json(EVIDENCE / "summary.json", summary)
        if result["status"] != "normal_exit":
            # Stop at first fault/error/timeout, never continue to the full run.
            if result["status"] == "fault_captured":
                print("Fault stack collected; no tracking PASS claim.", flush=True)
                return
            raise RuntimeError(f"Collection incomplete: {result['status']}")
    summary["state"] = "no_reproduction_not_tracking_acceptance"
    write_json(EVIDENCE / "summary.json", summary)
    print("All diagnostic cases exited normally; race/acceptance remain unproven.", flush=True)


def child(name):
    print("DIAG_PHASE=BEFORE_IMPORT", flush=True)
    provenance = json.loads((EVIDENCE / "provenance.json").read_text())
    import pyopenlpt as lpt
    if (Path(lpt.__file__).resolve() != Path(provenance["binary"]).resolve()
            or digest(lpt.__file__) != provenance["binary_sha256"]):
        raise RuntimeError("Wrong or changed native extension")
    print(f"NATIVE_BINARY={lpt.__file__}", flush=True)
    print("DIAG_PHASE=AFTER_IMPORT", flush=True)
    if name != "import-exit":
        manifest = json.loads((EVIDENCE / f"{name}-inputs.json").read_text())
        config, output, tracer = verify_case(manifest)
        if name == "empty-stb":
            print("DIAG_PHASE=BEFORE_SETTINGS", flush=True)
            setting = lpt.BasicSetting()
            if not setting.readConfig(str(config)):
                raise RuntimeError("BasicSetting read failed")
            if Path(setting._output_path).resolve() != output:
                raise RuntimeError("Native output path escaped")
            print("DIAG_PHASE=BEFORE_STB_CONSTRUCT", flush=True)
            obj = lpt.STB(setting, "Tracer", str(tracer))
            print("DIAG_PHASE=AFTER_STB_CONSTRUCT", flush=True)
            del obj
            gc.collect()
            print("DIAG_PHASE=AFTER_STB_DELETE", flush=True)
            del setting
            gc.collect()
            print("DIAG_PHASE=AFTER_SETTINGS_DELETE", flush=True)
        else:
            print("DIAG_PHASE=BEFORE_RUN", flush=True)
            lpt.run(str(config))
            print("DIAG_PHASE=AFTER_RUN", flush=True)
    print("DIAG_PHASE=BEFORE_EXIT", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "collect", "diagnose", "child"])
    parser.add_argument("--case", choices=[name for name, _ in CASES])
    args = parser.parse_args()
    try:
        if args.mode == "child":
            if not args.case:
                parser.error("child requires --case")
            child(args.case)
        else:
            {"prepare": prepare, "collect": collect, "diagnose": diagnose}[args.mode]()
    except Exception as exc:
        # No environment dump (tokens); only concrete collection failure.
        write_json(EVIDENCE / f"{args.mode}-error.json",
                   {"type": type(exc).__name__, "message": str(exc),
                    "tracking_acceptance": False})
        raise


if __name__ == "__main__":
    main()
