# OpenLPT Development Guide

This document defines the development environment, build instructions, and mandatory coding protocols for the pyOpenLPT project.

## Development Environment
- **Operating System**: Windows (Primary), Linux/Mac (Support)
- **Language**: Python 3.10+, C++ 17 (C++ Core)
- **Primary Entry Point**: `python GUI.py`

## Build & Installation
- Install dependencies: `pip install -r requirements.txt`
- Build C++ bindings: `python build_local.py` or `python setup.py install`
- Run GUI: `python GUI.py`

## Python Environment Rule
- For this repo, agents must use the `OpenLPT` conda environment for Python compilation, script execution, and verification commands.
- Preferred order:
  1. use `python ...` only if the active shell is already confirmed to be `OpenLPT`
  2. otherwise use `conda run -n OpenLPT python ...`
  3. if `conda run` is unreliable, use `C:\Users\tan_s\miniconda3\envs\OpenLPT\python.exe ...`
- Apply this rule to commands such as `py_compile`, `--help` checks, calibration script runs, and other Python-based validation in this project.
- Do not assume the default `python` points to the correct environment.

## Mandatory Coding Protocols
All AI agents and developers **MUST** adhere to the following protocols defined in the `.agent/skills/` directory:

0. **Check the Bindings First!** Always verify available methods and attributes in `src/pybind_OpenLPT/` before writing code for C++ objects. Do not assume methods exist.
1. **C++ Binding Interface**: When modifying or accessing `lpt.Camera` objects or any C++ parameters from Python, you **MUST** follow the **[cpp-protocol](file:///d:/0.Code/OpenLPTGUI/OpenLPT/.agent/skills/cpp_protocol/SKILL.md)**. Improper access will cause `AttributeError` or binary crashes.
2. **Internal Data Structures**: When interacting with `dataset`, `window_media`, or `window_planes` dictionaries, you **MUST** follow the **[python-protocol](file:///d:/0.Code/OpenLPTGUI/OpenLPT/.agent/skills/python_protocol/SKILL.md)**. Incorrect key access causes `KeyError`.

## Design Guidelines
- **Math/Physics First**: For refractive calibration, ensure coordinate shifts (Closest vs Farthest) are handled correctly.
- **Authority**: The C++ kernel (`lpt.Camera`) is the source of truth for ray tracing and triangulation. Always sync Python state to C++ objects before physical computation.
- **Fail-Fast**: Validate input consistency (e.g., wand length, camera mapping) at the start of every module.

## Workflow Commands
- **Testing**: Run scripts in `test/` directory.
- **Calibration**: Use `modules/camera_calibration/wand_calibration/`.
