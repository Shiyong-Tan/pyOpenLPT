# OpenLPT Agent Rules

## Python Environment Rule
- Use the `OpenLPT` conda environment for Python compilation, script execution, and verification commands in this repo.
- Preferred order:
  1. use `python ...` only if the current shell is already confirmed to be the `OpenLPT` environment
  2. otherwise use `conda run -n OpenLPT python ...`
  3. if `conda run` is unreliable, use `C:\Users\tan_s\miniconda3\envs\OpenLPT\python.exe ...`
- Apply this to `py_compile`, `--help` checks, calibration script runs, and Python-based validation.
- Do not assume the default `python` executable is correct.

## Existing Project Guidance
- Check available bindings in `src/pybind_OpenLPT/` before using C++ objects from Python.
- Follow the repo protocols documented in `CLAUDE.md` for camera/window/media data access and refractive-calibration workflows.

## Workspace Rule
- All code changes must be made in the current repository root working directory: `D:\0.Code\OpenLPTGUI\OpenLPT`.
- Do not create or use a separate git worktree for this repository unless the user explicitly requests it.
