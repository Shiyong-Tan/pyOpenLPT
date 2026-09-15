# LPT compute-mode selection

## Purpose

This change exposes the optional exact GPU assistance as an explicit LPT run
choice while retaining a guaranteed CPU-only path.

The default mode allows exact CUDA kernels when the installed OpenLPT build
supports them.  `CPU only (significantly slower)` sets
`CUDA_VISIBLE_DEVICES=-1` for the launched process, which forces those optional
kernels to take their existing CPU fallback.

## Backend selection

The Tracking view uses the first available backend in this order:

1. `build/Release/OpenLPT.exe` on Windows;
2. an installed `build/Release/run_openlpt_exact.sh` WSL bundle; or
3. the existing `python -m openlpt` fallback.

The WSL bundle is optional and is not created by this change.  It is selected
only on Windows when no native managed executable exists.  Windows drive paths
for the runner and configuration are converted to `/mnt/<drive>/...` paths
before `wsl.exe` is launched.

For CPU-only WSL runs, `env CUDA_VISIBLE_DEVICES=-1` is placed inside the WSL
command as well as in the parent process environment.  Unchecking CPU-only
restores a fresh copy of the system environment on the next run.

## Correctness boundary

This UI change does not implement a new numerical algorithm.  The exact CUDA
morphology backend remains limited to comparison and Boolean propagation; its
existing automatic CPU fallback is unchanged.  Hough accumulation and all
time-dependent LPT decisions remain on CPU.

## Verification

The Python module compiles successfully in the OpenLPT environment.  A launch
argument test verified Windows-to-WSL conversion for both GPU-enabled and
CPU-only runs and confirmed that only CPU-only mode injects the CUDA disable
variable.
