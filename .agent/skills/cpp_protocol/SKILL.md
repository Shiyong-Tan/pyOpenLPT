---
name: cpp-protocol
description: Use when interacting with pyOpenLPT C++ Python bindings, when getting AttributeError on Camera or other lpt objects, when setting camera intrinsics or extrinsics from Python
---

## Overview

pybind11 bindings expose C++ members **explicitly**. You cannot use arbitrary Python attribute syntax like `obj.foo = bar` unless `foo` is defined in the binding file. 

> [!IMPORTANT]
> **Check the Bindings First!** Always verify available methods and attributes in the C++ binding source files (located in `src/pybind_OpenLPT/`) before writing code. Do not assume a method exists just because it exists in the C++ header.

## When to Use

- Setting Camera intrinsics (`cam_mtx`, `dist_coeff`)
- Setting Camera extrinsics (`r_mtx`, `t_vec`)
- Updating refractive plane parameters (PinPlate)
- Getting `AttributeError: 'Camera' object has no attribute '...'`

## Quick Reference

| What You Want | Wrong | Correct | Binding File |
|---------------|-------|---------|--------------|
| Set focal length | `cam.fx = v` | `cam._pinhole_param.cam_mtx[0,0] = v` | `pyCamera.cpp` |
| Set refraction | `cam.setRefractivePlane(...)` | **Daisy-chain assignment** (see below) | `pyCamera.cpp` |
| Set translation | `cam.t = t` | `cam._pinhole_param.t_vec[:] = t` | `pyCamera.cpp` |

## Core Pattern: Daisy-Chain Assignment

Nested C++ structs in pybind11 often return **copies**. To modify them, you must re-assign the entire struct:

```python
# ❌ WRONG - Modifies a local copy, camera object is unchanged
cam._pinplate_param.plane.pt = lpt.Pt3D(x, y, z)

# ✅ CORRECT - Daisy-chain assignment
pp = cam._pinplate_param
pl = pp.plane
pl.pt = lpt.Pt3D(x, y, z)
# ... modify others ...
pp.plane = pl
cam._pinplate_param = pp
cam.updatePt3dClosest() # Trigger internal update
```

## When NOT to Use

- Pure Python camera calibration code
- Working with dict-based camera parameters

## Quick Reference

| What You Want | Wrong | Correct |
|---------------|-------|---------|
| Set focal length | `cam.fx = v` | `cam._pinhole_param.cam_mtx[0,0] = v` |
| Set principal point | `cam.cx = v` | `cam._pinhole_param.cam_mtx[0,2] = v` |
| Set distortion | `cam.dist = arr` | Loop: `cam._pinhole_param.dist_coeff[i] = arr[i]` |
| Set rotation | `cam.R = R` | `cam._pinhole_param.r_mtx[:,:] = R` |
| Set translation | `cam.t = t` | `cam._pinhole_param.t_vec[:] = t` |
| Set refractive plane | N/A | `cam.setRefractivePlane(pt, n, n_air, n_glass, n_medium, thick)` |

## Core Pattern

```python
# ❌ WRONG - AttributeError
cam.matrix = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
cam.dist = dist_coeffs

# ✅ CORRECT - Access via _pinhole_param
cam._pinhole_param.cam_mtx[0, 0] = f   # fx
cam._pinhole_param.cam_mtx[1, 1] = f   # fy
cam._pinhole_param.cam_mtx[0, 2] = cx
cam._pinhole_param.cam_mtx[1, 2] = cy
for i in range(len(dist)):
    cam._pinhole_param.dist_coeff[i] = dist[i]
```

## Common Mistakes

| Error | Cause | Fix |
|-------|-------|-----|
| `AttributeError: 'Camera' object has no attribute 'matrix'` | Non-exposed attribute | Use `_pinhole_param.cam_mtx` |
| `lineOfSight returns wrong values` | Didn't update `r_mtx_inv`/`t_vec_inv` | Update both forward and inverse |

## Reference Files

Binding source: `src/pybind_OpenLPT/pyCamera.cpp`

Check available attributes: `print(dir(cam._pinhole_param))`
