---
name: python-protocol
description: Use when interacting with internal data structures of pyOpenLPT (dataset, window_media, window_planes) to ensure correct key access and prevent KeyError or logic bugs.
---

# Python Data Structure Protocol (pyOpenLPT)

## Overview

This skill documents the schema and usage of core internal data structures used in the refractive calibration framework. Consistent access to these structures is critical for avoiding `KeyError` and logic bugs.

## When to Use

- Accessing `dataset` in calibrators or optimizers.
- Using `window_media` for refractive parameters (thickness, refractive indices).
- Accessing `window_planes` for geometry (normals, points).
- Implementing new optimization stages or data processing steps.

## Key Data Structures

### 1. `window_media` (Refractive Indices & Thickness)
A dictionary mapping window ID (`wid`) to media properties.

```python
# Schema
window_media = {
    wid: {
        'n1': float,        # n_air (usually 1.0)
        'n2': float,        # n_glass
        'n3': float,        # n_medium (e.g. water)
        'thickness': float  # glass thickness in mm
    }
}

# ✅ CORRECT ACCESS
media = window_media[wid]
thick = media['thickness']
n_glass = media['n2']

# ❌ WRONG ACCESS
thick = dataset['window_thickness_mm'] # dataset usually lacks this key
```

### 2. `dataset` (Observations & Metrics)
A dictionary containing aggregated wand observations and global metadata.

```python
# Common Schema
dataset = {
    "frames": [int, ...],           # List of frame IDs
    "cam_ids": [int, ...],          # List of active camera IDs
    "obsA": {fid: {cid: [u, v]}},   # Observations for point A
    "obsB": {fid: {cid: [u, v]}},   # Observations for point B
    "wand_length": float,           # Target wand length (mm)
    "n_air": float,                 # (Optional legacy) global n_air
    "n_glass": float,               # (Optional legacy) global n_glass
    "n_medium": float,              # (Optional legacy) global n_medium
}
```

### 3. `window_planes` (Geometry)
A dictionary mapping `wid` to orientation and position.

```python
# Schema
window_planes = {
    wid: {
        'plane_pt': np.ndarray(3,),  # Point ON the plane (closest to origin/camera)
        'plane_n':  np.ndarray(3,)   # Normal vector (n)
    }
}
```

## Common Pitfalls

| Symptom | Cause | Fix |
|---------|-------|-----|
| `KeyError: 'window_thickness_mm'` | Accessing `dataset` instead of `window_media` | Use `window_media[wid]['thickness']` |
| `TypeError: Object not subscriptable` | Confusion between `window_media` and `window_planes` | Ensure `wid` is used as primary key |
| `KeyError: 'n1'` | Trying to access `media['n_air']` | Use the correct keys: `n1`, `n2`, `n3` |

## Discovery Tip

If you encounter a `KeyError` on a dictionary passed to a function, use `print(dataset.keys())` or `view_file` on the `_collect_observations` and `calibrate` methods to verify the structure.
