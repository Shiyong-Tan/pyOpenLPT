# Synchronize voxel scale with the configured view volume

## Problem

OpenLPT writes `Voxel to MM` as the physical width of one voxel. The tracking
configuration convention uses 1000 voxels across the X extent, so the correct
conversion is:

```text
voxel_to_mm = (x_max - x_min) / 1000
```

Automatic view-volume estimation already applied this formula. Manual edits to
the X minimum or maximum did not, however, so a user could enlarge `[-5, 5]` to
`[-15, 15]` while the saved scale incorrectly remained `0.01` instead of
changing to `0.03`.

That stale scale also changes every configuration parameter expressed in voxels
but interpreted physically, including the IPR 3D tolerance.

## Change

The X minimum and maximum controls now update `Voxel to MM` immediately using
the same 1000-voxel convention as automatic volume estimation. Invalid or
temporarily reversed bounds do not overwrite the last valid positive scale.

This is a configuration-generation correction. It does not modify the native
LPT algorithm, camera model, VSC algorithm, or any existing result file.

## Acceptance examples

| X range (mm) | Expected voxel-to-mm |
| --- | ---: |
| `[-5, 5]` | `0.01` |
| `[-15, 15]` | `0.03` |
| `[-200, 200]` | `0.4` |

The generated `config.txt` continues to serialize the value already displayed
by the GUI, so no separate save-time calculation can diverge from the UI.
