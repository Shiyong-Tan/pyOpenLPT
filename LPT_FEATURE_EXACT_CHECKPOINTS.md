# Exact frame-boundary checkpoints and safe backend pause

## Why ordinary 500-frame output is not a checkpoint

The existing `ConvergeTrack` snapshots are scientific exports. They use six
decimal places and can omit active-track files at phase boundaries, while a
true continuation also needs inactive tracks, exited tracks, and the exact
in-memory BubbleRef image and intensity. Renaming a start frame or pointing at
an arbitrary result folder therefore cannot reproduce uninterrupted tracking.

## Checkpoint contents

Each object type receives an independent
`Checkpoints/object_N/frame_FRAME/` directory containing:

- active long tracks;
- active short tracks;
- inactive long tracks not yet finalized;
- exited tracks pending final output;
- `BubbleRefExact.bin` for Bubble tracking;
- optional run-identity metadata supplied by the GUI; and
- `CheckpointComplete.txt`, written last.

Checkpoint CSVs use `max_digits10`, the standard decimal precision needed to
round-trip every binary64 value through the existing text loader exactly. The
BubbleRef binary stores dimensions, cached mean intensity, and every double
pixel under a versioned magic header; user-facing integer TIFFs are never used
for an exact checkpoint resume.

## Publication and retention

A checkpoint is built in a temporary sibling directory. Only after all state
and the completion marker have been flushed is that directory renamed into
place. If a previous checkpoint at the same frame exists, it remains available
as a backup until the replacement is published. Incomplete temporary folders
are never recognized as resumable checkpoints.

The newest two complete checkpoint frames are retained per object type. This
bounds storage while preserving one rollback point if the newest snapshot is
damaged.

## Scheduling and pause behavior

The backend writes a checkpoint every 500 checkpointable frames. A file named
`.openlpt_pause_requested` in the output folder requests a safe pause. The
backend finishes the current frame, waits until initial-phase track formation
is complete if necessary, writes the complete checkpoint, removes the request,
prints `OPENLPT_PAUSED frame=N`, and exits successfully.

For multiple object types, all objects must reach the same checkpointable frame
before publication. Resume maps each object to its matching object-specific
checkpoint directory.

Legacy resume remains available for old scientific snapshots. It is explicitly
reported as non-bit-exact because integer BubbleRef TIFF storage and six-place
track files cannot satisfy the exact continuation contract.

## GUI workflow and input identity

The Tracking view replaces the destructive first-click stop action with
`Pause Safely`. The first click writes the pause request and lets the backend
finish its current frame. The button then becomes `Force Stop`; only that
explicit second click terminates a non-responsive process mid-frame.

After a safe pause marker is received, the GUI enables
`Resume latest safe checkpoint`. A resume scan accepts only a frame that is
complete for every configured object type. Bubble objects must contain the
lossless BubbleRef binary; Tracer objects do not require it.

Before every run, the GUI hashes the unchanged master `config.txt`, all camera
files, image lists, and object configuration files. The identity is stored with
each checkpoint. Resume is refused if any byte of those inputs changes. The GUI
uses a temporary runtime config for the load flag and checkpoint paths; it does
not rewrite the master config.

The resume checkbox owns continuation behavior. When it is unchecked, a stale
manual value such as `1,43000` in the master config is overridden with `0,0` in
the runtime copy, preventing an incomplete legacy folder from being treated as
a safe checkpoint.

Checkpoint discovery was tested with a Bubble plus Tracer fixture. A frame
present for only one object was rejected, the newest common complete frame was
selected, a changed camera file invalidated resume, and a pause marker split
across process-output chunks was still recognized.

## Correctness gate

The isolated branch must pass both before publication:

1. an uninterrupted run with no pause request must produce byte-for-byte the
   same scientific result files as pristine OpenLPT; and
2. a paused-and-resumed run must produce byte-for-byte the same final
   scientific result files as the uninterrupted candidate run.

No numeric-tolerance fallback is permitted.

The earlier integrated implementation already passed a 10-frame uninterrupted
versus paused-and-resumed comparison with zero SHA-256 differences across all
15 scientific result files. A clean build and repeat of that exact gate remain
required for this isolated PR branch before it is pushed.
