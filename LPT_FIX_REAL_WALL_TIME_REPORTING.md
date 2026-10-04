# End-to-end LPT wall-time reporting

## Problem

OpenLPT's C++ log reports durations using `clock()`. In a multithreaded run,
that is accumulated processor time rather than the elapsed time experienced by
the user. It also does not consistently include GUI/backend startup, image
input, result serialization, or shutdown. Treating that value as seconds per
frame therefore gives misleading performance estimates.

## Change

The tracking GUI now starts a monotonic `time.perf_counter()` timer immediately
before launching the backend and writes a summary only after the process has
finished. The summary includes:

- elapsed `HH:MM:SS.sss` and raw seconds;
- the number of completed frame reports;
- effective wall seconds per completed frame; and
- an explicit statement that startup, image input, LPT computation, result
  output, and shutdown are included.

User-terminated runs are labeled separately. Completed frames are parsed from
the existing `Total time for frame N:` messages with a carry buffer, so a line
split across arbitrary `QProcess` output chunks is counted exactly once. A set
of frame IDs also prevents duplicate log lines from inflating the count.

The final summary is appended before the log file closes, so it appears both in
the GUI and in `log.txt`.

## Validation

- Python syntax compilation passed.
- A chunk-boundary test split one frame message across two reads and left a
  second message unterminated until final flush; the completed set was exactly
  `{3, 4}`.
- A simulated successful process exit reported 10 seconds total, two completed
  frames, and 5 seconds per frame, including the end-to-end scope statement.

This is instrumentation only. It does not modify input configuration,
scientific computation, or result files.
