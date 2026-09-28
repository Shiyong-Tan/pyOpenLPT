# Runtime LPT output and camera selection

## Purpose

The Tracking view can now select an LPT output folder and choose between the
original `camFile` cameras and `camFile_VSC` cameras for an individual run.
These choices do not rewrite the project's `config.txt`.

## Runtime configuration

When Run OpenLPT is pressed, the GUI copies `config.txt` to the ignored
`.openlpt_runtime_config.txt` file and changes only two sections in that copy:

- `# Camera File Path` points to sorted `cam*.txt` or `vsc_cam*.txt` files.
- `# Output Folder Path` points to the selected output directory.

Every other setting, including frame range, object configuration, image lists,
tracking parameters, and existing resume configuration, is preserved exactly.
Camera intensity limits are also retained by camera index.

The run log records the selected camera source, selected output folder, and the
location of the unchanged master configuration.

## Validation and failure behavior

Before launching OpenLPT, the GUI verifies that it can read the configured
camera count and that the selected camera folder contains exactly that many
camera files.  It then runs the existing project-file checks against the
runtime configuration.  A missing or malformed selection stops before the
backend process starts.

The track statistics loader follows the selected output directory, so the
visualization does not silently continue reading the project's old `Results`
folder after a custom-output run.

## Verification

A two-camera fixture exercised both original-camera and VSC-camera modes.  The
test confirmed that:

- camera paths changed to the requested source;
- per-camera intensity values remained unchanged;
- the chosen output path was written to the runtime copy; and
- the master `config.txt` remained byte-for-byte unchanged after both runs.

This is a run-configuration/UI change.  It does not alter LPT numerical code or
scientific result generation.
