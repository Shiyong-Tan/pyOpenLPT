# Hash-bound VSC provenance tags

## Purpose

A VSC camera file alone cannot say which acquisition, camera baseline, settings,
or frames produced it. Folder names such as `T1` and `T3` are user conventions,
not reliable scientific identity. This change writes
`camFile_VSC/vsc_provenance.json` after a successful VSC run.

## Recorded fields

The manifest records:

- absolute source-project path and a human-readable source label;
- requested VSC frame bounds, when supplied;
- minimum/maximum correspondence frame and number of distinct contributing
  frames;
- VSC sampling and tolerance settings;
- number of valid correspondences;
- hashes of every input camera file;
- hashes of every generated VSC camera file;
- a name-independent acquisition fingerprint; and
- a fingerprint specific to the contributing frame interval.

The file is written to a temporary sibling and atomically replaced only after
JSON serialization succeeds. The VSC log prints the source label, actual frame
span, and manifest path so the user knows that tagging occurred.

## Dataset identity

SHA-256 converts file bytes into a 256-bit content digest. Here it is used as a
change detector and identity component: identical bytes produce the same
digest, while edited image or camera bytes overwhelmingly produce a different
one. It does not infer whether two different experiments are physically
equivalent.

The acquisition fingerprint hashes five distributed raw-image samples per
camera plus each image-list length. Paths and filenames are excluded from the
canonical payload, so copying or renaming the same acquisition does not change
its identity. The interval fingerprint hashes distributed samples at fixed
frame positions inside the VSC span; it remains comparable when another folder
contains the same acquisition with a different total list length.

This is deliberately a lightweight identity check, not a hash of all 49,932
frames. A later quality-validation step uses the tag to bind its report to the
unchanged VSC cameras and to select its validation frames reproducibly.

## Frame filtering

`VSCService.set_params` now accepts optional inclusive `frame_start` and
`frame_end` bounds. Invalid reversed bounds fail immediately. With both values
omitted, track loading and VSC computation are unchanged; the manifest still
records the actual correspondence-frame span observed by the service.

## Validation

- Python syntax compilation passed in the LPT environment.
- Two synthetic acquisitions with different folder, list, and image filenames
  but identical bytes produced identical acquisition and interval digests.
- Changing the selected interval changed its digest.
- A generated manifest correctly recorded requested range 1--2, actual range
  1--2, two unique frames, three correspondences, and hashes for both generated
  VSC camera files.
- Reversed range 3--2 raised `ValueError`.
- Recomputing the acquisition fingerprint for T3 matched the digest already
  recorded in T3's existing provenance file.

With no optional frame bounds, this adds metadata only and does not modify the
VSC optimizer or camera calculations.
