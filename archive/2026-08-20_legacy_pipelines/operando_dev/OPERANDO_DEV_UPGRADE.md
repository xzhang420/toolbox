# Operando Dev Upgrade Summary

> Compatibility note (2026-08-20): the maintained processing implementation
> now lives in `toolbox/lumacam_pipeline`. The files in `operando_dev` are
> compatibility runners/imports for existing notebooks; use
> `toolbox/lumacam_pipeline.ipynb` for new work.

Date: 2026-08-19

This document records the development upgrade that preceded the shared
LumaCam pipeline. Its correction and plotting backends have since been merged
into the common package.

## What changed

- Version-2 proposal discovery accepts both layouts used by
  `lumacam_measurementcontrol`:
  - `<proposal>/data/experiment/` (the July 2026 Leon proposal)
  - `<proposal>/data/experiments/`
- Final TIFFs under `final/<experiment>/<photon-range>/` are paired with their
  run metadata under `metadata/<experiment>/<run>/`.
- Timing selection is metadata-first:
  1. exact frame windows from `tpx3_exposure_manifest.json`;
  2. the run envelope from `acquisition.json`;
  3. the matching run entry in `experiment.json`;
  4. legacy raw-TPX3 modification-time fallback;
  5. TIFF metadata/file-time fallback when no other timing survives.
- The beam log and processing metadata record the selected timing source and
  JSON paths for auditability.
- Blank beam-history paths are supported. Intensity correction is skipped and
  a factor of `1.0` is recorded; reference-image division can still run if an
  open-beam/reference image is supplied.
- Blank electrochemistry paths are supported. The GUI uses chronological NCI
  time, plots the NCI metric against elapsed time, selects representative or
  manually entered image indices, omits the potential plot, and exports NCI-only
  2D and 3D videos.
- When no reference image is supplied, plots and summaries use mean NCI
  intensity in arbitrary units. Lithium-thickness/mass labels and plating
  metrics are enabled only after reference normalization is recorded.
- The notebook and GUI explicitly load sibling modules from `operando_dev`, so
  they cannot silently import the production `operando` backend.

## July 2026 example configuration

```ini
[general]
base_dir = /data01/Leon_BOA_July2026_P20260948
echem_folder =
result_root = analysis

[obint_correction.parameters]
exp_dir = data/experiment/final/exp101_GPE_Cell1_20x20NA_1.4LensAperture/3to1000
beamhistory_path =
reference_dir =
reference_beamhistory_path =
file_filter = 3to1000
```

Leaving all three optional correction paths blank produces timestamped,
otherwise unchanged float TIFFs for downstream NCI inspection. Set
`reference_dir` without beam histories to apply reference-image division only.

## Validation performed

- Python compilation passed for all five development modules.
- Seven focused tests pass in `operando_dev/tests/test_metadata_timing.py`.
- Proposal-root discovery found 25 selectable experiments in
  `/data01/Leon_BOA_July2026_P20260948/`.
- `exp001_OB1um`: 30/30 images used detailed exposure manifests (31 TPX3 frame
  windows per run, including the partial frame).
- `exp101_GPE_Cell1_20x20NA_1.4LensAperture`: 20/20 available `3to1000` images
  used `acquisition.json` after raw run folders had been archived.
- A headless NCI-only end-to-end run completed on those 20 `exp101` images with
  no beam history, reference image, background correction, or electrochemistry.
  It produced corrected and provenance TIFF stacks, the beam/provenance log,
  processing metadata, selected-image summary, 2D/3D NCI panels, and 2D/3D MP4
  videos. The NCI-only outputs used intensity units and suppressed automatic
  electrochemical cycle labels. No potential plot was produced, by design.

## Running the checks

From `/home/localadmin/Programs/toolbox`:

```bash
python3 -m py_compile operando_dev/*.py
python3 -m unittest discover -s operando_dev/tests -v
```

Start from `operando_dev/operando_config.ini` and
`operando_dev/operando_pipeline.ipynb`. If the notebook is
copied to another computer, set `OPERANDO_DEV_DIR` to the folder containing the
development Python modules before running the setup cell.
