# Operando Dev Pipeline Handbook

> This development runner is retained for notebook compatibility. The
> maintained section-based entry point is `toolbox/lumacam_pipeline.ipynb`, and
> the established calculations are shared from `toolbox/lumacam_pipeline`.

The notebook GUI in `operando_pipeline.ipynb` runs the NCI
correction, inspection, plotting, selection, and video workflow. It supports
both the former campaign layout and the version-2 layout produced by
`lumacam_measurementcontrol`.

The notebook still locates `toolbox/operando_dev` automatically, but its small
compatibility modules import the shared LumaCam backend.

## Supported NCI layouts

The preferred proposal layout is:

```text
<proposal>/
  data/experiments/
    tpx3Files/<experiment>/<run>/*.tpx3
    final/<experiment>/<photon-range>/<run-named TIFFs>
    derived/<experiment>/<run>/*.tiff
    metadata/<experiment>/<run>/tpx3_exposure_manifest.json
  analysis/
```

The early July 2026 `data/experiment/` spelling and marker-era proposals remain
readable. New proposals use `data/experiments/` and do not require a
`layout.json` file.

For `exp_dir` and `reference_dir`, select either
`data/experiments/tpx3Files/<experiment>` or
`data/experiments/final/<experiment>`. The toolbox resolves the matching raw
and final run directories automatically. It also accepts a proposal containing
one experiment, either version-2 data root containing one experiment, or one
specific run below `tpx3Files`/`metadata`, or a photon-range folder below
`final/<experiment>`.

The legacy layout remains supported:

```text
<campaign>/
  experiment/<experiment>/<run>/tpx3Files/*.tpx3
  experiment/<experiment>/<run>/final/*.tiff
  reference/<reference>/<run>/tpx3Files/*.tpx3
  reference/<reference>/<run>/final/*.tiff
```

In version 2, multiple filtered TIFF variants in one run are reconstructed from
the same complete TPX acquisition. They therefore share the run time and beam
interval. In the legacy layout, `n_tpx_per_image` continues to control the
sequential TPX-to-image grouping.

For version-2 acquisitions, timing is resolved in this order:

1. `metadata/<experiment>/<run>/tpx3_exposure_manifest.json` supplies exact
   per-frame windows, including the short final partial TPX3 frame.
2. `metadata/<experiment>/<run>/acquisition.json` supplies the recorded run
   start/end envelope when a detailed exposure manifest is unavailable.
3. `metadata/<experiment>/experiment.json` supplies the run envelope for early
   version-2 datasets whose per-run timing JSON is incomplete or absent.
4. Raw TPX3 modification times are the legacy fallback only.
5. TIFF metadata/file time is the last-resort image timestamp when no timing
   JSON and no raw TPX3 files remain.

Beam levels are duration-weighted across the selected windows. The `tpx_time`
setting is used only for the raw-file fallback; it does not override JSON
metadata. Processing therefore remains metadata-driven after raw TPX3 folders
are archived into experiment-level `.tar.gz` files.

## Optional inputs and correction behavior

Only `exp_dir` and a writable result location are required.

- Blank `beamhistory_path`: intensity correction is skipped and factor `1.0`
  is used.
- Blank `reference_dir`: OB/reference-image division is skipped.
- A reference image with blank beam-history paths: OB division still runs, but
  beam-intensity normalization is skipped.
- Blank `echem_folder`: the GUI enters NCI-only mode. It loads, previews,
  selects, and saves neutron-image panels without an EC trace; video export
  creates an NCI-only video.
- `background_processing.parameters.enabled = false`: background construction
  and subtraction are skipped, and downstream analysis uses the corrected
  images directly.

Without a reference image, the images remain relative NCI intensity maps. The
GUI labels their mean and color scale in arbitrary units and skips calibrated
lithium-mass/plating metrics. With reference normalization, the established
lithium-thickness and mass presentation is retained.

If experiment beam history is supplied, intensity correction also needs either
a usable reference beam integral supplied by code or both `reference_dir` and
`reference_beamhistory_path`. If those are unavailable, the GUI reports the
condition and continues without intensity correction.

## Configuration

Start from `operando_config.ini`:

```ini
[general]
base_dir = /path/to/proposal
echem_folder =
result_root = analysis
skip_pipeline_if_corrected_exists = true

[obint_correction.parameters]
exp_dir = data/experiments/final/experiment_name/3to1000
beamhistory_path =
reference_dir =
reference_beamhistory_path =
reference_bin_y = 2
reference_bin_x = 2
reference_final_subfolder = final
reference_thickness_um = 1.0
file_filter = 3to1000
tpx_time = 2
n_tpx_per_image = 30

[background_processing.parameters]
enabled = false
background_n_images = 7
recompute_background = false
recompute_final = false
clip_negatives = true
```

Relative paths are resolved against `base_dir`. In a version-2 proposal,
`result_root = analysis` keeps downstream results outside the immutable
acquisition tree.

Use `file_filter` when each run contains several reconstructed variants and
only one series should be analyzed, for example `3to1000`.

## GUI workflow

1. Select the experiment. Reference, EC, and beam-history paths may remain
   blank.
2. Inspect TPX ranges if beam histories will be used.
3. Validate any supplied beam-history CSVs.
4. Run correction. This always creates provenance copies and a timestamped
   processed TIFF stack; available intensity and OB corrections are applied.
5. Enable background correction only when the early corrected frames are a
   valid background for the experiment. Otherwise leave it disabled and run
   the background step once to confirm the corrected stack is selected for
   analysis.
6. Load the preview. With no EC folder, the GUI shows NCI-only evolution and
   image panels.
7. With EC data, detect turning points and use the interactive selector. With
   no EC data, the GUI chooses representative NCI frames and allows manual
   image indices.
8. Save final image panels. The potential plot is omitted in NCI-only mode.
9. Export a synchronized EC/NCI video or an NCI-only video.

## Results

```text
<result_root>/results_<experiment>/
  images/
    uncorrected_<experiment>/
    obint_corrected_<experiment>/
    final_<experiment>/             # only used when BG correction is enabled
  processing/
    obint_beam_log.csv
    processing_metadata.json
    background_<experiment>.tif     # only when BG correction is enabled
  output/
    selected_image_summary_<experiment>.txt
    potential_<experiment>.png      # omitted in NCI-only mode
    nci_<experiment>.png
    nci_<experiment>_3D.png
    <experiment>.mp4
    <experiment>_3D.mp4
```

The correction log is still written when intensity correction is skipped. Its
rows record factor `1.0` and `intensity_correction_applied = false`, while
preserving the image/run timestamps used by plotting.

## Reruns and troubleshooting

The GUI checks expected and existing output counts before reusing corrected or
background-subtracted stacks. Use the explicit recompute checkboxes when a
parameter change should rewrite an existing stage.

- No images found in version 2: select `final/<experiment>` or
  `tpx3Files/<experiment>`, and check `file_filter`.
- No timestamps in plots: check the run exposure manifest first, then the raw
  TPX3 folder. If neither exists, the pipeline falls back to TIFF/file times.
- No EC data: leaving the field blank is valid; use manual or representative
  NCI image selection.
- No background: leave background correction disabled. Downstream steps use
  `obint_corrected_<experiment>` directly.
- Changed imported modules while a notebook is open: restart the kernel or
  rerun the environment/import cell before testing again.
