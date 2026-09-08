# LumaCam processing pipeline

This package provides selectable processing sections for current and earlier
LumaCam proposals, legacy run folders, and flat TIFF datasets. New proposals
are recognized from `data/experiments`; `metadata/layout.json` is supported
only for old data and is never required or generated.

See the concise [user manual](user_manual.md) for routine processing.

## Start in Jupyter

Open `lumacam_pipeline/lumacam_pipeline.ipynb` from the toolbox, or open
`lumacam_pipeline.ipynb` after entering this folder, and run both code cells.
The dashboard has independent, unnumbered sections. Discover the project
first; after that, run only the capabilities needed for the dataset.

Typical ex-situ use:

1. Choose the `exsitu` preset and discover the data.
2. Run corrections if beam/reference/background inputs exist, or load the raw
   final TIFF products directly.
3. Load images, write an average, select panels, and export a figure or video.
4. Leave electrochemistry blank. Its section and only its dependent combined
   outputs remain unavailable.

Typical operando NCI use adds electrochemistry loading, image measurements,
synchronization, combined plots, and synchronized video. Without reference
normalization, image measurements are labeled as arbitrary intensity rather
than calibrated lithium thickness.

Beam-history loading accepts legacy AM/PM timestamps, the newer
`YYYY-MM-DD.HH:MM:SS` format, ISO 8601, and Unix timestamps at second through
nanosecond resolution. Sampling cadence is inferred automatically and long
timestamp gaps are reported before correction.

For version-2 data, beam integrals are calculated from the exposure-manifest
windows that actually contributed to each reconstructed TIFF. If upstream
reduction recorded a preserved failed tail in `processing_exclusions.json`,
that interval is omitted from the beam integral as well as from the image. The
correction factor combines the included interval's average beam level with its
fraction of the run's planned effective exposure, so a shortened TIFF is also
normalized for its actual beam dose. The beam log records total beam integral,
per-TPX3 integrals, included/input/excluded counts, excluded filenames, and both
metadata paths. Inconsistent exclusion metadata stops correction instead of
silently using a mismatched interval.

When reference/OB beam history is supplied, reference TIFFs receive the same
beam-dose and effective-exposure normalization before spatial processing and
averaging. Batch reference caches include the reference beam-history signature
and exclusion/timing metadata so changed inputs cannot silently reuse an older
normalization. Whole-run exclusions apply to normal image loading, ex-situ
discovery and reference processing, including when old TIFFs remain on disk.

Experiment-wide `_sumAllRuns.tiff` products are discovered as aggregate
products and excluded from time-series processing by default. They can be
included explicitly in the project section for overview work.

The project panel can report proposal-wide, current-experiment, or selected-
batch acquisition ranges from JSON metadata only. It displays UTC and local
times plus a padded beam-history download window and can save the per-
experiment summary to `analysis/proposal_summary/metadata_datetime_ranges.csv`.
It never uses raw TPX3 file timestamps.

For repeated processing, the notebook's **Batch processing** section validates
an explicit experiment selection made with separate unselected and selected
lists, reuses one prepared reference/OB, runs jobs sequentially, continues
after individual failures, and resumes from completion sentinels. Batch status
and logs are stored under `analysis/proposal_summary/batches/`; experiment
outputs remain separate.

Technical terms in every dashboard section have clickable, concise inline
explanations.

## Command line

Inspect section readiness:

```bash
cd /home/localadmin/Programs/toolbox/lumacam_pipeline
PYTHONPATH=src python3 -m lumacam_pipeline --config lumacam_pipeline_config.ini status
```

Run selected sections in an explicit order:

```bash
PYTHONPATH=src python3 -m lumacam_pipeline \
  --config lumacam_pipeline_config.ini \
  run project corrections images image_measurements selection image_plot
```

Run or resume a validated notebook-generated batch recipe:

```bash
PYTHONPATH=src python3 -m lumacam_pipeline \
  --config /proposal/analysis/proposal_summary/batches/<batch-name>/batch_recipe.ini \
  batch
```

For ex-situ data without corrections, omit `corrections`:

```bash
PYTHONPATH=src python3 -m lumacam_pipeline --preset exsitu --data /path/to/images \
  run project images average selection image_plot
```

## Built-in sections

- Project and data discovery
- Metadata-only acquisition date-range inspection
- Beam/reference correction
- Background correction
- Image loading and spatial/temporal merging
- Average image
- Generic ROI image measurements
- Optional NCI plating metrics
- Exact-photon-count hyperspectral cube and feature maps
- Reconstructed time-of-flight cube feature maps and global spectrum
- Independent electrochemistry loading
- Image/electrochemistry synchronization
- Image selection
- Image-only and combined figures
- Individual figures for every loaded image
- Image-only and synchronized videos

Each section declares its required artifacts. Missing optional inputs disable
only the section and its direct consumers; they do not stop unrelated image
processing.

The time-of-flight section accepts an already reconstructed 3-D TIFF, NPY, or
NPZ cube. It does not parse raw TPX3 packets; raw-event reconstruction remains
an upstream acquisition/reduction responsibility.

The Python implementation is under `src/lumacam_pipeline`, and tests are under
`tests`.
