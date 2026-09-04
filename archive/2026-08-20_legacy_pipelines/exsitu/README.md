# Ex-Situ NCI Pipeline

> The primary ex-situ interface is now the `exsitu` preset in
> `toolbox/lumacam_pipeline.ipynb`. The imports and old notebook launchers below
> remain compatible and call the same shared backend.

The ex situ pipeline copies source TIFFs, applies whichever corrections have
inputs, and writes an averaged image. It accepts flat TIFF folders, the legacy
per-run `tpx3Files/` + `final/` organization, and the version-2
`data/experiments` organization with raw files grouped by run and final TIFFs
grouped by photon range.

```python
from exsitu import run_exsitu_pipeline

result = run_exsitu_pipeline(
    data_dir="/proposal/data/experiments/final/sample",
    beamhistory_path=None,       # blank: skip beam-intensity correction
    ob_path="/proposal/data/experiments/final/open_beam",  # optional
    bg_path=None,                # blank: skip background correction
    file_filter="3to1000",      # optional TIFF variant keyword(s)
)
```

Correction branches are independent:

- no beam history: factors remain `1.0` and processing continues;
- no OB path: reference-image division is skipped;
- no BG path: background subtraction is skipped;
- an external BG path: its images receive the same available beam/OB
  corrections, are median-combined, and are subtracted from the sample;
- `background_correction=True` with no `bg_path`: backward-compatible mode
  uses the first `background_n_images` corrected sample frames as background.

Version-2 inputs default to `<proposal>/analysis/results_<experiment>`. Flat
and legacy inputs keep the former default beside `data_dir`:

```text
results_<data folder>/
  uncorrected/
  obint_corrected/
  background_corrected_input/  # only for an external BG dataset
  background.tif               # only when BG correction runs
  final/                        # only when BG correction runs
  <sample>.tif
```

Command-line example:

```bash
python -m exsitu.exsitu_pipeline \
  /proposal/data/experiments/final/sample \
  --ob /proposal/data/experiments/final/open_beam \
  --bg /proposal/data/experiments/final/background \
  --file-filter 3to1000
```

`--beamhistory` may be added when TPX timestamps and a matching beam-history
CSV are available. The notebook GUI is
`exsitu/exsitu_pipeline_gui_notebook.ipynb`.

For version-2 data, timestamps come first from
`metadata/<experiment>/<run>/tpx3_exposure_manifest.json`. Explicit exposure
windows are duration-weighted, including a short final partial frame. Raw TPX3
modification times and `tpx_time` are used only as a legacy fallback, so the
pipeline continues to apply beam-history correction after raw files are
archived as long as the manifest is retained.

When several keywords are supplied, for example
`"3to1000,5to1000"`, the pipeline writes one averaged TIFF per keyword rather
than combining different reconstruction variants into one image.
