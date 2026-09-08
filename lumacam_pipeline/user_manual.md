# LumaCam pipeline user manual

## Start

1. Open `lumacam_pipeline.ipynb` and restart the kernel if it was already open.
2. Run both cells to open the dashboard.
3. Choose a preset:
   - `operando_nci`: images, NCI processing, and optional electrochemistry.
   - `nci_images_only`: NCI images without electrochemistry.
   - `exsitu`: independent sample images.
   - `image_only`: generic TIFF processing without corrections.
   - `tof`: an already reconstructed time-of-flight cube.

## Select the data

In **Project and data**:

1. Click **Browse** beside **Data**, navigate to the proposal or image folder,
   and click **Use current folder**. The browser shows files on the computer
   running the Jupyter kernel (Ubuntu folders on Ubuntu, Windows drives on
   Windows).
2. Enter the short **Experiment** ID, such as `exp401`, or choose a discovered
   full name from the editable menu. The pipeline resolves a short ID and
   shows the full name after discovery.
3. Choose or enter the TIFF **Variant**, such as `3to1000`.
4. Leave **Results** blank to use the automatic
   `analysis/<full-experiment-name>/` folder.
5. Click **Discover data**.

Acquisition timing is read automatically from the run metadata JSON files.
Experiment-wide `_sumAllRuns.tiff` files are excluded unless **Include
all-runs aggregate products** is selected.

### Find the beam-history download window

In **Project and data**, choose a **Time-range scope**:

- **Entire proposal** for one file covering the full campaign.
- **Current experiment** for one experiment or OB/reference measurement.
- **Selected batch experiments** for only the experiments selected in the
  batch panel.

Set a small download margin (five minutes is the default) and click **Inspect
metadata date range**. The dashboard shows the exact UTC and local ranges, a
padded suggested beam-history window, and a per-experiment table. It reads
`tpx3_exposure_manifest.json` first, then falls back to `acquisition.json` or
`experiment.json`; raw TPX3 files and their timestamps are never inspected.

Click **Save date-range CSV** to write
`analysis/proposal_summary/metadata_datetime_ranges.csv`.

## Image products

Run this section after **Discover data** has completed:

1. Keep **Frames/merge** at `1` to retain every time point. A larger value
   averages that many consecutive images into one frame.
2. Keep **Spatial merge** at `1` to retain the original pixels. A larger value
   averages blocks of that size, such as `2` for 2×2 pixels.
3. Click **Load images**. This reads the TIFFs and their JSON acquisition times
   into the current notebook session; it does not alter the source data.
4. Click **Write average** only if an experiment-average TIFF is wanted.

For raw or already reconstructed final TIFFs, load images directly. If
corrections are required, run **Corrections** first and then return here and
click **Load images** so the corrected stack is loaded.

## Other processing

- For correction, supply any available beam-history and reference paths, then
  click **Run corrections**. Missing optional paths do not stop processing.
- If reduction preserved but excluded a failed final TPX3 frame, the pipeline
  reads `processing_exclusions.json` automatically and omits the same interval
  from the beam integral. It also uses the included/planned exposure fraction
  in the correction factor, preventing a shortened TIFF from being treated as
  a full-duration image. The correction CSV records the total and per-TPX3 beam
  integrals, exposure fraction, and included/excluded frame details.
- Reference/OB TIFFs are normalized in the same way before they are averaged
  whenever a reference beam-history file is supplied.
- Runs marked as whole-run exclusions are omitted automatically, even if stale
  derived TIFFs still exist. Their raw TPX3 files remain retained.
- Enable and run background correction only when needed.
- After loading images, optionally click **Measure images**.
- Select images and save an image figure or video from the output section.

For operando work, load electrochemistry after measuring images, then run
synchronization and event detection before creating combined plots or videos.
Leave electrochemistry blank for ex-situ or image-only work.

### Beam-history timestamps

Beam-history files need timestamp first and numeric beam value immediately
after it. Accepted timestamps include:

- `07/20/2026 12:00:35 PM`
- `2026-07-20.12:00:35`
- ISO 8601, such as `2026-07-20T12:00:35` or a value with `Z`/UTC offset
- Unix seconds, milliseconds, microseconds, or nanoseconds

Comma, semicolon, tab, and whitespace-separated files are accepted. Human-readable
timestamps without a UTC offset are treated as local time. Absolute ISO/Unix
timestamps are converted to local time for comparison with beam history. Sampling
interval is inferred automatically; warnings identify invalid beam values,
duplicate timestamps, and long gaps. Each beam value is treated as the average
from its timestamp until the next timestamp.

## Batch processing

Use **Batch processing** when several experiments need the same settings and
reference/OB:

1. Select the proposal under **Project and data** and fill in the shared
   correction, image, ROI, and output settings.
2. Open **Batch processing**, optionally enter a name/filter, and click
   **Discover batch experiments**. Setup, OB, test, background, and focus data
   are hidden by default; no experiment is selected automatically.
3. Highlight experiments in **Currently unselected**, then click **Add
   highlighted →**. They move to **Selected**. Use **Add all shown →** for the
   filtered list; highlight items on the right and click **← Remove
   highlighted** to remove them. Changing the filter does not remove experiments
   already selected.
4. Select the intended processing sections. Prerequisites are added
   automatically.
5. Click **Validate batch**. Resolve every invalid row before continuing.
6. Click **Run or resume batch**. Experiments run sequentially; one failure is
   recorded without stopping later experiments. A repeated run skips jobs
   already completed with the same recipe and unchanged source inputs unless
   **Re-run jobs** is selected.

The shared reference is prepared once per reference path, TIFF variant, and
binning combination. Batch recipes, logs, and status tables are saved under
`analysis/proposal_summary/batches/<batch-name>/`. Each experiment still gets
its own `analysis/<full-experiment-name>/lumacam_pipeline.ini` and outputs.

For a long terminal run, first save/validate the recipe in the notebook, then
run it from this project folder:

```bash
PYTHONPATH=src python3 -m lumacam_pipeline \
  --config /proposal/analysis/proposal_summary/batches/<batch-name>/batch_recipe.ini \
  batch
```

## Individual image figures

After **Load images**, use **Save individual image figures** to export every
loaded image, or enter selected indices only. Files are written to
`figures/individual_images/` with the experiment name, five-digit image index,
and source TIFF name. **Same scale for all** is recommended for comparison;
**Scale each image** emphasizes contrast within each frame.

## Section states

- **not configured**: the optional section is disabled or has no path.
- **waiting for input**: run the prerequisite section shown under missing artifacts.
- **ready**: the section can be run.
- **completed**: the last run succeeded.
- **failed**: read the **Details** column or output box, correct the reported
  problem, and run the section again.

## Configuration and outputs

Whenever a section is run for a recognized proposal, the dashboard automatically
creates or updates
`analysis/<full-experiment-name>/lumacam_pipeline.ini`. After **Discover data**
succeeds, the **Config** field switches to this proposal-local file. Every later
section run saves the current GUI settings to it before processing starts, so
there is no need to return to **Project and data** or click **Save config**. Its
path is also shown in **Project and data** and in the output box. Use **Load
config** to restore it in a later notebook session.

Use the separate **Config**, **Load config**, and **Save config** controls when
an additional manually named configuration file is wanted.
Every path field has a **Browse** button; file fields select files, while
reference, background, and electrochemistry fields accept a file or folder.
Outputs are organized beneath the experiment folder as `processing/`,
`images/`, `tables/`, `figures/`, and `videos/`. Each directory is created when
its first output is written. The section-status panel shows which actions are
ready and which optional inputs are missing.

Blue terms marked with an information symbol are clickable. Click one to show
or hide its concise explanation directly in the dashboard.
