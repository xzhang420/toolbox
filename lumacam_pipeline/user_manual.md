# LumaCam pipeline user manual

New proposals separate detector data in `<proposal>/raw/` from downstream products in
`<proposal>/reduced/`. Select the outer proposal and load
`reduced/analysis/lumacam_pipeline.ini`. In this manual, `analysis/`, `data/echem/`
and `data/beamhistory/` are relative to `reduced/` for these proposals. Earlier
proposal layouts remain supported.

## LumaCam/NCI workflow terminology

| Stage | Meaning | Responsible software |
| --- | --- | --- |
| **Data reduction** | Convert raw detector data into reconstructed products: pixel-to-photon conversion, photon-to-event conversion, image reconstruction, and related steps. | `lumacam_measurementcontrol`, including `pixel2photon`, `photon2event`, and associated reconstruction scripts. |
| **Data processing** | Correct and prepare reconstructed images: OB/open-beam correction, beam-intensity correction, reference normalization/calibration, background correction, white-spot removal, merging, and averaging. | Processing part of `lumacam_pipeline`, exposed through `lumacam_processing.ipynb`. |
| **Data analysis & visualization** | Extract scientific measurements from processed images, correlate electrochemistry, and create figures, tables, and videos. | Analysis/visualization part of `lumacam_pipeline`, exposed through `lumacam_visualization.ipynb`. |

NCI means neutron capture imaging; the same stage terminology applies to other
LumaCam modes such as time of flight. The two notebooks cover the last two stages;
raw detector reconstruction is upstream. Choose the sections you need, not every
step in a compulsory sequence. Analysis/visualization uses existing processed
data and does not rerun reduction or corrections.

Storage names do not define these stages: `reduced/` can contain reduction,
processing, and analysis outputs; `analysis/` also holds processing results.
Acquisition `final/` TIFFs are reconstructed images, not necessarily corrected
images. Historical measurement-control options such as `--defer-processing`
still refer to **data reduction**; their command spelling is unchanged.

## Start

1. Open **lumacam_processing.ipynb** for data processing. Restart the kernel if
   it was already open, then run its code cell.
2. Later, open **lumacam_visualization.ipynb** for analysis, figures and videos.
   It works independently; no processing notebook needs to remain running.
3. For a proposal created with the LumaCam setup routine, select
   `analysis/lumacam_pipeline.ini` under **Config** and click **Load config**.
   Data, electrochemistry and beam-history paths are already filled in.
4. Choose a **Processing mode**:
   - **Operando NCI**: time-series images, optional corrections and electrochemistry.
   - **Ex situ NCI**: independent samples, optional corrections, batches and averages.
   - **Time of flight**: an already reconstructed time-of-flight cube.

The old combined notebook and Harris-specific starter INI are archived under
`toolbox/archive/2026-10-29_combined_pipeline/`. Shared calculation modules remain
active. The sections below describe processing unless marked as visualization.

### Two independent workflows

**Processing:** Discover → your selected corrections/BG → **Load and publish
processed images** → optional **Preview processed image (QC)** / **Write average**.
Batch processing publishes the same outputs automatically. Publishing writes the
actual filtered, cropped and merged stack to `images/processed_frames/`, with
relative paths, checksums, optional pixel size and UTC acquisition times in
`processing/processed_dataset.json`. Do not edit that generated JSON by hand.
It is a dataset record, not a folder-layout configuration.

**Analysis & visualization:**

1. Browse to the experiment/variant result folder and click **Open images**.
   Alternatively select a plain TIFF/FITS frame folder or one image; no raw data
   or proposal layout is required. Default folder reading is non-recursive.
2. Set a profile name, such as `overview` or `publication`. Leave the output
   parent blank to use `<dataset>/visualization/<profile>/`, or select a writable
   location if the dataset is read-only.
3. Preview a frame, choose indices/colormap/colour limits, then make an image
   panel, individual figures or video. Blank figure indices choose representative
   frames; blank individual-figure indices export all frames. Choose image-sequence
   or elapsed-time image-video playback (details below).
4. Optionally measure an ROI. Operando mode also exposes electrochemistry,
   synchronization and combined outputs. ToF mode exposes reconstructed cube
   analysis. These tools do not change the input images.
5. Settings save before each action; use **Save settings** for edits without an
   action. Reopen a profile with **Load settings**. Each successful action keeps
   a separate settings/export record. New GUI profiles default to **Simple folders**:
   current files refresh directly in `figures/`, `videos/` and `tables/`, with
   settings/history in `records/`. **Keep every export in dated folders** retains
   old exports too; existing profiles keep their saved choice. The INI option is
   `[visualization] export_layout = flat` or `versioned` (the compatibility API/CLI
   default). Use another profile for a deliberate alternative analysis.

Choose **Open images** again after changing input decoding, source or profile.
An unknown pixel size means no physical scale bar. Set it only from known
calibration. Missing times still permit image-only work; synchronization requires
real timestamps. For timestamps without an offset, supply a timezone such as
`Europe/Zurich` or `UTC`. An explicit acquisition start and interval can supply
timing for a known uniformly spaced series, never merely to hide missing data.
FITS with multiple image extensions requires its HDU index/name; numeric scaling
and stored pixel orientation are preserved. Advanced section settings accept JSON
objects, e.g. `{"synchronization": {"tolerance_seconds": 30}}`.

For older processed TIFFs, stored image timestamps are accepted with an explicit
timezone. Fully timed plain folders are sorted chronologically, including resumed
runs whose filenames sort differently. Published manifests retain their recorded
order. Without timestamps, images remain usable for figures/video, but not timed
correlation. The pipeline does not invent missing acquisition metadata.

Electrochemistry accepts one TXT/CSV file or a folder. Select the file(s) for the
same cell: a proposal-wide folder may contain unrelated measurements. Legacy
`DD/MM/YYYY HH:MM:SS.ffff` dates are supported; set the electrochemistry timezone
separately. Synchronization reports matched/unmatched counts and does not extend
electrochemistry beyond its recorded date range in the visualization notebook.
Set `tolerance_seconds` to limit matches across internal gaps too. Nonoverlapping
image/electrochemistry date ranges stop combined outputs rather than correlating
unrelated measurements.
Combined videos default to the shared recorded time range, without holding an
electrochemistry endpoint across images acquired outside that range.
Uncalibrated inputs default to image intensity plus potential, without overlaying
electrochemical thickness on an arbitrary-intensity axis. Known micrometre units
or a calibrated processed manifest allow comparing both thickness traces.
The GUI's **Include electrochemical thickness** checkbox selects this explicitly;
the API/CLI default enables both for known thickness units. Arbitrary image counts
cannot be compared with EC thickness on one value axis.

### Thickness methods, regions and CSV exports

Use **lumacam_visualization.ipynb**; restart its kernel after updating the code.
No reduction or image corrections are rerun.

1. Open the processed images or load their visualization `settings.ini`.
2. In **Thickness / ROI measurements**, choose whole image, circle, rectangle or
   polygon, then **Preview measurement region**. Coordinates are original image
   pixels, independent of the display crop; rectangle upper edges are excluded.
   Polygon vertices must follow the perimeter without crossing.
3. Select methods (Ctrl/Cmd-click for multiple), choose a primary method and click
   **Measure images and save table**. Primary values appear on panels and videos.
4. For operando work, load the matching electrochemistry. Choose the methods to
   display under **Electrochemistry**, optionally include electrochemical thickness,
   then **Make combined figure**. Videos use the primary image method.
5. Under **CSV exports**, select full experiment or selected window, then
   **Export CSV tables**. Without electrochemistry, only the image table is written.

| Method | Calculation and required settings |
| --- | --- |
| ROI mean (default) | Average of finite pixels in the selected region. |
| Electrode-area mean | `sum(pixel thickness × pixel area) / electrode area`. Requires thickness in µm, pixel size in µm and electrode area in mm². The region must capture the material whose volume you intend to normalize; area is never guessed. |
| Active-area mean | Average of finite pixels strictly above your explicit threshold, in image units. Also exports active area and coverage. Threshold choice affects the result. |
| ROI median | Middle finite pixel value; not a volume-conserving average. |
| ROI trimmed mean | Remove the specified fraction from **each** tail, then average; default 0.1 means 10% low + 10% high. Not a volume-conserving average. |

Negative values are retained, not clipped. Nonfinite pixels are counted and flagged;
electrode-area thickness is left blank if any region pixels are nonfinite, instead
of treating missing material as zero. Empty active areas produce blank means.
Zero frames remain in the tables with a quality flag. Physical area fields stay
blank when pixel size is unknown. These methods summarize calibrated thickness;
they do not turn uncalibrated image counts into thickness.

Exports go in the profile's existing `tables/` (or dated table folders if enabled):

- `thickness.csv`: frame/source filename, UTC timestamp, original elapsed hours,
  each selected method (`nci_*_um` for thickness; `image_*_value` otherwise), region,
  valid/active areas and counts, active coverage, primary method and quality flags.
- `electrochemistry.csv`: native timestamps, elapsed hours, potential (V), current
  (mA, if available), recorded capacity (mAh), cycle and electrochemical thickness
  (µm), using the saved capacity conversion/mode. No new calibration is assumed.
- `thickness_comparison.csv` (optional): one row per image, nearest measured EC
  point within the matching tolerance (default 30 s), `matched` and `time_offset_s`
  (image time minus EC time). No interpolation or extrapolation; unmatched values
  stay blank, including across gaps larger than the tolerance.

Window-only exports use `_window` filenames, preserving full-export files. Image
and EC native tables retain their different sampling rates. Matching is performed
on the full loaded EC record before selecting window rows. Thus an accepted match
can lie just outside the chosen window; its timestamp/offset remain explicit.
Editable settings and per-export snapshots record methods, thresholds and windows.

### Time window / zoom

Enable **Time window / zoom**; enter elapsed hours from the first image or absolute
ISO dates with UTC offset (for example `2025-12-07T14:00:00+01:00`). Naive dates
need the input timezone. The hours slider is an alternative to typing; blank ends
use experiment bounds. Click **Apply time window**, then remake the desired outputs.
**Reset to full experiment** disables the window; remake outputs to refresh them.

Choose independently whether the window applies to time-series figures, image
panels/individual figures and videos. CSV range is selected separately. The window
never crops source TIFFs, remeasures a shortened stack, resets the time origin or
recalculates electrochemical thickness/cycle numbering. Event detection uses the
full record. Image panels retain original letters; if none of the selected frames
fall inside the window, select frame indices inside it. No timestamps means no
time-based zoom, but image measurements and image-only CSV exports still work.
With simple output folders, rerendering refreshes the current figures/videos;
choose another profile if you want to keep both full and zoomed figures.

### Plating inhomogeneity

In either NCI mode, choose the electrode region under **Thickness / ROI
measurements**, then open **Plating inhomogeneity** and click **Measure plating
inhomogeneity**. It works without electrochemistry and writes a full-frame CSV
and a time-series figure under the profile's existing `tables/` and `figures/`.
The time-window settings affect the figure, not the full CSV. If timestamps are
missing, the figure uses frame indices.

- **ROI standard deviation and CV** describe absolute and relative variation;
  CV is standard deviation divided by mean. P90 − P10 is a robust spread.
- With a circle, **edge − centre** compares the outer region (radius ≥ 0.8R)
  with the central region (radius ≤ 0.5R). **Centroid offset / R** uses positive
  image values as weights; it describes off-centre signal, not its cause.
- Set the **Active threshold** above to calculate coverage and active-region CV.
  The GUI shares this threshold with the active-area thickness method. Leave it
  blank when no defensible threshold is available. Set a minimum
  reliable mean signal only from independent baseline/noise evidence; frames
  below it have blank CV and centroid. Without a floor, CV near zero is unstable.

Nonfinite values are excluded and flagged. Negative values remain in the spread
and CV calculations; the centroid alone uses positive weights. A truncated circle
is flagged. Entirely zero ROIs remain in the CSV but break the trend lines and
are marked by dotted vertical lines. CV uses a labelled logarithmic scale when
its range exceeds eightfold; values in the CSV are unchanged. Noise, backgrounds, pixel scale, merging and threshold choice affect
these descriptions. Inspect images and use consistent settings for comparisons.
Formulas and geometry settings are saved in the profile/record settings; advanced
`inhomogeneity` options are `centre_radius_fraction`, `edge_start_fraction`,
`active_threshold` and `min_mean_signal`. Restart the notebook kernel after an
update. This does not alter processed images or rerun corrections.

### Quality, electrode geometry and cycle results

**Check image quality and save ROI diagnostic** writes `frame_quality.csv`,
`image_quality.json` and a measurement-region overlay. It flags empty/invalid
images and suspected dominant bright pixels without changing values. This check
also runs automatically before inhomogeneity. Red open markers identify suspected
bright outliers; inspect them before using a CV spike as a physical result.
Optional baseline indices summarize ROI mean and temporal variability. Marking
background-preparation frames as a baseline does not establish an independent
detection limit; no active threshold or reliable-mean floor is inferred.

For circular electrodes, enter the independently known **diameter (mm)** and
pixel size, then **Estimate centre from fixed electrode diameter**. The algorithm
searches temporal boundary contrast at the known radius. Registration smoothing
does not filter scientific images. A weak fit or a candidate near the search
limit leaves the previous centre in place. Review the overlay and remeasure after
changing geometry. Confirm the boundary only when supported independently of the
bright deposited area; inactive electrode areas must not be removed to improve CV.

Inhomogeneity also exports `radial_profiles.csv`, `radial_profiles.png` and
`roi_sensitivity.csv`. Annular means show where signal changes. Radius factors
0.9/1/1.1 are a sensitivity diagnostic and do not change the measurement ROI.

After measuring images and detecting events, click **Summarize cycle endpoints
and residual signal**. `phase_endpoints.csv` records each image/event timestamp,
mean/primary value, change from initial observed OCV, CV/spatial metrics when
calculated, matching offset and interrupted/incomplete status. The companion
figure shows endpoint trends. These values do not identify the chemical origin
of residual signal or measure coulombic efficiency. Phases are not automatically
paired into cycles when the protocol contains short reversals or interruptions.

EC thickness uses increments of the instrument's nonnegative **per-half-cycle
capacity counter**, adding at negative current and subtracting at positive
current. Rest holds net charge; counter resets and independent file restarts
do not reset it. Default current deadband is 0.01 mA (`electrochemistry.zero_current_ma`).
Negative net values are retained. The initial absolute Li inventory and the
capacity-to-thickness calibration are separate scientific inputs. A cumulative
signed-charge column has different semantics: do not treat it as a half-cycle
counter. The native CSV includes source filenames and net capacity for audit.

Combined figures include current by default (`combined_plot.show_current`).
Image and EC traces break on their own sampling gaps; an **EC data gap** does
not imply missing images. Synchronized videos leave an image unavailable when
the nearest acquisition is farther than `combined_video.image_tolerance_seconds`
(default 90 s), while continuing any valid EC trace. Supplied manifest fields
`exposure_start_utc`/`exposure_end_utc` constrain endpoint selection to a full
exposure within the phase; older images retain representative-time matching and
an explicit timing warning. Comparison CSVs explain unmatched rows in `match_status`.

### Signed background and revisiting legacy images

The processing background panel exposes **Clip negative BG values to zero**.
Clear it for signed quantitative analysis; clipping changes low-signal
means, percentiles and CV. Existing saved recipes retain their choice. Visualization
preserves its inputs and cannot recover negatives already discarded upstream.

For retained beam/OB-corrected intermediates, create a new processing result with
the existing saved BG, without repeating beam/OB correction or raw reduction:

```bash
lumacam derive-background --input /existing/beam_ob_corrected \
  --background /existing/background.tif --output /new/experiment/3to9999 \
  --name experiment --product-variant 3to9999 --timezone Europe/Zurich \
  --pixel-size-um 55 --unit um
```

This command retains signed values and requires a new output folder. Optional
`--white-spot-absolute-threshold VALUE` applies the existing 3×3 median replacement
only above the chosen positive residual. Inspect before choosing a threshold;
`processing/derivation.json` records source hashes and every changed coordinate.
Entirely zero source acquisitions retain their timestamps as explicitly invalid
NaN frames, avoiding an artificial negative copy of the background. Reopen the
new published dataset in visualization. No source files are rewritten.

### Operando figures: OCV and half-cycle images

Measure images → load the **same cell's** electrochemistry → **Detect events** →
make the image panel and combined figure. Leave figure indices blank to use the
detected events, or enter explicit indices to override them. The two figures
use matching letters. Reports show the combined potential/thickness figure,
without a second standalone thickness curve. Image-only reports still show it.

Automatic detection prefers current: initial observed zero current, then each
sustained current half-cycle's last point. Same-sign rest/restart segments are
combined; segments shorter than 30 seconds are ignored. The selected image is
at/before the event, within 90 seconds and the same phase. Zero/nonfinite ROI
means are excluded **only from representative selection**, never from the data.
The last recorded point is not proof that a cycle completed. Check
`tables/.../detected_events.json` and `*_points.csv` for omissions and offsets.
Even a short opposite-current pulse separates distinct same-sign phases;
irregular sequences use phase names without inventing paired cycle numbers.
Matching uses representative image timestamps, not full exposure windows, so an
image close to a reversal may integrate both phases. Review the offsets.

In advanced settings, `event_detection` supports `method` (`auto`, `current`,
`capacity`), `zero_current_ma` (default 0.01), `min_phase_seconds`,
`match_tolerance_seconds`, and `plating_current_sign` (`-1` or `1`). Set the sign
only when known for the imaged electrode; otherwise labels state current polarity.
Without current, `auto` uses the older capacity-turning-point detection; this does
not establish plating/stripping identity. Manual indices remain available.

Panels have a shared colour scale, letters and a scale bar only on the first
image. `image_plot.columns = 0` chooses a grid automatically. An optional
`display_roi = {"x0": 221, "y0": 222, "r0": 160}` crops the **display only**;
it does not alter the measurement ROI or TIFFs. The same option is available
for videos. Default image orientation preserves array orientation (`upper`);
use `image_origin = lower` explicitly when matching older NCI figures.
Set `clean_layout = False` for diagnostic pixel axes/frame timestamps.

To share results, copy the experiment/variant result folder, including `images/`
and `processing/`; include desired visualization profiles and optional external
electrochemistry files. The receiver installs this package and opens the
visualization notebook. See README for installation and CLI commands.

### Processing modes

The mode filters the panels, buttons, batch choices and section-status rows.
In the processing notebook, modes show relevant project/data, shared OB/BG,
correction, image-product and batch tools. Analysis and rendering live exclusively
in the visualization notebook. There, ex-situ hides electrochemistry and spectral
tools; operando and ToF expose their corresponding analysis sections.

Use **Show all sections** to reveal advanced processing tools in any mode. It
does not bring visualization back into the processing notebook. Showing a tool
does not run it. All modes use the same processing engine; sections remain
independently runnable.

Switching modes keeps entered paths, correction settings and completed results.
Before selecting data, untouched product defaults can follow the mode (for example,
ToF selects derived products). Existing settings otherwise remain in place.
Changing actual processing inputs still invalidates affected results as usual.

Hidden batch choices are remembered, but only choices currently shown and
selected are saved as executable batch steps. They reappear when you return
to their mode or enable **Show all sections**. Check the selection before
running. The mode, advanced-view preference and batch choices are saved with
the config by the normal save/processing actions.

For already-corrected images, skip Corrections and load the TIFFs directly.
For operando images without electrochemistry, leave its path blank and omit
electrochemistry-related batch steps. No separate image-only modes are needed.
Older `image_only` configs open as Ex situ NCI; `nci_images_only` opens as Operando NCI.
Their saved settings and legacy defaults are retained, including disabled
corrections/electrochemistry and average-copy preferences. Saving uses the new
mode name; existing processing snapshots are not rewritten.

### Results overview: Markdown and PDF

After **Load and publish processed images**, **Write average**, and analysis/
visualization actions, the result/profile folder gets:

- `overview.md`: numerical results, settings, quality notes and actual plot/image
  links for AI or a Markdown reader. An AI must open the linked images to see them.
- `overview.pdf`: the same overview with plots/images embedded for human readers.
- `report_assets/`: small plot copies/previews used by the Markdown. Keep this
  folder beside the Markdown when sharing; the PDF can be shared alone.

Processing reports show representative stored frames and the saved average when
available. Visualization reports also show recorded ROI results, curves, image
panels and combined electrochemistry plots when available. Unavailable sections
are not fabricated. Latest successful section exports are included; their exact
settings are linked because different exports may use different settings.

Reports refresh in place; underlying versioned exports remain untouched. Do not
edit the generated reports manually. Report failures appear in the action message
and do not invalidate successful processing. Set `[reports] enabled = False` in
the INI to disable automatic refresh. **Update overview (MD + PDF)** in the
visualization notebook rebuilds reports from recorded outputs, not unsaved GUI edits.
GUI/CLI reports use the concise PDF layout with actual figures and phase-endpoint
tables; full settings and audit links remain in Markdown. Set `[reports] style =
detailed` or use `lumacam report --style detailed` for the longer report. For a
shared summary folder, set `combined_output = ..` and
`combined_profiles = ["exp701", "exp1101", "exp1301"]` under `[reports]` in each
profile. Subsequent successful actions refresh both local and combined overviews
once all named sibling profiles exist. No analysis is rerun by report assembly.

For a proposal-wide report, keep profiles in `analysis/visualization/exp.../`
and the combined report in `analysis/proposal_summary/`. In each profile use
`combined_output = ../../proposal_summary`, `combined_profiles_root = ../visualization`
and the desired `combined_profiles` list. Add more experiment IDs when ready;
these folders need no material-specific group name.

To retain scientific interpretation through automatic refresh, place
`interpretation.json` beside the combined report. Schema version 1 contains
`source_profiles` (paths relative to the report), `evidence` (each file's `path`
and `sha256`), `scope`, `summary` (objects with `heading`, `observation`,
`conclusion`), optional `experiments` keyed by profile ID, `limitations`, and
`comparison_pages` (title, observation, conclusion, methods, figure path/caption).
The report highlights conclusions and embeds the comparison figures. Cite the
tables, settings and comparison figures used to support them. A changed source
set or evidence file withholds old interpretations with a visible review notice.
Scientific interpretation and comparison calculations are authored separately;
assembling the report does not infer causes or treat frames as replicated cells.

Processing batches also write a combined overview in the batch folder, including
job failures/skips. To report existing results or combine visualization profiles
without rerunning any pipeline sections:

```bash
lumacam report --input /path/to/result_or_profile
lumacam report --input /path/to/profile1 /path/to/profile2 --output /path/to/summary
```

Use `--title` and repeated `--note` options for context. A global option such as
`lumacam --variant 3to9999 report ...` records a user-confirmed variant in the
report only; it does not rename folders or rewrite source metadata.

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
   `analysis/<full-experiment-name>/<variant>/` folder. Leave **Analysis name**
   blank for the default result, or enter a label for an alternative recipe.
5. Click **Discover data**.

Acquisition timing is read automatically from the run metadata JSON files.
Experiment-wide `_sumAllRuns.tif[f]` files are excluded from sample, OB, BG,
correction, averaging, and batch processing. **Discover all-runs sums for
preview only** exposes them as `session.artifacts.require("aggregate_products")`
without adding them to processing inputs or input counts.

Individual TIFFs live in `final/<experiment>/<variant>/`. New experiment sums
live in `derived/<experiment>/summed/` with unchanged filenames; older sums in
the final folders remain discoverable. Aggregation manifests remain under
`metadata/<experiment>/summed/`. Both `data/experiment/` and `data/experiments/`
are supported without creating a `layout.json` marker.

Preview sums retain their original variant (for example `3to1000` or `5to15`).
Reference validation rejects datasets containing only sums. Shared OB cache keys
use individual reference TIFFs and exclusion/timing metadata, so relocating a
preview sum does not change the reference cache or the number of processed inputs.

### A different proposal layout

First use **Inspect layout (read-only)** or `lumacam --data /proposal inspect`.
The current default is the Harris `raw/` + `reduced/` structure. Older supported
layouts are detected automatically. If detection is unsuitable, set explicit
patterns in the processing GUI or INI, relative to **Data**:

```ini
[layout]
images_pattern = pictures/{experiment}/{variant}/*.tiff
metadata_pattern = timing/{experiment}/{run}
run_regex = (?P<run>.+_\d{5})_sumImage
```

Here `metadata_pattern` names a directory containing the existing timing JSONs;
`run_regex` extracts the run ID from each image filename. Omit the regex if
`{run}` is already a directory component in `images_pattern`. Inspect the matches
before processing. Custom layouts may use individually prepared OB/BG TIFFs;
automatic shared preparation expects a supported proposal layout. A new metadata
schema may require a reader change, not just different patterns. No layout.json
is needed or created. Never guess OB/BG assignments, units or timing from names.

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

## Prepare OB/BG files (optional)

**Skip this section if you already have processed, averaged OB and median BG
TIFFs.** Go directly to **Use prepared OB/BG and correct samples** below.

To create new files:

1. Select the proposal and TIFF **Variant** in **Project and data**. Supply
   **Beam history** there if beam normalization is needed; it should cover the
   OB, BG and sample acquisition times.
2. In **Prepare OB/BG files (optional)**, select the **OB experiment**.
3. If needed, enable **Also prepare a shared BG**, select the **BG experiment**,
   and set **BG frames for median**. This count is independent of a sample's
   first-frame background setting.
4. Leave **Preparation name** blank for your first preparation. Enter a new
   descriptive name, such as `day2`, when changing preparation settings and
   keeping the earlier files.
5. Click **Prepare OB file** or **Prepare OB and BG files**.

Preparation beam-normalizes OB frames when beam history is supplied, spatially
processes them, and averages them. For BG, it corrects the BG frames with the
same beam/OB settings and takes their pixel-wise **median**, retaining the
existing background calculation. The report gives actual TIFF counts.
All-runs preview sums are excluded.

Files are saved under `analysis/shared_corrections/<variant>/` as `ob.tiff`,
`background.tiff` and preparation records. Named preparations use
`alternatives/<name>/`. The resulting file paths automatically fill the next
section; preparation does not run sample correction.

## Use prepared OB/BG and correct samples

1. Select your **sample experiment** in **Project and data** and click
   **Discover data**.
2. Set **Prepared OB TIFF** to the processed, averaged OB image. Leave it blank
   to skip OB division (older source-folder configurations remain available
   in **Advanced / legacy OB settings**).
3. For a shared BG, select **BG source → Prepared BG TIFF**, choose the
   processed median **Prepared BG TIFF**, and enable background correction.
   The BG must have the same corrected units as the sample frames.
4. Click **Run corrections**, then **Run background** if enabled, then
   **Load images** in **Image products**.

The selected OB and BG images are reused directly for individual and batch
processing. They are not averaged or spatially processed again. Prepared OB
beam normalization is read from its adjacent `ob_preparation.json` when
available. For an external prepared OB with beam correction, supply the
corresponding **OB beam integral / min** in advanced settings; selecting a
different OB clears the previous normalization value.

As an alternative to a shared BG file, select **BG source → First X frames per
experiment** and set **First images (per experiment)**. Each sample's BG is the
pixel-wise median of its first X valid beam/OB-corrected TIFFs, before frame
merging. These frames remain in the output series. Unreadable and all-zero
TIFFs are skipped; if fewer than X valid images exist, all are used. Counts
are recorded in `processing/background_metadata.json` and the image is saved
as `processing/background.tiff`.

For batches, include **Corrections** and, when needed, **Background correction**
in the selected steps. The same prepared-file paths and BG source are used.

## Image products

Run this section after **Discover data** has completed:

1. Keep **Frames/merge** at `1` to retain every time point. A larger value
   averages that many consecutive images into one frame.
2. Keep **Spatial merge** at `1` to retain the original pixels. A larger value
   averages blocks of that size, such as `2` for 2×2 pixels.
3. Click **Load and publish processed images** (called **Load images** in the old
   notebook). This reads TIFFs and JSON acquisition times and publishes a portable
   processed stack; it does not alter source data.
4. Click **Write average** only if an experiment-average TIFF is wanted. Output
   reports loaded-image and source-TIFF counts.

With the `exsitu` mode, **Write average** also copies each experiment's average
to `analysis/proposal_summary/averaged_images/`. This happens for individual
processing and batch jobs that include **Write average TIFF**. Filenames include
the full experiment name and TIFF variant; rerunning averaging refreshes that
copy. Originals remain in each experiment's `images/` folder. Set
`collect_in_proposal = False` in `[average]` to disable copying.

For raw or already reconstructed final TIFFs, load images directly. If
corrections are required, run **Corrections** first and then return here and
click **Load images** so the corrected stack is loaded.

### White spots / hot pixels

In **Image products**, enable **Remove white spots / hot pixels** (off by
default). Start with a **3 px** window, **Automatic noise (MAD)**, threshold **8**.
Choose a **Preview frame (0-based)** and click **Preview white-spot removal**
to compare before/after and the replaced-pixel mask without saving TIFFs.
Real small bright features can also be removed: inspect the preview and raise
the threshold if needed. For flat/sparse images with zero estimated noise, try
an absolute threshold instead.

Click **Load and publish processed images** to apply it, then **Write average**
or open the result in visualization. Batches apply the same settings through **Load processed images**
(also added automatically for averaging/plots). No additional batch step is
needed. For existing finished analyses, choose a new **Analysis name**, such as
despotted, discover data, and rerun the desired corrections before loading.

Only bright pixels exceeding their local median by the threshold are replaced
with that median. **Absolute difference** uses the current image units;
**Pyerre count-noise** uses threshold × √max(local median, 0) and is intended
for count images. Automatic noise uses a robust MAD estimate of each frame's
median-filter residuals. This adapts pyerre's ro_filter_2D idea with reflected
edges and bright-only replacement; it is not a whole-image smoothing filter.

Filtering runs after any selected beam/OB/BG corrections, before cropping and
temporal/spatial merging. It does not change shared OB/BG preparations, raw
TPX3 files, source TIFFs, or 3D/TOF cubes. Filtered full-size TIFFs are saved in
images/white_spot_filtered_frames/; processing/white_spot_filter.json
records parameters, input files, and replacement counts for every frame.
Settings are also saved in the main config and per-experiment snapshot.

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
  whenever shared beam history is supplied.
- Runs marked as whole-run exclusions are omitted automatically, even if stale
  derived TIFFs still exist. Their raw TPX3 files remain retained.
- Enable and run background correction only when needed.
- Open the published result in visualization to measure images or export figures/videos.

In the visualization notebook for operando work, load electrochemistry after measuring images, then run
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

The shared **Beam history** field accepts either one file or a folder. For
a folder, the pipeline recursively combines supported `.dat`, `.csv`, `.tsv`,
`.txt`, `.log`, and extensionless text files, then sorts all rows by timestamp.
For overlapping timestamps, the value from the later filename is kept.

## Batch processing

Use **Batch processing** when several experiments need the same settings and
reference/OB:

1. Select the proposal under **Project and data**. For shared correction, first
   optionally prepare OB/BG files, then select their paths in **Use prepared OB/BG and correct samples**. Fill in correction and image-product settings.
2. Open **Batch processing**, optionally enter a name/filter, and click
   **Discover batch experiments**. Setup, OB, test, background, and focus data
   are hidden by default; no experiment is selected automatically.
3. Highlight experiments in **Currently unselected**, then click **Add
   highlighted →**. They move to **Selected**. Use **Add all shown →** for the
   filtered list; highlight items on the right and click **← Remove
   highlighted** to remove them. Changing the filter does not remove experiments
   already selected.
4. Select the intended processing sections. Include **Background correction**
   to apply the **BG source** chosen above (shared or per-experiment first X
   frames). Prerequisites are added automatically.
5. Click **Validate batch**. Resolve every invalid row before continuing.
6. Click **Run or resume batch**. Experiments run sequentially; one failure is
   recorded without stopping later experiments. A repeated run skips jobs
   already completed with the same recipe and unchanged source inputs unless
   **Re-run jobs** is selected.

Shared OB and background caches are reused when inputs and settings are
unchanged. Batch recipes, logs, and status tables are saved under
`analysis/batches/<batch-name>/`. Named alternative analyses also separate the
batch records beneath `alternatives/<name>/`. Each experiment gets its own
outputs and a `processing/config_used.ini` snapshot.

For a long terminal run, first save/validate the recipe in the notebook, then
run it from this project folder:

```bash
PYTHONPATH=src python3 -m lumacam_pipeline \
  --config /proposal/analysis/batches/<batch-name>/batch_recipe.ini \
  batch
```

## Individual image figures (visualization)

After **Open images**, use **Make individual figures** to export every loaded
image, or enter selected indices only. Files are written to the profile's
`figures/<export-id>/` with the five-digit image index and source filename. The
**Individual-figure colour scaling** offers **Global** (all loaded frames use the
same limits, recommended for comparisons) or **Per image** (each source frame's
own finite minimum/maximum, useful for detail but colours are not comparable).
Entering both colour limits overrides either automatic mode. Automatic limits
use full source frames, not the display crop. Panels and videos keep a global
scale. Individual exports do not overwrite an existing image panel.

### 3D image figures and videos

In **Preview and appearance**, enable **3D surfaces**. This affects previews,
image panels, individual figures and image-only videos, not synchronized EC videos.
Set elevation/azimuth, display stride, and optional paired height limits. Blank
height limits use the colour range. Output names include `_3d`, preserving 2D
exports. Spatial axes replace 2D scale bars; they use µm only when pixel size is
known, otherwise pixels. Height is the measured image value, not a reconstructed
physical electrode surface. Display stride subsamples rendering only; images and
measurement calculations are unchanged. Use the same height/colour limits for comparisons.

### Image-only video timing

In **Preview and appearance**, choose:

- **Image sequence** (default): one video frame per input image, including when
  timestamps are unavailable. Acquisition gaps are compressed.
- **Elapsed-time playback**: each video frame advances `dt_seconds` experimental
  seconds; blank uses the full-record median acquisition interval. For example,
  60 seconds per frame at 10 FPS represents 10 experimental minutes per playback
  second. Real, unique, increasing timestamps are required. The selected time
  window is respected without resetting the original elapsed-time origin.

Elapsed mode shows the nearest acquired image only within **Maximum time-to-image
difference** (`image_tolerance_seconds`). Blank uses 0.75 × the full-record median
interval, independent of playback step. Otherwise the image area is blank and
labelled **No acquisition data**. No pixel interpolation occurs; matched frames
show both timeline and acquisition time. This is timestamp proximity, not proof
of continuous exposure coverage. Check the tolerance for sparse/irregular records
and specify it explicitly when the automatic value is unsuitable. A coarse step
can skip short gaps or images; choose an appropriate resolution. The final grid
point never exceeds the window end. Exports above 100,000 frames are rejected.

`tables/images[_3d]_video_timing.csv` maps every output frame to its acquisition
or gap; the adjacent JSON records resolved interval, tolerance and gap count.
These are included in export provenance alongside the profile settings.

CLI/AI example (global options precede the command):

```bash
lumacam --config /proposal/reduced/analysis/visualization/exp101/settings.ini visualize \
  --set image_video.render_3d=true \
  --set image_video.timing_mode=elapsed_time \
  --set image_video.dt_seconds=60 \
  --set image_video.image_tolerance_seconds=45 \
  --sections image_video
```

## Section states

Click **Section status — click to show or fold** to expand or collapse the full
status table.

- **not configured**: the optional section is disabled or has no path.
- **waiting for input**: run the prerequisite section shown under missing artifacts.
- **ready**: the section can be run.
- **completed**: the last run succeeded.
- **failed**: read the **Details** column or output box, correct the reported
  problem, and run the section again.

## Configuration and outputs

New proposals created by the LumaCam setup routine already contain
`analysis/lumacam_pipeline.ini`: **Data** points to that proposal,
electrochemistry to `data/echem/`, and sample/OB/BG beam history to the same
`data/beamhistory/` folder. Experiment, TIFF variant and OB/BG selections start
blank; background correction starts off. Other processing defaults are supplied
when the pipeline loads the file. Choose the mode, experiment and variant before
discovery. Put optional input files in those folders, or clear the corresponding
paths when not using them (also skip electrochemistry steps when unused).
An empty beam-history folder is treated as absent by the new processing workflow;
malformed non-empty inputs still report errors rather than silently skipping.
Copying the proposal template manually does not generate or repoint this config.

Dashboard maintains one editable proposal-level main config:
`analysis/lumacam_pipeline.ini`. It includes shared OB/BG
experiment IDs, beam paths, prepared cache paths, batch selection, and normal
processing settings. Discover data and normal processing actions save it
automatically; batch validation/run also save it, and shared OB/BG preparation
saves its completed settings. **Config** switches to this path after proposal
discovery. Edits alone are not continuously autosaved: click **Save config**
if you are changing settings without running anything.

When returning to an existing proposal, select this file under **Config** and
click **Load config** before discovery/processing. Selecting the Data folder
alone does not restore its saved settings; discovery saves the current GUI
settings. For a new proposal without an INI, start from the built-in GUI defaults,
choose paths/settings and discover data to create the proposal's main config.
The old Harris-specific starter INI is archived and should not be loaded for new work.

Individual processing and batches write `processing/config_used.ini` within
the experiment/variant result folder (or its named alternative). It is refreshed
after successful processing, **not** a new history file for every click.
`processing/recipes.json` records each
completed correction/averaging recipe. Edit the proposal main config, not
these automatic records. Loading a snapshot uses it as starting settings;
subsequent GUI saves go to the main proposal config.

Use **Load config** to restore settings and **Save config** to save without
processing. Normal processing actions also save the main config automatically.
Every path field has a **Browse** button. Prepared OB and BG fields select TIFF
files. Beam history, legacy OB source, and electrochemistry accept a file or folder.
Navigate into a folder and click **Use current folder** to select the whole
folder.
The result structure is:

```text
analysis/
  lumacam_pipeline.ini
  shared_corrections/<variant>/       # shared OB/BG, preparation records
  batches/<batch-name>/              # recipe, status and logs
  proposal_summary/averaged_images/   # comparison copies
  <experiment>/<variant>/
    images/corrected_frames/          # final corrected frames
    images/processed_frames/          # published filtered/cropped/merged frames
    images/average_<experiment>_<variant>.tiff
    visualization/<profile>/         # separate settings + versioned exports
    processing/processed_dataset.json
    processing/config_used.ini
    processing/recipes.json
    processing/intermediate/uncorrected/
    processing/intermediate/beam_ob_corrected/  # before BG subtraction
```

Uncorrected TIFF copies are still produced. Shared BG files are referenced,
not copied into every experiment. A background calculated specifically from
an experiment's first frames stays in that experiment's `processing/` folder.

For a different correction/averaging recipe, set **Analysis name**, such as
`background_check`, then click **Discover data** and process again. Outputs go
under `<experiment>/<variant>/alternatives/background_check/`; collected
averages also include the name. The pipeline rejects overwriting a recorded
recipe with different settings. Rerunning the same recipe is allowed.

Blue terms marked with an information symbol are clickable. Click one to show
or hide its concise explanation directly in the processing dashboard.
