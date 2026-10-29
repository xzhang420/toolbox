# LumaCam processing, analysis & visualization

Two notebooks, one Python package. Use independent sections rather than running
every step in a fixed sequence.

| Open | Use for |
| --- | --- |
| [lumacam_processing.ipynb](lumacam_processing.ipynb) | Discover data, prepare shared OB/BG, correct images, remove hot pixels, merge, average, QC and batch processing |
| [lumacam_visualization.ipynb](lumacam_visualization.ipynb) | Open processed TIFF/FITS images, measure ROIs, optionally correlate electrochemistry, make figures and videos |

The old combined notebook and proposal-specific starter INI are archived in
`toolbox/archive/2026-10-29_combined_pipeline/`. Shared backend code and compatibility
APIs remain required and available. Use the two notebooks above; restart kernels after updates.
See the [user manual](user_manual.md) for controls and processing details, and
[AGENTS.md](AGENTS.md) for AI/developer instructions.

Processing publication/averaging and visualization actions automatically refresh
`overview.md` (AI-readable results, tables, embedded plot links) and `overview.pdf`
(human-readable report with the plots embedded). Keep `report_assets/` with the
Markdown. Batches also produce a combined report in their batch folder. These
are summaries of recorded results, not additional reduction/correction steps.

## LumaCam/NCI workflow terminology

| Stage | Meaning | Responsible software |
| --- | --- | --- |
| **Data reduction** | Convert raw detector data into reconstructed products: pixel-to-photon conversion, photon-to-event conversion, image reconstruction, and related steps. | `lumacam_measurementcontrol`, including `pixel2photon`, `photon2event`, and associated reconstruction scripts. |
| **Data processing** | Correct and prepare reconstructed images: OB/open-beam correction, beam-intensity correction, reference normalization/calibration, background correction, white-spot removal, merging, and averaging. | Processing part of `lumacam_pipeline`, exposed through `lumacam_processing.ipynb`. |
| **Data analysis & visualization** | Extract scientific measurements from processed images, correlate electrochemistry, and create figures, tables, and videos. | Analysis/visualization part of `lumacam_pipeline`, exposed through `lumacam_visualization.ipynb`. |

These terms apply across LumaCam modes, including neutron capture imaging (NCI)
and time of flight. A request for analysis/visualization does not require or
authorize rerunning reduction or corrections. Optional sections remain independent.
Folder names are storage conventions: `reduced/` includes downstream processing
and analysis outputs, while acquisition `final/` TIFFs are reduction products,
not necessarily corrected images.

## Start on this computer

Open the processing notebook from toolbox; no notebook copy in the proposal is
necessary. Load the proposal's main INI **before** discovering data. For Harris:

```text
/home/localadmin/data/Harris_ICON_Oct2026_P20261127/reduced/analysis/lumacam_pipeline.ini
```

Choose a mode (**Operando NCI**, **Ex situ NCI**, **Time of flight**), experiment
and variant. Run the corrections you need, then **Load and publish processed
images**, optionally **Preview processed image (QC)** and **Write average**.
Processing batches publish the same handover files for every completed experiment.
Neither raw TPX3 reconstruction nor automatic scientific choice of OB/BG is performed.

Open the visualization notebook and select the resulting
`analysis/<experiment>/<variant>/` folder. For older results or external data,
select a folder containing just the desired TIFF/FITS frames instead.

The visualization notebook includes **Thickness / ROI measurements**, **Time
window / zoom**, and **CSV exports**. Measure several methods together, choose
which to plot with electrochemistry, and export native or nearest-matched tables.
Windowing changes presentation/export only; full-data calibration, elapsed time
and electrochemical thickness remain unchanged. See the user manual for formulas,
units, threshold requirements and missing-data flags.

**Plating inhomogeneity** adds ROI spread/CV and, for circular electrodes,
edge–centre contrast and centroid displacement. Optional coverage needs an
explicit threshold; a defensible minimum mean signal can suppress unstable CVs.
Results and settings are saved in the profile. See the manual for definitions.

**Image quality** flags empty/nonfinite frames and dominant bright-pixel outliers,
and saves an electrode-region overlay. Inhomogeneity always runs this check first.
For a known diameter and pixel size, **Estimate centre** searches each experiment
independently; weak boundary evidence retains the previous centre and is reported.
**Cycle endpoints and residual signal** exports phase-end values and their change
from observed initial OCV. Radial profiles and ROI-size sensitivity accompany the
inhomogeneity results. Main figures can show mean thickness with potential/current;
other calculated methods remain available in CSVs.

Electrochemical thickness now accumulates signed capacity increments continuously
through counter resets, rests and file restarts. It describes net transferred
charge relative to the record start, not an absolute Li inventory. Image curves
use their own acquisition gaps; missing EC does not hide valid image measurements.
GUI/CLI reports default to a concise, figure-led PDF with numerical endpoint tables;
the Markdown keeps the full audit information. `lumacam report --style detailed`
retains the longer presentation.

The same controls are available to scripts/AI through INI sections and `--set`:

```bash
lumacam --config /path/to/profile/settings.ini visualize \
  --set 'image_measurements.methods=["mean","median","trimmed_mean"]' \
  --set 'image_measurements.primary_method="mean"' \
  --set 'combined_plot.image_methods=["mean","median"]' \
  --set 'time_window.enabled=true' --set 'time_window.mode="elapsed"' \
  --set 'time_window.start=2' --set 'time_window.end=5' \
  --set 'csv_export.scope="window"' --set 'csv_export.tolerance_seconds=30' \
  --sections image_measurements electrochemistry event_detection selection combined_plot csv_export
```

This example needs configured electrochemistry. For image-only work, use
`--sections image_measurements csv_export` and omit the EC/combined-plot actions.
The existing editable profile and per-export records retain these settings.

## Default layout and portability

Image-only exports support 2D images or 3D value surfaces. Individual figures
offer global or per-image colour scaling (explicit limits override either).
Image videos offer image-sequence or elapsed-time playback; elapsed mode labels
unobserved gaps and writes a frame-to-acquisition timing CSV/JSON. These controls
are under **Preview and appearance**, with details in the user manual.

```text
proposal/
  raw/<experiment>/<run>/                 # not needed for downstream processing
  reduced/
    data/
      experiment/{final,metadata,derived,logs,.work}/
      beamhistory/
      echem/
    documentation/
    analysis/
      lumacam_pipeline.ini               # editable processing settings
      shared_corrections/<variant>/
      batches/<batch-name>/
      proposal_summary/averaged_images/
      <experiment>/<variant>/
        images/processed_frames/         # exact filtered/cropped/merged stack
        images/corrected_frames/         # correction output before image merging
        images/average_*.tiff
        processing/
          processed_dataset.json         # relative frame paths, UTC times, hashes
          config_used.ini                # processing snapshot
          recipes.json
          intermediate/                  # existing copy policy retained
        visualization/<profile>/
          settings.ini                   # independently editable analysis settings
          dataset.json                   # input identity
          figures/<export-id>/
          videos/<export-id>/
          tables/<export-id>/
          records/<export-id>/            # settings + output checksums per action
```

Copy an experiment/variant result folder to a collaborator; they can visualize it
without acquisition paths or a running processing notebook. Include any external
electrochemistry files if needed. Relative visualization inputs survive relocation
when their relative arrangement is preserved. An explicit external output folder
is supported for read-only datasets. Different figure recipes use profile names;
different processing recipes use **Analysis name** alternatives.

For a simpler presentation folder, choose an output parent such as
`analysis/visualization/`, use the experiment ID as the profile, and set
`[visualization] export_layout = flat`. Current figures/videos/tables then sit
directly in those three folders; per-action settings/checksums remain in
`records/`. A proposal-wide `overview.md`/`overview.pdf` can live in
`analysis/proposal_summary/`, alongside cross-experiment comparison tables/figures.
An evidence-linked `interpretation.json` adds highlighted observations and
conclusions to both reports. Changed evidence suppresses outdated interpretations;
see the user manual for the format and automatic report settings.
This changes only visualization exports, not source-data locations.

Operando **Detect events** now prefers observed OCV and current half-cycle
endpoints. Image panels and potential/thickness curves share lettered points;
only the first image has a scale bar. Reports omit a duplicate standalone
thickness curve when a valid combined figure is available. See the manual for
thresholds, polarity labels, manual overrides and event audit tables.

Earlier `data/experiment`, `data/experiments`, legacy run folders and flat TIFF
inputs remain supported. No folders are moved and **no layout.json is generated**.
For an unfamiliar layout, run `inspect` and set explicit `[layout]` mappings in
the INI; see the manual. AI can inspect and edit these same settings, with no
private chat history or device-specific skill required.

## Install on another Ubuntu or Windows computer

Copy/clone this package folder, then use Python 3.10 or newer:

```bash
python -m venv .venv
# Ubuntu: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[notebook]"
lumacam doctor
jupyter lab
```

TIFF and FITS support is included. MP4 export additionally needs an FFmpeg
executable on PATH. `doctor` reports installed versions and FFmpeg availability.
Installation does not copy proposal data. The notebooks can be used anywhere
after package installation; without installation, launch them from this folder.
On Ubuntu, creating a virtual environment also requires the OS `python3-venv`
package. If it is unavailable, use an existing suitable environment or ask the
computer administrator; the pipeline does not install system packages itself.

## Command line and Python

Global options precede the command. `inspect` and the new `--validate-only`
commands do not write data or settings. Inspection validates discovery and timing
availability, not the scientific suitability of a reference or correction recipe.

```bash
lumacam --config /proposal/reduced/analysis/lumacam_pipeline.ini inspect
lumacam --config /proposal/reduced/analysis/lumacam_pipeline.ini process
lumacam --config /proposal/reduced/analysis/lumacam_pipeline.ini process-batch \
  --experiments exp101 exp103 --name samples --validate-only
# Remove --validate-only to process.

lumacam visualize --input /processed/exp101/5to1000 --profile overview \
  --sections image_measurements selection image_plot image_video
lumacam --config /processed/exp101/5to1000/visualization/overview/settings.ini \
  visualize --set image_plot.vmin=0 --set image_plot.vmax=10

# Rebuild reports from saved results without rerunning any processing/analysis:
lumacam report --input /processed/exp101/5to1000/visualization/overview
lumacam report --input /results/profile1 /results/profile2 --output /results/summary
```

Without installation, replace `lumacam` with
`PYTHONPATH=src python -m lumacam_pipeline` on Ubuntu. Legacy `run`, `status` and
`batch` commands are retained; `process` and `process-batch` enforce the new split.

```python
from lumacam_pipeline import create_processing_session, open_visualization

processing = create_processing_session("exsitu", {
    "project": {"data_path": "/proposal", "experiment": "exp101", "variant": "5to1000"}
})
# Run chosen sections and check each outcome before proceeding.
for section in ("project", "corrections", "images", "average"):
    outcome = processing.run(section)
    if outcome.state.value != "completed":
        raise RuntimeError(outcome.message)

view = open_visualization(processing.artifacts.require("results_dir"), profile="overview")
view.run("selection")
view.run("image_plot")
```

Synthetic example (choose a directory that does not exist):

```bash
python -m lumacam_pipeline.example ./example_images
lumacam visualize --input ./example_images
```

## Safety and tests

Visualization never runs corrections or rewrites processing snapshots. Missing
timestamps do not prevent image figures/video, but block time-dependent analysis.
Unknown pixel size remains unknown; enter a measured size to enable physical
scale bars. Preview sums are excluded. FITS data retains its pixel orientation
and scaled numeric values; multiple image HDUs require an explicit choice.

```bash
MPLBACKEND=Agg python -m unittest discover -s tests -p 'test_*.py' -q
```

Tests cover synthetic processing/batches, old/new/custom layouts, relocation,
TIFF/FITS reading, optional inputs, write isolation and MP4 export when FFmpeg is
installed. The automated checks run on Ubuntu; Windows GUI behavior still needs
testing on a Windows machine.
