# LumaCam pipeline instructions

## LumaCam/NCI workflow terminology

| Stage | Meaning | Responsible software |
| --- | --- | --- |
| **Data reduction** | Convert raw detector data into reconstructed products: pixel-to-photon conversion, photon-to-event conversion, image reconstruction, and related steps. | `lumacam_measurementcontrol`, including `pixel2photon`, `photon2event`, and associated reconstruction scripts. |
| **Data processing** | Correct and prepare reconstructed images: OB/open-beam correction, beam-intensity correction, reference normalization/calibration, background correction, white-spot removal, merging, and averaging. | Processing part of `lumacam_pipeline`, exposed through `lumacam_processing.ipynb`. |
| **Data analysis & visualization** | Extract scientific measurements from processed images, correlate electrochemistry, and create figures, tables, and videos. | Analysis/visualization part of `lumacam_pipeline`, exposed through `lumacam_visualization.ipynb`. |

Use these terms consistently. NCI means neutron capture imaging; LumaCam also
supports other modes such as time of flight. These are stage boundaries, not a
mandatory linear sequence or a requirement to run every optional operation.
A processing request does not mean raw reconstruction; analysis/visualization
does not authorize rerunning reduction or corrections. Clarify genuinely ambiguous
requests before crossing stage boundaries.

Folder names are storage conventions: `reduced/` may hold outputs from all three
stages, `analysis/` also contains processing outputs, and acquisition `final/`
TIFFs are reduction products, not necessarily corrected images. Legacy command
names such as measurement control's `--defer-processing` still refer to reduction;
do not rename APIs or flags merely to align terminology. These definitions remain
with the package when copied independently of toolbox or a proposal.

## Start here

- Read README.md and user_manual.md. Use the installed `lumacam` command or
  `PYTHONPATH=src python -m lumacam_pipeline` from this directory.
- Run `lumacam doctor` to inspect dependencies, then `lumacam --config FILE inspect`
  before processing unfamiliar data. Inspection is read-only and returns JSON.
- Use `lumacam_processing.ipynb` for corrections and publication of processed
  images. Use `lumacam_visualization.ipynb` for independent analysis and rendering.
  The previous combined notebook and Harris-specific starter INI are archived in
  toolbox/archive/2026-10-29_combined_pipeline/. Do not remove legacy_backend:
  current processing and visualization still import it.

## Data and configuration boundaries

- Default proposal layout: `raw/` beside `reduced/`; images and metadata under
  `reduced/data/experiment/`; outputs and the main INI under `reduced/analysis/`.
- Preserve older proposal layouts. Do not rename directories, rewrite acquisition
  metadata, create layout.json, or change raw data to make discovery work.
- Explicit `[layout]` path/pattern mappings belong in the processing INI.
  Inspect matches before changing settings. Do not infer physical calibration,
  OB/BG assignments, or timing from ambiguous names or filesystem modification times.
- Processing starts from reconstructed images; raw TPX3 reconstruction is upstream.
  Exclude sumAllRuns previews. Never silently combine image variants.
- Visualization must not run corrections or modify source images, processing INIs,
  or processing/config_used.ini. Save settings and exports in a named profile.
- Relative processed-frame paths and schema-versioned processing metadata make
  datasets portable. No previous chat, private account, or local skill is required.

## Result reports

- Image-only rendering supports `render_3d`, `surface_stride`, `surface_elev`,
  `surface_azim`, `surface_zlim` in image_plot/individual_image_plots/image_video.
  Height represents image values, not inferred morphology; stride affects only display.
  Individual `color_scale` is global or individual; explicit vmin/vmax win.
  image_video `timing_mode` is image_sequence (default) or elapsed_time;
  dt_seconds is experimental seconds per frame, not FPS. Elapsed mode requires
  real strictly increasing timestamps. Nearest matching is bounded by
  image_tolerance_seconds (default 0.75 × full-record median interval); unmatched
  frames show No acquisition data. Review the inferred tolerance for sparse data.
  Keep the timing CSV/JSON and export records; do not invent acquisitions in gaps.

- Spatial measurements are in `[image_measurements]`: `region` (whole, circle,
  rectangle, polygon), `roi`, `methods` (mean, area_normalized, active_mean,
  median, trimmed_mean) and `primary_method`. Preserve old circular ROI settings.
  Never infer electrode area or an active threshold. Area normalization requires
  known µm calibration, pixel size, and explicit `electrode_area_mm2`; active mean
  requires `active_threshold` in image units. Preserve negative values and flags.
- `[inhomogeneity]` runs independently on the same image region in either NCI
  mode. CV = population SD / positive ROI mean; P90-P10 is a descriptive spread.
  Centre is r ≤ 0.5R, edge is r ≥ 0.8R by default, circular ROI only. Centroid
  uses positive signal weights and is normalized by radius. Coverage and active
  CV require an explicit threshold. No floor or threshold is inferred; near-zero
  CV is unstable, and invalid/zero frames are flagged. Full CSV is preserved when
  presentation time windows are selected. Noise can mimic inhomogeneity.
- `[time_window]` is presentation-only: `enabled`, `mode` (elapsed hours or
  absolute ISO), `start`, `end`, `apply_figures`, `apply_panels`, `apply_videos`.
  Never slice input EC before capacity-to-thickness conversion for a zoom.
  Preserve the first-image time origin and full-record event/cycle numbering.
- `[combined_plot] image_methods` selects calculated methods to display; blank
  uses the primary. `[csv_export] scope` is full/window, `include_comparison`
  enables nearest matching, and `tolerance_seconds` bounds it (default 30 s).
  Run image_measurements before csv_export, plus electrochemistry if wanted.
  Read native tables and matching flags; blanks are not zeros or interpolations.
  All settings use the same core in GUI, CLI and Python. CSVs and settings/history
  belong to the existing profile, never the acquisition folders.

- Deliver `overview.md` for AI and `overview.pdf` for human readers, containing
  actual plots/images, quantitative results with units, settings/provenance,
  completed or missing sections, and quality warnings—not only output links.
- Proposal reports should lead with evidence-backed observations and explicitly
  distinguished conclusions, highlighted in the PDF. Use material-independent
  `analysis/proposal_summary/` and `analysis/visualization/<experiment>/` folders.
  Compare spatial variation at comparable signal levels as well as endpoints;
  state ROI/noise/protocol limits and never count frames as independent cells.
  Keep authored interpretation with evidence hashes in `interpretation.json` so
  report refreshes retain it only while the cited evidence remains current.
- Processing publication/averaging and visualization actions refresh these
  reports automatically. Processing batches also write a combined overview.
  Report errors are separate from processing failures; check the action message.
- For existing outputs, use `lumacam report --input RESULT_OR_PROFILE` or
  `lumacam report --input PROFILE1 PROFILE2 --output SUMMARY_FOLDER`.
  This reads saved results; it must not rerun reduction, corrections or analysis.
  Use global `--variant` only for a user-confirmed report label, never a guess.
- Keep `report_assets/` with the Markdown when sharing; the PDF embeds its images.
  A text-only AI must inspect linked images separately. Read tables/quality notes
  before interpreting curves. Do not turn descriptive trends into causal claims.
- Do not silently discard zero/nonfinite measurements, infer calibration from
  filenames, or imply that report creation revalidated the underlying science.
- For operando presentation, prefer one linked electrochemistry/thickness figure
  and a clean lettered NCI image panel; do not duplicate the thickness curve.
  Use observed OCV and sustained current half-cycle endpoints where supported,
  with bounded image-time matching. Confirm electrode current polarity before
  calling phases plating/stripping. Record skips and incomplete final phases.
  One shared colour scale and a scale bar only on the first image are sufficient.
- Keep human-facing exports shallow (`visualization.export_layout = flat` is
  available); full coordinates, timestamps and settings belong in audit files,
  not every panel. Preserve legacy sources and user-authored exports.

## Development and validation

- Keep calculation code in src/lumacam_pipeline, not notebook cells. GUI and CLI
  must call the same Python APIs. Preserve existing public API behavior.
- Tests use synthetic temporary datasets, not beamtime data. Run:
  `MPLBACKEND=Agg python -m unittest discover -s tests -p 'test_*.py' -q`.
- Test FITS extension ambiguity, absent timestamps, copied processed datasets,
  raw/reduced and older layouts, and visualization/processing write isolation.
- For source changes, work in a user-authorized branch when requested; never
  stage unrelated changes, discard a dirty worktree, or push without authorization.
- Restart notebook kernels after module edits. Code on disk does not update an
  already loaded Python module.
