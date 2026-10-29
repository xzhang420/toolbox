# Toolbox instructions

For LumaCam/NCI work, use the terminology below. Read
`lumacam_pipeline/AGENTS.md`, `lumacam_pipeline/README.md`, and
`lumacam_pipeline/user_manual.md` before modifying that package. Other toolbox
utilities need not be moved into the pipeline merely because they analyze images.

## LumaCam/NCI workflow terminology

| Stage | Meaning | Responsible software |
| --- | --- | --- |
| **Data reduction** | Convert raw detector data into reconstructed products: pixel-to-photon conversion, photon-to-event conversion, image reconstruction, and related steps. | `lumacam_measurementcontrol`, including `pixel2photon`, `photon2event`, and associated reconstruction scripts. |
| **Data processing** | Correct and prepare reconstructed images: OB/open-beam correction, beam-intensity correction, reference normalization/calibration, background correction, white-spot removal, merging, and averaging. | Processing part of `lumacam_pipeline`, exposed through `lumacam_processing.ipynb`. |
| **Data analysis & visualization** | Extract scientific measurements from processed images, correlate electrochemistry, and create figures, tables, and videos. | Analysis/visualization part of `lumacam_pipeline`, exposed through `lumacam_visualization.ipynb`. |

NCI means neutron capture imaging; LumaCam also supports other modes such as time
of flight. These are responsibility boundaries, not a mandatory linear sequence.

- Use existing reconstructed inputs for processing and existing processed inputs
  for analysis/visualization. A request for a later stage does not authorize
  rerunning earlier stages. Clarify the intended stage when genuinely ambiguous.
- Storage names do not define stages: `reduced/` can contain all downstream
  outputs, `analysis/` also contains processing outputs, and acquisition `final/`
  TIFFs are reduction products rather than guaranteed corrected images.
- Preserve source data, metadata, historical records, and unrelated code changes.
  Do not commit or push software changes without authorization.
- Store proposal-specific processing/analysis/visualization scripts, outputs and
  documentation in the proposal's `reduced/analysis/` (or existing `analysis/`),
  never `~/Programs` or proposal `documentation/`. Use `analysis/jobs/` for run
  records and `analysis/notes/` for shared notes; retain reusable code/manuals here.
- The package repeats these definitions so copying only `lumacam_pipeline/` to
  another device retains the terminology. Proposal folders do not need duplicate
  general instructions; keep any proposal-specific exceptions with that dataset.
