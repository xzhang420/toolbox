# Archived combined LumaCam entry point

Moved without changing file contents on 2026-10-29:

- `lumacam_pipeline.ipynb`: superseded all-in-one launcher.
- `lumacam_pipeline_config.ini`: historical Harris-specific settings, not a portable template.

Use `../../lumacam_pipeline/lumacam_processing.ipynb` and
`../../lumacam_pipeline/lumacam_visualization.ipynb` for new work. Load each
proposal's own `analysis/lumacam_pipeline.ini`; new setups use built-in defaults.

These files are recoverable historical references, not the runnable distribution.
To restore the notebook, copy it back to the package folder under a distinct name;
its source lookup expects that location. The compatibility API remains available.
Do not remove `src/lumacam_pipeline/legacy_backend/`: the current package uses it.
No proposal data, results, or unrelated toolbox utilities were moved.
