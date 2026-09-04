# Recycled legacy pipeline entry points

Moved here on 2026-08-20 after the processing implementations were merged
into `toolbox/lumacam_pipeline`.

## Contents

- `operando/`: original operando notebook/config/compatibility entry point.
- `operando_dev/`: development notebook/config, retained legacy GUI, and its
  historical tests.
- `exsitu/`: former ex-situ notebook/API/GUI entry point and worked photon
  hyperspectral notebook.
- `plots/plots.py`: former compatibility import for the shared image plotter.
- `plots/__pycache__/`: generated cache associated with that compatibility
  import.

The maintained calculations were not discarded. They live in:

```text
toolbox/lumacam_pipeline/legacy_backend/
toolbox/lumacam_pipeline/image_plots.py
toolbox/lumacam_pipeline/sections/
```

## Restore

From the toolbox directory, move an entry back only if an old notebook or
external script still requires its historical path, for example:

```bash
mv recycle/2026-08-20_legacy_pipelines/operando ./operando
```

No acquisition data, analysis results, beam history, or electrochemistry data
were moved into this recycle bundle.
