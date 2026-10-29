# LumaCam toolbox

The [LumaCam pipeline](lumacam_pipeline/README.md) has two independent notebooks:
`lumacam_pipeline/lumacam_processing.ipynb` for processing, and
`lumacam_pipeline/lumacam_visualization.ipynb` for analysis, figures and videos.
Routine instructions are in the
[user manual](lumacam_pipeline/user_manual.md).
Its modes are Operando NCI, Ex situ NCI and Time of flight; optional sections do
not require one fixed linear sequence.

Obsolete `operando`, `operando_dev`, and `exsitu` entry points have been moved
to `archive/` for recoverability. Active processing, tests, layout discovery,
metadata timing, corrections, optional electrochemistry, plots, and videos now
live in the self-contained `lumacam_pipeline` project folder.
