# LumaCam toolbox

The primary entry point is the section-based
[LumaCam processing pipeline](lumacam_pipeline/README.md). Start it by opening
`lumacam_pipeline/lumacam_pipeline.ipynb`.
Routine instructions are in the
[user manual](lumacam_pipeline/user_manual.md).
It supports operando NCI, NCI image-only, ex-situ, generic image, and
exact-photon-count hyperspectral workflows without requiring one fixed linear
sequence.

Obsolete `operando`, `operando_dev`, and `exsitu` entry points have been moved
to `archive/` for recoverability. Active processing, tests, layout discovery,
metadata timing, corrections, optional electrochemistry, plots, and videos now
live in the self-contained `lumacam_pipeline` project folder.
