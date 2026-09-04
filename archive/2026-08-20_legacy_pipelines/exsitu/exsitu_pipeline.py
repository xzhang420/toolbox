"""Compatibility import for the shared LumaCam ex-situ backend."""

from lumacam_pipeline.legacy_backend import exsitu_pipeline as _backend

globals().update(
    {name: value for name, value in vars(_backend).items() if not name.startswith("__")}
)


if __name__ == "__main__":
    raise SystemExit(_backend.main())
