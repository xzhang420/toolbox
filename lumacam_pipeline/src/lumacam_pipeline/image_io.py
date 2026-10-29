"""Read-only TIFF/FITS readers with explicit dimensionality and timing rules."""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import tifffile

IMAGE_SUFFIXES = {".tif", ".tiff", ".fit", ".fits", ".fts"}


def natural_key(path):
    return tuple((0, int(part)) if part.isdigit() else (1, part.lower())
                 for part in re.split(r"(\d+)", str(path)))


def parse_timestamp(value, timezone=""):
    """Return UTC datetime; never substitute filesystem modification times."""
    if value in (None, ""):
        return None
    try:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp):
            return None
        if stamp.tzinfo is None:
            if not timezone:
                return None
            stamp = stamp.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
        return stamp.tz_convert("UTC").to_pydatetime()
    except (ValueError, TypeError):
        return None


def read_image(path, *, fits_hdu=None, timestamp_timezone="", allow_cube=False):
    """Keep stored pixel order and physical values, including FITS BSCALE/BZERO.

    Multiple image HDUs require an explicit selection. No flipping, rotation,
    normalization, or filesystem-time inference takes place here.
    """
    path = Path(path)
    metadata = {"timestamp": None, "unit": "", "pixel_size_um": None}
    if path.suffix.lower() in {".fits", ".fit", ".fts"}:
        try:
            from astropy.io import fits
        except ImportError as exc:
            raise RuntimeError("FITS support requires astropy: install lumacam-pipeline[fits]") from exc
        with fits.open(path, memmap=False) as hdus:
            candidates = [i for i, hdu in enumerate(hdus)
                          if hdu.data is not None and np.asarray(hdu.data).ndim in (2, 3)]
            if fits_hdu in (None, ""):
                if len(candidates) != 1:
                    raise ValueError(f"Select input.fits_hdu; image HDUs in {path.name}: {candidates}")
                index = candidates[0]
            else:
                index = int(fits_hdu) if str(fits_hdu).isdigit() else fits_hdu
            array = np.array(hdus[index].data, copy=True)
            header = hdus[index].header
            metadata["unit"] = str(header.get("BUNIT", ""))
            metadata["timestamp"] = parse_timestamp(header.get("DATE-OBS"), timestamp_timezone)
            metadata["fits_hdu"] = index
    elif path.suffix.lower() in {".tif", ".tiff"}:
        with tifffile.TiffFile(path) as tif:
            array = tif.asarray()
            tags = tif.pages[0].tags
            values = [tags[key].value for key in (65002, 65000, "DateTime") if key in tags]
            for line in (tif.pages[0].description or "").splitlines():
                if line.startswith(("AcquisitionTime:", "MeanTPXModifiedTime:")):
                    values.append(line.split(":", 1)[1].strip())
            for value in values:
                if re.match(r"^\d{4}:\d{2}:\d{2} ", str(value)):
                    value = str(value)[:10].replace(":", "-") + str(value)[10:]
                stamp = parse_timestamp(value, timestamp_timezone)
                if stamp is not None:
                    metadata["timestamp"] = stamp
                    break
    else:
        raise ValueError(f"Unsupported image format: {path.suffix}")
    if array.ndim != 2 and not (allow_cube and array.ndim == 3):
        raise ValueError(f"Expected a 2-D image, got {array.shape}: {path}. Select a plane or use the ToF tools.")
    if not array.size or not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError(f"Expected non-empty, real numeric image data: {path}")
    return array, metadata
