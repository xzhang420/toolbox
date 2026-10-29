"""Create a small synthetic visualization example in a new directory."""

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import tifffile


def create_example(directory):
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    yy, xx = np.mgrid[:64, :64]
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for index in range(6):
        image = (20 + 80 * np.exp(-((xx - 20 - 3*index)**2 + (yy - 32)**2) / 100)).astype(np.float32)
        stamp = (start + timedelta(seconds=60*index)).isoformat()
        tifffile.imwrite(root / f"frame_{index:03d}.tiff", image, metadata=None,
                         description=f"Synthetic demonstration data\nAcquisitionTime:{stamp}")
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="New output directory; existing directories are refused")
    print(create_example(parser.parse_args().directory))
