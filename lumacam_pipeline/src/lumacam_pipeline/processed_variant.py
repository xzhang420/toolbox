"""Create an explicit processing variant from existing beam/OB-corrected images."""
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .config import save_config
from .handover import atomic_json, file_digest, load_processed_images, publish_processed_dataset
from .image_io import read_image
from .section import PipelineSession
from .white_spots import remove_white_spots, filter_settings


def derive_background_variant(source, background, output, *, experiment, variant,
                              timestamp_timezone='', pixel_size_um=None, unit='',
                              clip_negatives=False, white_spots=None):
    """Subtract the selected saved BG and optionally replace bright outliers.

    Inputs are already beam/OB corrected. A new output directory is required;
    original files and processing history are never modified. No reference,
    geometry, physical calibration or exposure duration is inferred.
    """
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new processing variant: {output}')
    dataset = load_processed_images(source, timestamp_timezone=timestamp_timezone,
                                    pixel_size_um=pixel_size_um, unit=unit)
    background = Path(background).expanduser().resolve()
    bg, _ = read_image(background)
    if bg.shape != dataset['images'][0].shape or not np.isfinite(bg).all():
        raise ValueError('Background must be finite and have the same shape as the sample images.')
    settings = dict(white_spots or {})
    selected = filter_settings(settings)
    rows, arrays = [], []
    for index, image in enumerate(dataset['images']):
        values = np.asarray(image, dtype=np.float32) - np.asarray(bg, dtype=np.float32)
        if clip_negatives: values = np.maximum(values, 0)
        row = {'frame': index, 'replaced_pixels': 0, 'negative_pixels': int(np.count_nonzero(values < 0))}
        if np.all(np.asarray(image) == 0):
            # Keep the frame/time, explicitly missing rather than manufacturing
            # an inverse-background image from a completely empty acquisition.
            values = np.full_like(values, np.nan)
            row['quality_flag'] = 'all_zero_input_retained_as_invalid'
        if selected['enabled']:
            values, mask, stats = remove_white_spots(values, filter_size=selected['filter_size'],
                                                   method=selected['method'], threshold=selected['threshold'])
            row.update(stats)
            # Retain the exact changed coordinates for review and reproducibility.
            row['replaced_yx'] = np.argwhere(mask).tolist()
        arrays.append(values); rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.derive-', dir=output.parent) as temp:
        stage = Path(temp) / 'result'
        config = {'workflow': {'role': 'processing'},
                  'project': {'data_path': str(dataset['root']), 'experiment': experiment, 'variant': variant},
                  'corrections': {'enabled': False, 'input_already_beam_ob_corrected': True},
                  'background': {'enabled': True, 'background_path': 'processing/background.tiff', 'clip_negatives': bool(clip_negatives)},
                  'images': {**settings, 'unit': unit, 'pixel_size_um': pixel_size_um, 'frames_per_merge': 1, 'spatial_pixel_merge': 1}}
        session = PipelineSession([], config)
        for key, value in {'results_dir': stage, 'image_stack': arrays, 'image_times': dataset['times'],
                           'image_paths': dataset['paths'], 'experiment': experiment,
                           'image_exposure_bounds': dataset.get('exposure_bounds'),
                           'source_tiff_count': len(arrays), 'nci_calibrated': unit in {'um','µm','μm'}}.items():
            session.artifacts.put(key, value, 'derivation')
        publish_processed_dataset(session)
        (stage / 'processing').mkdir(exist_ok=True)
        shutil.copy2(background, stage / 'processing/background.tiff')
        save_config(stage / 'processing/config_used.ini', 'operando_nci', config)
        atomic_json(stage / 'processing/derivation.json', {
            'operation': 'existing beam/OB-corrected images minus saved background',
            'source': str(dataset['root']), 'source_records': dataset['records'],
            'background_source': str(background), 'background_sha256': file_digest(background),
            'clip_negatives': bool(clip_negatives), 'white_spot_settings': selected, 'frames': rows,
            'total_replaced_pixels': sum(row['replaced_pixels'] for row in rows)})
        if output.exists(): raise FileExistsError(output)
        stage.rename(output)
    return output
