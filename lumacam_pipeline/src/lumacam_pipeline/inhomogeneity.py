"""Spatial variation of already-processed images; no corrections or filtering."""
import numpy as np
import pandas as pd

from .measurements import region_mask
from .section import PipelineSection, SectionOutcome, SectionState


def measure_pattern(images, times, measurement, options=None, *, unit=''):
    """Return descriptive per-frame metrics, retaining signed input values.

    Circular geometry is evaluated only for an explicitly circular ROI. The
    centroid uses positive weights; this does not change the input image.
    """
    options = options or {}
    if not len(images) or len(times) != len(images):
        raise ValueError('Images and image timestamps must have the same nonzero length.')
    mask = region_mask(np.shape(images[0]), measurement)
    roi = measurement.get('roi') or {}
    kind = measurement.get('region') or ('circle' if all(k in roi for k in ('x0', 'y0', 'r0')) else 'whole')
    threshold = options.get('active_threshold')
    threshold = None if threshold in (None, '') else float(threshold)
    if threshold is not None and (not np.isfinite(threshold) or threshold < 0):
        raise ValueError('Active threshold must be finite and nonnegative.')
    floor = options.get('min_mean_signal')
    floor = None if floor in (None, '') else float(floor)
    if floor is not None and (not np.isfinite(floor) or floor < 0):
        raise ValueError('Minimum mean signal must be finite and nonnegative.')
    inner = float(options.get('centre_radius_fraction', .5))
    outer = float(options.get('edge_start_fraction', .8))
    if not (0 < inner < outer < 1):
        raise ValueError('Require 0 < centre radius fraction < edge start fraction < 1.')
    yy, xx = np.indices(mask.shape)
    if kind == 'circle':
        x0, y0, r0 = (float(roi[k]) for k in ('x0', 'y0', 'r0'))
        rr = np.hypot(xx-x0, yy-y0)/r0
        centre, edge = mask & (rr <= inner), mask & (rr >= outer)
        truncated = (x0-r0 < 0 or y0-r0 < 0 or x0+r0 > mask.shape[1]-1 or y0+r0 > mask.shape[0]-1)
    else:
        centre = edge = None
        truncated = False
    valid_times = [t for t in times if t is not None]
    origin = min(valid_times) if len(valid_times) == len(times) else None
    results = []
    for index, (image, timestamp) in enumerate(zip(images, times)):
        array = np.asarray(image, dtype=float)
        if array.shape != mask.shape:
            raise ValueError('All images must have the same shape.')
        valid = mask & np.isfinite(array)
        values = array[valid]
        mean = float(values.mean()) if len(values) else np.nan
        sigma = float(values.std(ddof=0)) if len(values) else np.nan
        flags = []
        if truncated: flags.append('circle_truncated_by_image')
        if valid.sum() != mask.sum(): flags.append('nonfinite_pixels')
        if not len(values): flags.append('no_valid_pixels')
        if len(values) and np.all(values == 0): flags.append('all_zero_roi')
        from .quality import pixel_diagnostics
        diagnostics, pixel_flags = pixel_diagnostics(values)
        flags.extend(pixel_flags)
        eligible = np.isfinite(mean) and mean > 0 and (floor is None or mean > floor)
        if not eligible: flags.append('cv_undefined_or_below_signal_floor')
        p10, p90 = np.percentile(values, [10, 90]) if len(values) else (np.nan, np.nan)
        active = values[values > threshold] if threshold is not None else np.array([])
        result = {'frame': index, 'time_utc': timestamp, 'elapsed_h': (timestamp-origin).total_seconds()/3600 if origin else np.nan,
            'unit': unit or 'a.u.', 'roi_pixels': int(mask.sum()), 'valid_pixels': int(valid.sum()),
            'mean': mean, 'std': sigma, 'cv': sigma/mean if eligible else np.nan,
            'p10': float(p10), 'p90': float(p90), 'p90_minus_p10': float(p90-p10),
            'active_threshold': threshold, 'coverage_fraction': len(active)/len(values) if threshold is not None and len(values) else np.nan,
            'cv_active': float(active.std(ddof=0)/active.mean()) if len(active) and active.mean() > 0
                         and (floor is None or active.mean() > floor) else np.nan,
            'centre_mean': np.nan, 'edge_mean': np.nan, 'edge_minus_centre': np.nan, 'centroid_offset_radius': np.nan,
            'min_mean_signal': floor, 'quality_flags': ''}
        result.update(diagnostics)
        if threshold is not None and not len(active): flags.append('empty_active_region')
        if centre is not None:
            centre_values = array[valid & centre]; edge_values = array[valid & edge]
            if len(centre_values) and len(edge_values):
                result['centre_mean'] = float(centre_values.mean())
                result['edge_mean'] = float(edge_values.mean())
                result['edge_minus_centre'] = result['edge_mean']-result['centre_mean']
            else:
                flags.append('empty_centre_or_edge')
            weights = np.maximum(array[valid], 0.)
            if weights.sum() > 0 and eligible:
                xc = np.average(xx[valid], weights=weights); yc = np.average(yy[valid], weights=weights)
                result['centroid_offset_radius'] = float(np.hypot(xc-x0, yc-y0)/r0)
            else:
                flags.append('centroid_undefined_or_below_signal_floor')
        result['quality_flags'] = ';'.join(flags)
        results.append(result)
    return pd.DataFrame(results)


def spatial_diagnostics(images, measurement, indices, bins=12):
    """Radial means and ROI-size sensitivity, without choosing a new electrode."""
    roi = measurement.get('roi') or {}
    if (measurement.get('region') or ('circle' if all(k in roi for k in ('x0','y0','r0')) else 'whole')) != 'circle':
        return pd.DataFrame(), pd.DataFrame()
    if int(bins) < 2 or int(bins) > 100:
        raise ValueError('Radial bins must be between 2 and 100.')
    yy, xx = np.indices(np.shape(images[0]))
    radius = np.hypot(xx-float(roi['x0']), yy-float(roi['y0'])) / float(roi['r0'])
    edges = np.linspace(0, 1, int(bins)+1)
    radial, sensitivity = [], []
    for index in indices:
        array = np.asarray(images[index], dtype=float)
        for j, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
            valid = (radius >= left) & (radius <= right if j == int(bins)-1 else radius < right) & np.isfinite(array)
            values = array[valid]
            radial.append(dict(frame=index, radius_fraction=(left+right)/2, pixels=int(valid.sum()),
                               mean=float(values.mean()) if len(values) else np.nan,
                               std=float(values.std()) if len(values) else np.nan))
        for factor in (.9, 1., 1.1):
            values = array[(radius <= factor) & np.isfinite(array)]
            mean = float(values.mean()) if len(values) else np.nan
            sensitivity.append(dict(frame=index, radius_factor=factor, pixels=len(values), mean=mean,
                                    cv=float(values.std()/mean) if mean>0 else np.nan,
                                    diagnostic_only=True))
    return pd.DataFrame(radial), pd.DataFrame(sensitivity)


class InhomogeneitySection(PipelineSection):
    id = 'inhomogeneity'
    title = 'Plating pattern / inhomogeneity'
    requires = ('image_stack', 'image_times', 'results_dir')
    optional_inputs = ('measurement_mask', 'view_window', 'selected_image_indices', 'event_context', 'image_unit', 'image_metrics_table')
    produces = ('inhomogeneity_table', 'inhomogeneity_csv_path', 'inhomogeneity_figure_path', 'pattern_diagnostic_paths')

    def run(self, session, config):
        import matplotlib.pyplot as plt
        from .event_selection import selected_records
        from .sections.outputs import _category_dir
        from .time_window import bounds, time_mask
        times = session.artifacts.require('image_times')
        data = measure_pattern(session.artifacts.require('image_stack'), times,
            session.section_config('image_measurements'), config, unit=session.artifacts.get('image_unit', ''))
        timed = all(t is not None for t in times)
        window = bounds(session, 'figures')
        keep = time_mask(times, window)
        if not keep.any():
            raise ValueError('No image frames are in the selected figure window.')
        csv = _category_dir(session, 'tables')/'plating_inhomogeneity.csv'
        data.to_csv(csv, index=False, float_format='%.12g')
        fig, axes = plt.subplots(3, 1, figsize=(8, 6.5), sharex=True, constrained_layout=True)
        try:
            x = data.elapsed_h.to_numpy() if timed else data.frame.to_numpy()
            unit = session.artifacts.get('image_unit') or 'a.u.'
            unit = 'µm' if unit in {'um', 'µm', 'μm'} else unit
            specs = [('cv', 'ROI CV (×)', '#7046a3'), ('p90_minus_p10', f'P90 − P10 ({unit})', '#d36c22'),
                     ('edge_minus_centre', f'Edge − centre ({unit})', '#177f7a')]
            rows = selected_records(session, category='figures') if timed else []
            zero_frames = data.quality_flags.str.contains('all_zero_roi').to_numpy()
            for axis, (column, label, color) in zip(axes, specs):
                y = data[column].to_numpy(dtype=float).copy()
                # A completely zero ROI is flagged as missing-quality data in
                # trend figures; preserve its actual numerical values in CSV.
                y[zero_frames] = np.nan
                if column == 'cv':
                    finite = y[keep & np.isfinite(y)]
                    positive = finite[finite > 0]
                    if len(positive) and positive.max()/positive.min() > 8:
                        if np.all(finite > 0):
                            axis.set_yscale('log')
                            label = 'ROI CV (×, log scale)'
                        else:
                            axis.set_yscale('symlog', linthresh=.1)
                            label = 'ROI CV (×, symlog scale)'
                x_plot, y_plot = x[keep], y[keep]
                if timed and len(x_plot) > 1:
                    # Do not imply continuity through long acquisition gaps.
                    gaps = np.flatnonzero(np.diff(x_plot) > float(config.get('gap_minutes', 5))/60)+1
                    x_plot = np.insert(x_plot, gaps, np.nan); y_plot = np.insert(y_plot, gaps, np.nan)
                axis.plot(x_plot, y_plot, color=color, linewidth=1.3)
                suspect = data.quality_flags.str.contains('suspected_bright_outlier|background_dominated_roi').to_numpy() & keep
                if suspect.any():
                    axis.plot(x[suspect], y[suspect], 'o', mfc='none', mec='red', ms=5,
                              label='Flagged image / region')
                    if column == 'cv': axis.legend(fontsize=7, loc='upper right')
                axis.set_ylabel(label); axis.grid(alpha=.18)
                for position in x[zero_frames & keep]:
                    axis.axvline(position, color='0.55', alpha=.45, linestyle=':', linewidth=.7)
                if column == 'edge_minus_centre': axis.axhline(0, color='0.55', lw=.7)
                for row in rows:
                    i = row['image_index']
                    if keep[i] and np.isfinite(y[i]):
                        axis.plot(x[i], y[i], 'ko', ms=2.5)
                        axis.annotate(row['label'], (x[i], y[i]), xytext=(2, 3), textcoords='offset points', fontsize=7)
                if not np.isfinite(y[keep]).any():
                    axis.text(.5, .5, 'Requires an explicit circular ROI' if column == 'edge_minus_centre' else 'No valid values',
                              transform=axis.transAxes, ha='center', va='center', fontsize=9)
            axes[-1].set_xlabel('Time since first image (h)' if timed else 'Frame index')
            path = _category_dir(session, 'figures')/'plating_inhomogeneity.png'
            fig.savefig(path, dpi=int(config.get('dpi', 150)), bbox_inches='tight')
        finally:
            plt.close(fig)
        records = selected_records(session, category='figures')
        indices = [row['image_index'] for row in records] or [int(data['mean'].fillna(-np.inf).idxmax())]
        radial, sensitivity = spatial_diagnostics(session.artifacts.require('image_stack'),
            session.section_config('image_measurements'), indices, config.get('radial_bins', 12))
        diagnostics = {}
        if not radial.empty:
            folder = _category_dir(session, 'tables')
            for name, table in (('radial_profiles', radial), ('roi_sensitivity', sensitivity)):
                destination = folder / f'{name}.csv'; table.to_csv(destination, index=False, float_format='%.12g')
                diagnostics[name] = destination
            fig, axes = plt.subplots(1, 2, figsize=(8, 3.3), constrained_layout=True)
            try:
                mapping = {row['image_index']: row for row in records}
                for index in indices:
                    row = mapping.get(index, {'label': str(index), 'kind': 'other'})
                    subset = radial.loc[radial.frame == index]
                    axis = axes[1] if row.get('kind') == 'stripping' else axes[0]
                    axis.plot(subset.radius_fraction.to_numpy(), subset['mean'].to_numpy(), lw=1.2, label=row['label'])
                for axis, title in zip(axes, ('OCV / plating / selected frames', 'Stripping endpoints')):
                    axis.set(xlabel='Radius / ROI radius', ylabel=f'Mean ({unit})', title=title)
                    axis.axhline(0, color='0.7', lw=.6)
                    if axis.get_legend_handles_labels()[0]: axis.legend(ncol=3, fontsize=8, title='Image letter')
                target = _category_dir(session, 'figures') / 'radial_profiles.png'
                fig.savefig(target, dpi=int(config.get('dpi', 150)), bbox_inches='tight')
                diagnostics['radial_figure'] = target
            finally:
                plt.close(fig)
        message = f'Measured {len(data)} frames; CSV and figure saved.'
        if config.get('active_threshold') in (None, ''):
            message += ' Coverage and active-region CV require an explicit threshold.'
        if config.get('min_mean_signal') in (None, ''):
            message += ' CV has no validated low-signal floor; interpret near-zero means cautiously.'
        return SectionOutcome(self.id, SectionState.COMPLETED, message,
            {'inhomogeneity_table': data, 'inhomogeneity_csv_path': csv, 'inhomogeneity_figure_path': path,
             'pattern_diagnostic_paths': diagnostics})
