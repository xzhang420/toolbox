"""Auditable phase endpoints and residual signal from existing image measurements."""
import numpy as np
import pandas as pd

from .section import PipelineSection, SectionOutcome, SectionState


class CycleSummarySection(PipelineSection):
    id = 'cycle_summary'
    title = 'Cycle endpoints and residual signal'
    requires = ('event_context', 'image_metrics_table', 'image_times', 'results_dir')
    optional_inputs = ('inhomogeneity_table', 'frame_quality_table')
    produces = ('cycle_endpoints_table', 'cycle_summary_csv', 'cycle_summary_figure')

    def run(self, session, config):
        import matplotlib.pyplot as plt
        from .sections.outputs import _category_dir
        events = session.artifacts.require('event_context').get('events', [])
        table = session.artifacts.require('image_metrics_table')
        pattern = session.artifacts.get('inhomogeneity_table')
        times = session.artifacts.require('image_times')
        ocv = next((float(table.value.iloc[e['image_index']]) for e in events if e['kind']=='ocv'), np.nan)
        rows, prior = [], None
        for event in events:
            index = event['image_index']; value = float(table.value.iloc[index])
            row = {'label': event['label'], 'phase': event['kind'], 'caption': event['caption'],
                   'half_cycle': event.get('half_cycle'), 'frame': index,
                   'image_time_utc': times[index], 'event_time_utc': event['time'],
                   'elapsed_h': (times[index]-min(times)).total_seconds()/3600,
                   'value': value, 'unit': session.artifacts.get('image_unit') or 'a.u.',
                   'value_minus_initial_ocv': value-ocv, 'change_from_previous_endpoint': value-prior if prior is not None else np.nan,
                   'status': event.get('status', ''), 'interrupted': event.get('interrupted', False),
                   'timing_basis': event.get('timing_basis', 'representative timestamp; exposure unavailable'),
                   'image_offset_seconds': event.get('image_offset_seconds'),
                   'quality_flags': table.quality_flags.iloc[index] if 'quality_flags' in table else ''}
            if pattern is not None:
                for key in ('cv','std','p90_minus_p10','edge_minus_centre','centroid_offset_radius'):
                    row[key] = pattern.iloc[index][key]
            rows.append(row); prior = value
        result = pd.DataFrame(rows)
        if result.empty: raise ValueError('No selected phase endpoints; detect current events first.')
        csv = _category_dir(session, 'tables')/'phase_endpoints.csv'
        result.to_csv(csv, index=False, float_format='%.12g')
        fig, axes = plt.subplots(1, 2, figsize=(8,3.2), constrained_layout=True)
        try:
            for phase, color in (('plating','#ce7b26'),('stripping','#357f8e')):
                sub = result.loc[result.phase == phase]
                axes[0].plot(sub.elapsed_h.to_numpy(), sub.value.to_numpy(), 'o-', color=color, label=phase.capitalize())
                for row in sub.itertuples(): axes[0].annotate(row.label,(row.elapsed_h,row.value),xytext=(3,4),textcoords='offset points',fontsize=8)
            sub = result.loc[result.phase == 'stripping']
            axes[1].plot(sub.elapsed_h.to_numpy(), sub.value_minus_initial_ocv.to_numpy(), 'o-', color='#357f8e')
            axes[1].axhline(0,color='0.6',lw=.6)
            unit = session.artifacts.get('image_unit') or 'a.u.'
            unit = 'µm' if unit in {'um','µm','μm'} else unit
            axes[0].set(xlabel='Elapsed time (h)',ylabel=f'Endpoint value ({unit})')
            axes[0].legend(fontsize=8)
            axes[1].set(xlabel='Elapsed time (h)',ylabel=f'Stripping end − initial OCV ({unit})')
            if not np.isfinite(ocv): axes[1].text(.5,.5,'No observed initial OCV',transform=axes[1].transAxes,ha='center')
            for axis in axes: axis.grid(alpha=.15)
            fig.suptitle('Recorded phase endpoints; interrupted/final phases require review',fontsize=9)
            path = _category_dir(session,'figures')/'phase_endpoints.png'
            fig.savefig(path,dpi=int(config.get('dpi',160)),bbox_inches='tight')
        finally:
            plt.close(fig)
        return SectionOutcome(self.id,SectionState.COMPLETED,f'Saved {len(result)} phase endpoints; no efficiency or Li identity inferred.',
            {'cycle_endpoints_table':result,'cycle_summary_csv':csv,'cycle_summary_figure':path})
