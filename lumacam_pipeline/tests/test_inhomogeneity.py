from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from lumacam_pipeline.inhomogeneity import measure_pattern
from lumacam_pipeline.measurements import region_mask
from lumacam_pipeline.visualization import open_visualization, load_visualization_settings
from lumacam_pipeline.visualization_gui import VisualizationGUI
from lumacam_pipeline.section import SectionState


class PatternTests(unittest.TestCase):
    def test_uniform_and_geometry_specific_patterns(self):
        shape = (21, 21); roi = {'region': 'circle', 'roi': {'x0': 10, 'y0': 10, 'r0': 9}}
        mask = region_mask(shape, roi); yy, xx = np.indices(shape)
        uniform = np.ones(shape)
        ring = np.where(np.hypot(xx-10, yy-10) >= 7.2, 3., 1.)
        one_side = np.where(xx >= 10, 2., 0.)
        values = measure_pattern([uniform, ring, one_side], [None]*3, roi)
        self.assertAlmostEqual(values.cv[0], 0.)
        self.assertAlmostEqual(values.p90_minus_p10[0], 0.)
        self.assertAlmostEqual(values.edge_minus_centre[0], 0.)
        self.assertAlmostEqual(values.centroid_offset_radius[0], 0.)
        self.assertAlmostEqual(values.edge_minus_centre[1], 2.)
        self.assertGreater(values.centroid_offset_radius[2], .1)
        self.assertTrue(values.coverage_fraction.isna().all())
        self.assertEqual(values.valid_pixels.iloc[0], mask.sum())

    def test_signed_nan_threshold_and_floor(self):
        image = np.array([[-1., 0.], [1., np.nan]])
        result = measure_pattern([image], [None], {}, {'active_threshold': .5}).iloc[0]
        self.assertEqual(result['mean'], 0.)
        self.assertTrue(np.isnan(result.cv))
        self.assertAlmostEqual(result.coverage_fraction, 1/3)
        self.assertEqual(result.cv_active, 0.)
        self.assertIn('nonfinite_pixels', result.quality_flags)
        self.assertIn('cv_undefined', result.quality_flags)
        image = np.full((2, 2), .1)
        self.assertTrue(np.isnan(measure_pattern([image], [None], {}, {'min_mean_signal': .2}).cv.iloc[0]))
        self.assertTrue(np.isfinite(measure_pattern([image], [None], {}).cv.iloc[0]))
        with self.assertRaises(ValueError):
            measure_pattern([image], [None], {}, {'active_threshold': -.5})

    def test_partial_circle_and_non_circle(self):
        image = np.ones((10, 10))
        truncated = measure_pattern([image], [None], {'roi': {'x0': 0, 'y0': 0, 'r0': 3}}).iloc[0]
        self.assertIn('circle_truncated_by_image', truncated.quality_flags)
        plain = measure_pattern([image], [None], {}).iloc[0]
        self.assertTrue(np.isnan(plain.edge_minus_centre))
        self.assertTrue(np.isnan(plain.centroid_offset_radius))

    def test_zero_frame_is_retained_but_does_not_create_a_false_trend(self):
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'images'; source.mkdir()
            for i, value in enumerate((1, 0, 2)):
                tifffile.imwrite(source/f'{i}.tif', np.full((8, 8), value, np.float32), metadata=None)
            view = open_visualization(source, output_root=Path(folder)/'out', config={'reports': {'enabled': False}})
            with patch('matplotlib.pyplot.close'):
                outcome = view.run('inhomogeneity')
                self.assertEqual(outcome.state, SectionState.COMPLETED, outcome.message)
                figure = plt.gcf()
                self.assertTrue(np.isnan(figure.axes[1].lines[0].get_ydata()[1]))
                self.assertEqual(view.artifacts.require('inhomogeneity_table').p90_minus_p10.iloc[1], 0)
            plt.close('all')

    def test_session_export_and_gui_both_nci_modes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root/'images'; source.mkdir()
            start = datetime(2026, 1, 1, tzinfo=timezone.utc)
            for i in range(3):
                tifffile.imwrite(source/f'{i}.tif', np.full((12, 12), i+1, np.float32), metadata=None,
                    description=f'AcquisitionTime:{(start+timedelta(minutes=i)).isoformat()}')
            config = {'reports': {'enabled': False}, 'input': {'unit': 'um'},
                      'visualization': {'export_layout': 'flat'},
                      'image_measurements': {'region': 'circle', 'roi': {'x0': 6, 'y0': 6, 'r0': 4}},
                      'time_window': {'enabled': True, 'start': .01, 'end': .03}}
            view = open_visualization(source, output_root=root/'output', config=config)
            result = view.run('inhomogeneity')
            self.assertEqual(result.state, SectionState.COMPLETED, result.message)
            table = pd.read_csv(view.artifacts.require('inhomogeneity_csv_path'))
            self.assertEqual(len(table), 3)  # Window only changes figure, not calculations.
            self.assertTrue(table.cv.eq(0).all())
            self.assertTrue(view.artifacts.require('inhomogeneity_figure_path').is_file())
            restored = load_visualization_settings(view.artifacts.require('results_dir')/'settings.ini')
            self.assertIn('inhomogeneity', restored.sections)
            for mode in ('exsitu', 'operando_nci'):
                gui = VisualizationGUI(mode)
                self.assertIn('Plating inhomogeneity', [gui.accordion.get_title(i) for i in range(len(gui.accordion.children))])
            gui = VisualizationGUI('tof')
            self.assertNotIn('Plating inhomogeneity', [gui.accordion.get_title(i) for i in range(len(gui.accordion.children))])


if __name__ == '__main__':
    unittest.main()
