from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lumacam_pipeline.legacy_backend.operando_plots import capacity_from_current_sign_half_cycles, plot_potential_and_thickness
from lumacam_pipeline.quality import pixel_diagnostics
from lumacam_pipeline.inhomogeneity import spatial_diagnostics
from lumacam_pipeline.event_selection import current_events
from lumacam_pipeline.processed_variant import derive_background_variant
from lumacam_pipeline.handover import load_processed_images, file_digest
from lumacam_pipeline.visualization import open_visualization,load_visualization_settings,save_visualization_settings
from lumacam_pipeline.reports import create_report
from lumacam_pipeline.geometry import estimate_electrode_center
from lumacam_pipeline.processing import create_processing_session


class ScientificReviewTests(unittest.TestCase):
    def test_absent_beamhistory_never_signs_the_working_directory(self):
        from lumacam_pipeline.batch import _input_files
        with patch.object(Path, 'rglob', side_effect=AssertionError('Must not scan cwd')):
            for value in ('', '   ', None):
                self.assertEqual(_input_files(value), [])

    def test_processing_publishes_merged_metadata_exposure_bounds(self):
        import json
        from test_lumacam_pipeline import make_markerless_proposal
        with tempfile.TemporaryDirectory() as temp:
            proposal,_,experiment,_=make_markerless_proposal(Path(temp))
            session=create_processing_session('exsitu',{'reports':{'enabled':False},
                'project':{'data_path':str(proposal),'experiment':experiment,'variant':'3to1000'},
                'images':{'frames_per_merge':2}})
            for step in ('project','corrections','images'):
                out=session.run(step);self.assertEqual(out.state.value,'completed',out.message)
            root=session.artifacts.require('results_dir')
            payload=json.loads((root/'processing/processed_dataset.json').read_text())
            self.assertEqual(payload['frames'][0]['exposure_start_utc'],'2026-08-20T12:00:00+00:00')
            self.assertEqual(payload['frames'][0]['exposure_end_utc'],'2026-08-20T12:01:30+00:00')
            self.assertIsNotNone(load_processed_images(root)['exposure_bounds'][0])

    def test_fixed_known_diameter_finds_a_shifted_boundary(self):
        yy,xx=np.indices((90,90));disk=np.where((xx-45)**2+(yy-44)**2<=20**2,3.,0.)
        result=estimate_electrode_center([disk*0,disk],{'x0':41,'y0':42,'r0':20},
                                        pixel_size_um=100,diameter_mm=4,max_shift_px=10)
        self.assertTrue(result['accepted'])
        self.assertEqual(result['roi'],{'x0':45.,'y0':44.,'r0':20.})

    def test_small_bright_patch_does_not_redefine_a_large_electrode(self):
        yy,xx=np.indices((100,100));patch=np.where((xx-50)**2+(yy-50)**2<=5**2,3.,0.)
        result=estimate_electrode_center([patch*0,patch],{'x0':50,'y0':50,'r0':25},
                                        pixel_size_um=100,diameter_mm=5,max_shift_px=10)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['roi'],{'x0':50.,'y0':50.,'r0':25.})

    def test_repeated_equal_plating_stripping_has_no_artificial_baseline(self):
        actual=capacity_from_current_sign_half_cycles([0,.5,0,.5,0,.5],[-1,-1,1,1,-1,-1])
        np.testing.assert_allclose(actual,[0,.5,.5,0,0,.5])

    def test_rest_counter_resets_and_current_deadband(self):
        actual=capacity_from_current_sign_half_cycles([0,.5,.5,0,.1,.5,0,.2],[-1,-1,0,.001,1,1,-1,-1],zero_current=.01)
        np.testing.assert_allclose(actual,[0,.5,.5,.5,.4,0,0,.2],atol=1e-12)

    def test_same_polarity_counter_reset_and_file_restart(self):
        actual=capacity_from_current_sign_half_cycles([.1,.2,0,.1],[-1]*4,source_ids=['a','a','b','b'])
        np.testing.assert_allclose(actual,[.1,.2,.2,.3])

    def test_missing_samples_stay_missing_without_resetting_net_charge(self):
        actual=capacity_from_current_sign_half_cycles([0,.5,np.nan,0,.2],[-1,-1,1,1,1])
        np.testing.assert_allclose(actual,[0,.5,np.nan,.5,.3],equal_nan=True)

    def test_outlier_flag_does_not_filter_the_input(self):
        image=np.zeros((30,30));image[12,12]=170
        original=image.copy();stats,flags=pixel_diagnostics(image.ravel())
        self.assertIn('suspected_bright_outlier',flags)
        self.assertGreater(stats['top_pixel_variance_fraction'],.9)
        np.testing.assert_array_equal(image,original)
        _,flags=pixel_diagnostics(np.full(500,-1.))
        self.assertIn('background_dominated_roi',flags)

    def test_radial_profiles_and_sensitivity_do_not_redefine_roi(self):
        roi={'region':'circle','roi':{'x0':15,'y0':15,'r0':10}}
        radial,sensitivity=spatial_diagnostics([np.full((31,31),2.)],roi,[0],bins=5)
        np.testing.assert_allclose(radial['mean'],2)
        np.testing.assert_allclose(sensitivity['mean'],2)
        self.assertEqual(roi['roi']['r0'],10)
        self.assertEqual(sensitivity.radius_factor.tolist(),[.9,1.,1.1])

    def test_ec_gap_never_removes_valid_image_measurements(self):
        import matplotlib.pyplot as plt
        start=datetime(2026,1,1,tzinfo=timezone.utc)
        ec=[start+timedelta(seconds=i) for i in (0,10,1190,1200)]
        images=[start+timedelta(seconds=i) for i in range(0,1201,60)]
        fig=plot_potential_and_thickness(ec,[0,.1,.2,.3],img_datetime_series=images,
            avg_thickness_series=np.arange(21),plot_thickness=True,thickness_traces='nci',
            current_series=[0,-1,1,0],image_gap_minutes=5,show=False)
        try:
            np.testing.assert_array_equal(fig.axes[1].lines[0].get_ydata(),np.arange(21))
            self.assertEqual(len(fig.axes),3)
            self.assertIn('EC data gap',[t.get_text() for t in fig.axes[0].texts])
        finally:plt.close(fig)

    def test_exposure_crossing_phase_boundary_is_not_an_endpoint_image(self):
        start=datetime(2026,1,1,tzinfo=timezone.utc)
        times=[start+timedelta(seconds=i*10) for i in range(13)]
        images=[times[4],times[6]]
        exposures=[(times[2],times[5]),(times[5],times[7])]
        context=current_events(times,[0]*3+[-1]*5+[1]*5,images,min_phase_seconds=20,
                               plating_current_sign=-1,image_exposure_bounds=exposures)
        plating=[e for e in context['events'] if e['kind']=='plating']
        self.assertEqual(plating[0]['image_index'],1)
        self.assertEqual(plating[0]['timing_basis'],'full exposure bounds')

    def test_signed_derivation_is_portable_and_preserves_originals(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'input';source.mkdir()
            for i,v in enumerate((1,3,4)):
                tifffile.imwrite(source/f'{i}.tif',np.full((12,12),v,np.float32),metadata=None,
                    description=f'AcquisitionTime:2026-01-01T00:0{i}:00Z')
            bg=root/'background.tif';tifffile.imwrite(bg,np.full((12,12),2,np.float32))
            before={p:file_digest(p) for p in [*source.iterdir(),bg]}
            target=derive_background_variant(source,bg,root/'variant',experiment='sample',variant='3to9999',
                                             pixel_size_um=55,unit='um')
            loaded=load_processed_images(target)
            np.testing.assert_allclose([a.mean() for a in loaded['images']],[-1,1,2])
            moved=root/'moved';target.rename(moved)
            self.assertEqual(len(load_processed_images(moved)['images']),3)
            self.assertEqual(before,{p:file_digest(p) for p in before})
            with self.assertRaises(FileExistsError):
                derive_background_variant(source,bg,moved,experiment='sample',variant='3to9999')

    def test_quality_cycles_and_concise_report_use_recorded_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'input';source.mkdir()
            start=datetime(2026,1,1,tzinfo=timezone.utc)
            times=[start+timedelta(seconds=i*30) for i in range(9)]
            for i in range(9):
                tifffile.imwrite(source/f'{i}.tif',np.full((24,24),.1+i*.01,np.float32),metadata=None,
                    description=f'AcquisitionTime:{times[i].isoformat()}')
            view=open_visualization(source,preset='operando_nci',config={'reports':{'enabled':False},
                'visualization':{'export_layout':'flat'},'input':{'unit':'um'},
                'image_measurements':{'roi':{'x0':12,'y0':12,'r0':10}},'event_detection':{'plating_current_sign':-1},
                'combined_plot':{'potential_ylim':None,'thickness_traces':'nci'}})
            for key,value in {'echem_times':times,'echem_current':[0,0,-1,-1,-1,1,1,1,0],
                              'echem_potential':np.zeros(9),'echem_thickness':np.zeros(9)}.items():
                view.artifacts.put(key,value,'fixture')
            for section in ('image_measurements','event_detection','selection','inhomogeneity','cycle_summary','image_plot'):
                out=view.run(section);self.assertEqual(out.state.value,'completed',out.message)
            self.assertTrue(view.artifacts.has('frame_quality_table'))
            with patch('lumacam_pipeline.section.PipelineSession.run',side_effect=AssertionError('Report must not analyze')):
                report=create_report(view.artifacts.require('results_dir'),style='concise')
            text=report['markdown'].read_text()
            self.assertIn('Recorded phase endpoints',text)
            self.assertIn('Radial deposition profiles',text)
            self.assertIn('Electrode boundary is unconfirmed',text)
            self.assertIn('Settings and numerical audit',text)
            self.assertTrue(report['pdf'].read_bytes().startswith(b'%PDF'))
            restored=load_visualization_settings(save_visualization_settings(view))
            restored.config['reports']['enabled']=False
            for key in ('echem_times','echem_current','echem_potential','echem_thickness'):
                restored.artifacts.put(key,view.artifacts.require(key),'fixture')
            for step in ('image_measurements','selection','combined_plot'):
                out=restored.run(step);self.assertEqual(out.state.value,'completed',out.message)


if __name__=='__main__':unittest.main()
