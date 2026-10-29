import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lumacam_pipeline import backends, create_session, discover_dataset
from lumacam_pipeline.batch import _prepare_shared_reference, _reference_source_files, validate_batch
from lumacam_pipeline.section import SectionState
from test_lumacam_pipeline import make_markerless_proposal


class AggregateSeparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.proposal, self.root, self.experiment, self.runs = make_markerless_proposal(
            Path(self.temp.name), variants=("3to1000", "5to15"), values=(1., 3.))

    def relocate(self):
        for source in (self.root / "final").glob("*/*/*_sumAllRuns.tiff"):
            target = self.root / "derived" / self.experiment / "summed" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)
            sources = list((self.root / "final" / self.experiment / source.parent.name).glob("*.tiff"))
            manifest = self.root / "metadata" / self.experiment / "summed" / f"aggregation_manifest_{source.parent.name}.json"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(json.dumps({"outputs": [str(target)], "inputs": {str(target): [str(p) for p in sources]}}))

    def test_old_and_new_discovery_keep_original_variants_and_provenance(self):
        before = discover_dataset(self.proposal)
        self.relocate()
        after = discover_dataset(self.proposal)
        self.assertEqual(before.image_products(), after.image_products())
        for variant in ("3to1000", "5to15"):
            products = after.image_products(variant=variant, include_aggregates=True)
            self.assertEqual(len(products), 3)
            aggregate = [p for p in products if p.aggregate][0]
            self.assertEqual(aggregate.kind, "final")
            self.assertEqual(aggregate.variant, variant)
            self.assertIsNone(aggregate.run)
            self.assertEqual(len(aggregate.source_paths), 2)
        photon = self.root / "derived" / self.experiment / "summed/exp001_photons_256_sumAllRuns.tiff"
        tifffile.imwrite(photon, np.ones((6, 6)))
        dataset = discover_dataset(self.proposal)
        self.assertEqual(len(dataset.image_products(include_aggregates=True)), 6)
        self.assertEqual([p.kind for p in dataset.products if p.path == photon], ["derived_photon"])

    def test_preview_selection_does_not_enter_loading_average_or_batch(self):
        for moved in (False, True):
            if moved:
                self.relocate()
            with self.subTest(moved=moved):
                config = {"project": {"data_path": str(self.proposal), "variant": "3to1000", "include_aggregates": True}}
                session = create_session("nci_images_only", config)
                for section in ("project", "images", "average"):
                    result = session.run(section)
                    self.assertEqual(result.state, SectionState.COMPLETED, result.message)
                self.assertEqual(len(session.artifacts.require("aggregate_products")), 1)
                self.assertEqual(len(session.artifacts.require("source_products")), 2)
                self.assertEqual(session.artifacts.require("source_tiff_count"), 2)
                np.testing.assert_allclose(session.artifacts.require("average_image"), 2.)
                validation = validate_batch(self.proposal, [self.experiment], config=config, sections=("images", "average"))
                self.assertEqual(validation.rows[0].image_count, 2)
                self.assertTrue(validation.rows[0].valid, validation.rows[0].message)

    def test_reference_cache_unchanged_by_sum_move_or_preview_content(self):
        reference = self.root / "final" / self.experiment
        config = {"project": {"variant": "3to1000"}, "corrections": {"reference_dir": str(reference), "reference_bin_x": 1, "reference_bin_y": 1}}
        log = Path(self.temp.name) / "reference.log"
        before_inputs = _reference_source_files(reference, "3to1000")
        before = _prepare_shared_reference(self.proposal, config, log)
        self.assertEqual(json.loads((before.parent / "ob_preparation.json").read_text())["source_count"], 2)
        self.relocate()
        preview = next((self.root / "derived").rglob("*_3to1000_sumAllRuns.tiff"))
        tifffile.imwrite(preview, np.full((6, 6), 10000.))
        self.assertEqual(before_inputs, _reference_source_files(reference, "3to1000"))
        self.assertEqual(before, _prepare_shared_reference(self.proposal, config, log))
        np.testing.assert_allclose(tifffile.imread(before), 2.)

    def test_only_aggregate_reference_is_invalid_and_cannot_be_cached(self):
        reference = self.root / "final" / self.experiment
        for file in reference.rglob("*.tiff"):
            if "_sumAllRuns" not in file.name:
                file.unlink()
        config = {"project": {"variant": "3to1000"}, "corrections": {"reference_dir": str(reference)}}
        validation = validate_batch(self.proposal, [self.experiment], config=config, sections=("corrections", "images"))
        self.assertTrue(any("no individual" in message for message in validation.common_errors))
        with self.assertRaisesRegex(ValueError, "No individual"):
            _prepare_shared_reference(self.proposal, config, Path(self.temp.name) / "log")

    def test_flat_sample_ob_bg_and_corrected_loading_exclude_preview(self):
        folder = Path(self.temp.name) / "flat"
        folder.mkdir()
        for name, value in (("run_a.tiff", 1), ("run_b.tif", 3), ("exp_sumImage_3to1000_sumAllRuns.tif", 1000)):
            tifffile.imwrite(folder / name, np.full((6, 6), value, dtype=np.uint32),
                extratags=[(65002, "s", 0, "2026-09-13 12:00:00", False)])
        backend = backends.nci_data()
        self.assertEqual(len(backend._matching_tiff_files(str(folder))), 2)
        np.testing.assert_allclose(backend.compute_background_image(str(folder)), 2.)
        self.assertEqual(len(backends.exsitu_pipeline()._tiff_files(folder)), 2)
        self.assertEqual(len(backends.nci_plots().load_corrected_images_with_time(str(folder))[1]), 2)
        with self.assertRaisesRegex(ValueError, "preview"):
            _reference_source_files(folder / "exp_sumImage_3to1000_sumAllRuns.tif", "3to1000")

    def test_legacy_run_contamination_is_excluded_from_correction_and_ob(self):
        reference = Path(self.temp.name) / "legacy"
        run = reference / "run001/final"
        run.mkdir(parents=True)
        tifffile.imwrite(run / "run001_sumImage_3to1000.tiff", np.ones((6, 6)))
        tifffile.imwrite(run / "legacy_sumImage_3to1000_sumAllRuns.tiff", np.full((6, 6), 900.))
        module = backends.nci_data()
        self.assertEqual(module.count_experiment_images(reference, file_filter="3to1000"), 1)
        series = module.load_reference_image_series(reference, file_filter="3to1000")
        self.assertEqual(len(series), 1)
        np.testing.assert_allclose(series[0], 1.)


if __name__ == "__main__":
    unittest.main()
