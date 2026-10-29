"""Result reports must show data and preserve the processing/analysis boundary."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lumacam_pipeline import create_report, open_visualization, create_processing_session
from lumacam_pipeline.cli import main
from lumacam_pipeline.handover import file_digest
from lumacam_pipeline.section import SectionState
from lumacam_pipeline.batch import run_batch
from test_lumacam_pipeline import make_markerless_proposal


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def profile(self, name="exp101", automatic=False, figures=True):
        inputs = self.root / name
        inputs.mkdir()
        for i, value in enumerate((0., 2., 4.)):
            tifffile.imwrite(inputs / f"frame_{i}.tiff", np.full((12, 12), value, np.float32), metadata=None,
                             description=f"AcquisitionTime:2026-08-20T12:0{i}:00+00:00")
        view = open_visualization(inputs, config={"reports": {"enabled": automatic}})
        sections = ["image_measurements", "selection", "image_plot"] if figures else ["image_measurements"]
        for section in sections:
            outcome = view.run(section)
            self.assertEqual(outcome.state, SectionState.COMPLETED, outcome.message)
        return view

    def test_markdown_pdf_contain_results_images_settings_and_warnings(self):
        view = self.profile()
        root = view.artifacts.require("results_dir")
        before = {p: file_digest(p) for p in root.rglob("*") if p.is_file()}
        result = create_report(root)
        md = result["markdown"].read_text()
        self.assertIn("Minimum / maximum | 0 / 4", md)
        self.assertIn("Zero-mean measurement frames", md)
        self.assertIn("Electrochemistry was not included", md)
        self.assertIn("image_measurements: settings used", md)
        self.assertIn("frames: 3", md)
        image_links = re.findall(r"!\[[^]]*\]\(([^)]+)\)", md)
        self.assertGreaterEqual(len(image_links), 2)
        for value in image_links:
            self.assertTrue((root / value).is_file())
        pdf = result["pdf"].read_bytes()
        self.assertTrue(pdf.startswith(b"%PDF-"))
        self.assertIn(b"/Subtype /Image", pdf)
        self.assertGreaterEqual(len(re.findall(rb"/Type /Page\b", pdf)), 4)
        for p, digest in before.items():
            self.assertEqual(file_digest(p), digest)
        self.assertEqual(result, create_report(root))

    def test_combined_report_variant_override_does_not_edit_sources(self):
        views = [self.profile(name) for name in ("exp101", "exp201")]
        roots = [v.artifacts.require("results_dir") for v in views]
        settings = [(p / "settings.ini").read_bytes() for p in roots]
        result = create_report(roots, output_dir=self.root / "summary", variant="3to9999", notes=["User-confirmed variant."])
        md = result["markdown"].read_text()
        self.assertIn("Datasets: 2; recorded frames: 6", md)
        self.assertEqual(md.count("3to9999 (user-confirmed"), 2)
        for p, original in zip(roots, settings):
            self.assertEqual((p / "settings.ini").read_bytes(), original)

    def test_interpretation_is_visible_and_withheld_when_evidence_changes(self):
        view = self.profile(figures=False)
        root = view.artifacts.require('results_dir')
        table = next((root/'tables').glob('*/image_measurements.csv'))
        notes = {'schema_version': 1, 'source_profiles': ['.'],
                 'evidence': [{'path': str(table.relative_to(root)), 'sha256': file_digest(table)}],
                 'summary': [{'heading': 'Observed rise', 'observation': 'Mean rises from 0 to 4.',
                              'conclusion': 'The recorded signal increased.'}],
                 'experiments': {root.name: {'observation': 'Three frames measured.',
                                           'conclusion': 'No repeated cells were measured.'}}}
        (root/'interpretation.json').write_text(json.dumps(notes))
        from matplotlib.backends.backend_pdf import PdfPages
        original = PdfPages.savefig
        printed = []
        def capture(pdf, figure, **kwargs):
            printed.extend(text.get_text() for text in figure.texts)
            return original(pdf, figure, **kwargs)
        with patch.object(PdfPages, 'savefig', capture):
            result = create_report(root, style='concise')
        self.assertIn('The recorded signal increased.', result['markdown'].read_text())
        self.assertTrue(any('Conclusion: The recorded signal increased.' in text for text in printed))
        table.write_text(table.read_text()+'\n')
        md = create_report(root, style='concise')['markdown'].read_text()
        self.assertIn('Scientific interpretation needs review', md)
        self.assertNotIn('The recorded signal increased.', md)

    def test_combined_reports_can_live_beside_a_visualization_folder(self):
        view = self.profile(automatic=True, figures=False)
        root = view.artifacts.require('results_dir')
        summary = self.root/'proposal_summary'
        view.config['reports'].update(combined_output=str(summary),
            combined_profiles_root=str(root.parent), combined_profiles=[root.name], title='Proposal results')
        out = view.run('selection')
        self.assertEqual(out.state, SectionState.COMPLETED)
        self.assertIn('Proposal results', (summary/'overview.md').read_text())

    def test_automatic_and_nonfatal_report_failure(self):
        view = self.profile(automatic=True, figures=False)
        root = view.artifacts.require("results_dir")
        self.assertTrue((root / "overview.pdf").is_file())
        original = (root / "overview.md").read_bytes()
        with patch("lumacam_pipeline.reports.create_report", side_effect=OSError("report disk unavailable")):
            outcome = view.run("selection")
        self.assertEqual(outcome.state, SectionState.COMPLETED)
        self.assertIn("Overview report failed", outcome.message)
        self.assertIn("report disk unavailable", view.artifacts.require("overview_error"))
        self.assertEqual((root / "overview.md").read_bytes(), original)

    def test_explicit_disable_and_user_files_preserved(self):
        view = self.profile(figures=False)
        root = view.artifacts.require("results_dir")
        self.assertFalse((root / "overview.md").exists())
        (root / "overview.md").write_text("Human-authored report")
        with self.assertRaisesRegex(ValueError, "not a generated"):
            create_report(root)
        self.assertEqual((root / "overview.md").read_text(), "Human-authored report")

    def test_changed_table_is_not_presented_as_verified_measurement(self):
        view = self.profile()
        root = view.artifacts.require("results_dir")
        path = next((root / "tables").glob("*/image_measurements.csv"))
        path.write_text(path.read_text().replace("4.0", "999.0"))
        md = create_report(root)["markdown"].read_text()
        self.assertIn("Missing or changed export excluded", md)
        self.assertNotIn("Minimum / maximum", md)

    def test_movable_report_assets_and_saved_exports_without_sources(self):
        view = self.profile()
        root = view.artifacts.require("results_dir")
        create_report(root)
        moved = self.root / "moved_profile"
        shutil.copytree(root, moved)
        root.parents[1].rename(self.root / "offline_inputs")
        result = create_report(moved)
        md = result["markdown"].read_text()
        self.assertIn("Minimum / maximum | 0 / 4", md)
        for link in re.findall(r"!\[[^]]*\]\(([^)]+)\)", md):
            self.assertTrue((moved / link).is_file())

    def test_cli_report_does_not_run_any_section(self):
        view = self.profile()
        root = view.artifacts.require("results_dir")
        with patch("lumacam_pipeline.section.PipelineSession.run", side_effect=AssertionError("must not process")):
            with redirect_stdout(io.StringIO()) as output:
                code = main(["--variant", "3to9999", "report", "--input", str(root)])
        self.assertEqual(code, 0, output.getvalue())
        self.assertTrue(Path(json.loads(output.getvalue())["pdf"]).is_file())

    def test_processing_and_batch_reports_use_saved_average(self):
        proposal, _, experiment, _ = make_markerless_proposal(self.root)
        config = {"project": {"data_path": str(proposal), "experiment": experiment, "variant": "3to1000"}}
        session = create_processing_session("exsitu", config)
        for section in ("project", "corrections", "images", "average"):
            outcome = session.run(section)
            self.assertEqual(outcome.state, SectionState.COMPLETED, outcome.message)
        root = session.artifacts.require("results_dir")
        md = (root / "overview.md").read_text()
        self.assertIn("Saved experiment-average image", md)
        self.assertIn("Frames contributing to saved average", md)
        self.assertIn("not scientific ROI measurements", md)
        result = run_batch(proposal, [experiment], preset="exsitu", config=session.config,
                           sections=["corrections", "images", "average"], batch_name="reports")
        self.assertEqual(result.failed_count, 0, result.results)
        self.assertEqual(result.report_error, "")
        self.assertTrue((result.batch_dir / "overview.pdf").is_file())

    def test_tof_saved_spectrum_and_integrated_map_are_embedded(self):
        path = self.root / "cube.npy"
        np.save(path, np.ones((4, 8, 8), dtype=np.float32))
        view = open_visualization(path, cube=True, preset="tof", config={"reports": {"enabled": False}})
        outcome = view.run("tof_cube")
        self.assertEqual(outcome.state, SectionState.COMPLETED, outcome.message)
        root = view.artifacts.require("results_dir")
        md = create_report(root)["markdown"].read_text()
        self.assertIn("Global ToF spectrum", md)
        self.assertIn("Integrated ToF image", md)
        self.assertIn("Spectral channels | 4", md)

    def test_report_assets_symlink_is_rejected(self):
        view = self.profile(figures=False)
        root = view.artifacts.require("results_dir")
        (root / "report_assets").symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            create_report(root)

    def test_dangling_asset_symlink_cannot_write_outside_report(self):
        view = self.profile()
        root = view.artifacts.require("results_dir")
        result = create_report(root)
        asset = next((root / "report_assets").glob("*.png"))
        external = self.root / "unrelated.png"
        asset.unlink()
        asset.symlink_to(external)
        with self.assertRaisesRegex(ValueError, "symlink"):
            create_report(root)
        self.assertFalse(external.exists())
        self.assertTrue(result["pdf"].is_file())

    def test_no_timing_nonfinite_values_are_explicit(self):
        inputs = self.root / "no_time"
        inputs.mkdir()
        for i, value in enumerate((1., float('nan'), 3.)):
            tifffile.imwrite(inputs / f"frame_{i}.tiff", np.full((8, 8), value, np.float32), metadata=None)
        view = open_visualization(inputs, config={"reports": {"enabled": False}})
        self.assertEqual(view.run("image_measurements").state, SectionState.COMPLETED)
        md = create_report(view.artifacts.require("results_dir"))["markdown"].read_text()
        self.assertIn("1 nonfinite measurements", md)
        self.assertIn("Acquisition times unavailable", md)
        self.assertNotIn("Image time range (UTC)", md)


if __name__ == "__main__":
    unittest.main()
