"""Optional image correction sections."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from .. import backends
from ..section import PipelineSection, SectionOutcome, SectionState


def _optional_path(value):
    text = str(value or "").strip()
    return str(Path(text).expanduser().resolve()) if text else None


class CorrectionSection(PipelineSection):
    id = "corrections"
    title = "Corrections"
    requires = ("dataset", "results_dir")
    produces = ("processed_image_dir", "correction_result", "nci_calibrated")

    def run(self, session, config):
        dataset = session.artifacts.require("dataset")
        results_dir = Path(session.artifacts.require("results_dir"))
        reference_dir = _optional_path(config.get("reference_dir"))
        beamhistory_path = _optional_path(config.get("beamhistory_path"))
        reference_beamhistory_path = _optional_path(config.get("reference_beamhistory_path"))
        prepared_reference_path = _optional_path(config.get("prepared_reference_path"))
        file_filter = str(config.get("file_filter") or dataset.selected_variant or "").strip() or None
        results_dir.mkdir(parents=True, exist_ok=True)

        common = {
            "beamhistory_path": beamhistory_path,
            "tpx_time": float(config.get("tpx_time", 2)),
            "n_tpx_per_image": int(config.get("n_tpx_per_image", 30)),
            "file_filter": file_filter,
            "reference_beam_integral_per_minute": config.get("reference_beam_integral_per_minute"),
            "reference_thickness_um": float(config.get("reference_thickness_um", 1.0)),
        }
        if dataset.layout_type == "flat":
            module = backends.exsitu_pipeline()
            result = module.run_exsitu_pipeline(
                str(dataset.input_path),
                ob_path=prepared_reference_path or reference_dir,
                results_dir=str(results_dir),
                ob_bin_y=1 if prepared_reference_path else int(config.get("reference_bin_y", 2)),
                ob_bin_x=1 if prepared_reference_path else int(config.get("reference_bin_x", 2)),
                background_correction=False,
                verbose=bool(config.get("verbose", True)),
                organized_output=True,
                **common,
            )
            processed_dir = result["obint_corrected_dir"]
            calibrated = bool(result.get("ob_corrected"))
        else:
            module = backends.nci_data()
            image_root = results_dir / "images"
            processing_root = results_dir / "processing"
            experiment_input = dataset.input_path
            if dataset.layout_root is not None:
                candidates = (
                    dataset.layout_root / "metadata" / dataset.experiment,
                    dataset.layout_root / "final" / dataset.experiment,
                    dataset.layout_root / "tpx3Files" / dataset.experiment,
                )
                experiment_input = next(
                    (path for path in candidates if path.is_dir()), dataset.input_path
                )
            result = module.run_operando_pipeline(
                exp_dir=str(experiment_input),
                reference_dir=reference_dir,
                reference_beamhistory_path=reference_beamhistory_path,
                reference_bin_y=int(config.get("reference_bin_y", 2)),
                reference_bin_x=int(config.get("reference_bin_x", 2)),
                reference_final_subfolder=str(config.get("reference_final_subfolder", "final")),
                clip_ob_to_small_roi_p5=bool(config.get("clip_reference", False)),
                ob_clip_roi=config.get("reference_clip_roi"),
                ob_clip_percentile=float(config.get("reference_clip_percentile", 5.0)),
                results_dir=str(results_dir),
                corrected_dir=str(image_root / f"obint_corrected_{dataset.experiment}"),
                uncorrected_dir=str(image_root / f"uncorrected_{dataset.experiment}"),
                metadata_path=str(processing_root / "processing_metadata.json"),
                processed_reference_path=prepared_reference_path,
                **common,
            )
            processed_dir = result["corrected_dir"]
            calibrated = bool(result.get("reference_corrected"))

        applied = []
        if result.get("intensity_corrected"):
            applied.append("beam intensity")
        if result.get("reference_corrected") or result.get("ob_corrected"):
            applied.append("reference")
        message = "Prepared image stack"
        message += f" with {', '.join(applied)} correction." if applied else " without optional beam/reference correction."
        return SectionOutcome(
            self.id,
            SectionState.COMPLETED,
            message,
            {
                "processed_image_dir": Path(processed_dir),
                "correction_result": result,
                "nci_calibrated": calibrated,
            },
        )


class BackgroundSection(PipelineSection):
    id = "background"
    title = "Background correction"
    requires = ("processed_image_dir", "results_dir", "experiment")
    produces = ("analysis_image_dir", "background_path", "background_result")

    def is_configured(self, config):
        return bool(config.get("enabled", False))

    def run(self, session, config):
        module = backends.nci_data()
        input_dir = Path(session.artifacts.require("processed_image_dir"))
        results_dir = Path(session.artifacts.require("results_dir"))
        experiment = session.artifacts.require("experiment")
        processing_dir = results_dir / "processing"
        output_dir = results_dir / "images" / f"final_{experiment}"
        processing_dir.mkdir(parents=True, exist_ok=True)
        background_path = processing_dir / f"background_{experiment}.tif"
        external_path = _optional_path(config.get("background_path"))
        if external_path:
            external = Path(external_path)
            if external.is_file():
                background = tifffile.imread(external)
            elif external.is_dir():
                background = module.compute_background_image(
                    str(external),
                    n_images=int(config.get("n_images", 10)),
                    compute_dtype=np.float32,
                )
            else:
                raise ValueError(f"Background path does not exist: {external}")
        else:
            background = module.compute_background_image(
                str(input_dir),
                n_images=int(config.get("n_images", 10)),
                compute_dtype=np.float32,
                metadata_path=str(processing_dir / "background_metadata.json"),
            )
        tifffile.imwrite(background_path, np.asarray(background, dtype=np.float32))
        result = module.background_subtract_directory(
            str(input_dir),
            str(output_dir),
            str(background_path),
            clip_negatives=bool(config.get("clip_negatives", True)),
            skip_existing=not bool(config.get("recompute", False)),
            metadata_path=str(processing_dir / "background_subtraction_metadata.json"),
        )
        return SectionOutcome(
            self.id,
            SectionState.COMPLETED,
            f"Background-corrected images are in {output_dir}",
            {
                "analysis_image_dir": output_dir,
                "background_path": background_path,
                "background_result": result,
            },
        )
