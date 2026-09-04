"""Compatibility front end for the shared section-based LumaCam dashboard."""

from pathlib import Path

from lumacam_pipeline.gui import LumacamPipelineGUI


class ExSituPipelineGUI(LumacamPipelineGUI):
    """Open the common dashboard with the ex-situ preset selected."""

    def __init__(self, base_dir=None, history_path=None):
        del history_path  # Retained only for API compatibility.
        super().__init__(preset="exsitu")
        if base_dir is not None:
            self.data_path.value = str(Path(base_dir).expanduser().resolve())

    def display(self):
        return self.show()


def launch_exsitu_gui(base_dir=None, history_path=None):
    gui = ExSituPipelineGUI(base_dir=base_dir, history_path=history_path)
    return gui.display()
