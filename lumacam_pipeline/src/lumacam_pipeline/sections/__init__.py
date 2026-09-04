"""Built-in selectable pipeline sections."""

from .corrections import BackgroundSection, CorrectionSection
from .electrochemistry import ElectrochemistrySection, SynchronizationSection
from .events import EventDetectionSection
from .images import AveragingSection, ImageLoadSection, ImageMeasurementSection, SelectionSection
from .nci import PlatingMetricsSection
from .outputs import (
    CombinedPlotSection,
    CombinedVideoSection,
    ImagePlotSection,
    ImageVideoSection,
    IndividualImagePlotSection,
    NciEventOutputsSection,
)
from .photon_hyperspectral import PhotonHyperspectralSection
from .project import ProjectSection
from .tof import TimeOfFlightCubeSection


def default_sections():
    """Return all built-in sections in display order, not execution order."""

    return [
        ProjectSection(),
        ImageLoadSection(),
        CorrectionSection(),
        BackgroundSection(),
        AveragingSection(),
        ImageMeasurementSection(),
        PlatingMetricsSection(),
        PhotonHyperspectralSection(),
        TimeOfFlightCubeSection(),
        ElectrochemistrySection(),
        SynchronizationSection(),
        EventDetectionSection(),
        SelectionSection(),
        ImagePlotSection(),
        IndividualImagePlotSection(),
        CombinedPlotSection(),
        NciEventOutputsSection(),
        ImageVideoSection(),
        CombinedVideoSection(),
    ]


__all__ = [
    "AveragingSection",
    "BackgroundSection",
    "CombinedPlotSection",
    "CombinedVideoSection",
    "CorrectionSection",
    "ElectrochemistrySection",
    "EventDetectionSection",
    "ImageLoadSection",
    "ImageMeasurementSection",
    "ImagePlotSection",
    "ImageVideoSection",
    "IndividualImagePlotSection",
    "NciEventOutputsSection",
    "PlatingMetricsSection",
    "PhotonHyperspectralSection",
    "ProjectSection",
    "SelectionSection",
    "SynchronizationSection",
    "TimeOfFlightCubeSection",
    "default_sections",
]
