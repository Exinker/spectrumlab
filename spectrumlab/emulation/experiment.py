from dataclasses import dataclass, field

from spectrumlab.alias import Array, Micro
from spectrumlab.emulation.detector.characteristic.aperture import ApertureProfile
from spectrumlab.emulation.detector.linear_array_detector import Detector
from spectrumlab.emulation.device import Device
from spectrumlab.emulation.intensity import IntensityConfig

import warnings
warnings.filterwarnings('ignore')


@dataclass
class BaseEmittedExperimentConfig:

    # --------        emulation config        --------
    device: Device
    detector: Detector

    n_numbers: int
    n_frames: int

    line_width: Micro
    line_asymmetry: float
    line_ratio: float

    aperture_profile: ApertureProfile

    # --------        intensity config        --------
    intensity: IntensityConfig

    # --------        calibration curve config        --------
    n_probes: int
    n_parallels: int

    position: float
    concentration: float

    ref: Array[float] | None = field(default=None)

    # --------        others        --------
    background_level: float = field(default=0)
    concentration_ratio: float = field(default=1)

    @property
    def concentrations(self) -> tuple[float]:
        return tuple(reversed([self.concentration * (1/(2**(i))) for i in range(self.n_probes)]))


class EmittedExperimentConfigNaive(BaseEmittedExperimentConfig):
    '''Experiment's config (naive).'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class EmittedExperimentConfig(BaseEmittedExperimentConfig):
    '''Experiment's config.'''

    def __init__(self, *args, apparatus_width: Micro, apparatus_asymmetry: float, apparatus_ratio: float, **kwargs):
        super().__init__(*args, **kwargs)

        # --------        emulation config        --------
        self.apparatus_width = apparatus_width
        self.apparatus_asymmetry = apparatus_asymmetry
        self.apparatus_ratio = apparatus_ratio
