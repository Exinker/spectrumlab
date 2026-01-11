from typing import TYPE_CHECKING

import numpy as np

from spectrumlab.peaks.analyte_peaks.intensity.estimators.base_intensity_estimator import (
    IntensityEstimatorABC,
)
from spectrumlab.picture.colors import COLOR_INTENSITY, Color
from spectrumlab.types import R

if TYPE_CHECKING:
    from spectrumlab.peaks.analyte_peaks.analyte_peak import AnalytePeak


class AmplitudeIntensityEstimator(IntensityEstimatorABC):

    def __init__(self, verbose: bool = False) -> None:
        super().__init__(verbose)

    @property
    def color(self) -> Color:
        return COLOR_INTENSITY['amplitude']

    def calculate(self, peak: 'AnalytePeak') -> R:

        value = np.max(peak.value)

        if self.verbose:
            print(f'Peak\'s intensity: {value}')

        return value
