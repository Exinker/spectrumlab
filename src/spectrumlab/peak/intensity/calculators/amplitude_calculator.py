from typing import TYPE_CHECKING

import numpy as np

from spectrumlab.peak.intensity.abstract_calculator import AbstractIntensityCalculator
from spectrumlab.peak.units import R
from spectrumlab.picture.color import COLOR_INTENSITY, Color


if TYPE_CHECKING:
    from spectrumlab.peak.analyte_peak import AnalytePeak


class AmplitudeIntensityCalculator(AbstractIntensityCalculator):

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
