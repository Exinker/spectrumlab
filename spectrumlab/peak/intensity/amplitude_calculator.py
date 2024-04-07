from typing import TYPE_CHECKING

import numpy as np

from spectrumlab.peak.intensity import AbstractIntensityCalculator
from spectrumlab.picture import COLOR_INTENSITY
from spectrumlab.typing import Percent

if TYPE_CHECKING:
    from spectrumlab.peak.analyte_peak import AnalytePeak


class AmplitudeIntensityCalculator(AbstractIntensityCalculator):

    def __init__(self, verbose: bool = False) -> None:
        super().__init__(verbose)

    @property
    def color(self) -> str:
        return COLOR_INTENSITY['amplitude']

    def calculate(self, peak: 'AnalytePeak') -> Percent:

        value = np.max(peak.value)

        # verbose
        if self.verbose:
            print(f'Peak\'s intensity: {value}')

        #
        return value
