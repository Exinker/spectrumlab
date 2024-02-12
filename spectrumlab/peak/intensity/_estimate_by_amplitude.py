from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from spectrumlab.peak.analyte_peak import AnalytePeak


@dataclass
class AmplitudeIntensityConfig:

    @property
    def color(self) -> str:
        return '#2ca02c'


def estimate_intensity_by_amplitude(peak: 'AnalytePeak', config: AmplitudeIntensityConfig, verbose: bool = False) -> float:
    """Estimate analyte peak's intensity by amplitude."""

    # intensity
    value = np.max(peak.value)

    # verbose
    if verbose:
        print(f'Peak\'s intensity: {value}')

    #
    return value