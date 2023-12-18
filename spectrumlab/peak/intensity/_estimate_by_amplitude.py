
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spectrumlab.peak.analyte_peak import AnalytePeak


@dataclass
class AmplitudeIntensityConfig:

    @property
    def color(self) -> str:
        return '#2ca02c'


def estimate_intensity_by_amplitude(peak: 'AnalytePeak', config: AmplitudeIntensityConfig, verbose: bool = False) -> float:
    """Estimate analyte peak's intensity by amplitude."""

    value = peak.amplitude

    # verbose
    if verbose:
        print(f'Peak\'s intensity: {value}')

    #
    return value