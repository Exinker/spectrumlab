
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from scipy import interpolate

from spectrumlab.alias import Number

if TYPE_CHECKING:
    from spectrumlab.peak.analyte_peak import AnalytePeak


@dataclass
class InterpolationPositionConfig:
    verbose: bool = field(default=False)


def estimate_position_by_interpolation(peak: 'AnalytePeak', config: InterpolationPositionConfig) -> Number:
    """Estimate analyte peak's position by interpolation."""
    line = peak.settings.line

    value = interpolate.interp1d(
        peak.wavelength, peak.number,
        kind='linear',
        bounds_error=False,
        fill_value=0,
    )(line.wavelength).item()

    #
    if config.verbose:
        print(f'position: {value}')

    #
    return value
