from typing import TYPE_CHECKING

from scipy import interpolate

from spectrumlab.peak.position.abstract_calculator import AbstractPositionCalculator
from spectrumlab.types import Number


if TYPE_CHECKING:
    from spectrumlab.peak.analyte_peak import AnalytePeak


class InterpolationPositionCalculator(AbstractPositionCalculator):
    """Estimate analyte peak's position by interpolation."""

    def __init__(self, verbose: bool = False) -> None:
        super().__init__(verbose=verbose)

    def calculate(self, peak: 'AnalytePeak') -> Number:
        line = peak.config.line

        value = interpolate.interp1d(
            peak.wavelength, peak.number,
            kind='linear',
            bounds_error=False,
            fill_value=0,
        )(line.wavelength).item()

        if self.verbose:
            print(f'position: {value}')

        return value
