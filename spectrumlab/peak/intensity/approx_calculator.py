from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from spectrumlab.peak.intensity import AbstractIntensityCalculator
from spectrumlab.peak.shape.utils import approx_peak
from spectrumlab.picture import COLOR_INTENSITY
from spectrumlab.typing import Number, Percent

if TYPE_CHECKING:
    from spectrumlab.peak.analyte_peak import AnalytePeak
    from spectrumlab.peak.shape import PeakShape


class ApproxIntensityCalculator(AbstractIntensityCalculator):
    """Estimate analyte peak's intensity by approximation."""

    def __init__(self, shape: 'PeakShape', delta: Number = 1, by_tail: bool = False, verbose: bool = False, show: bool = False) -> None:
        super().__init__(verbose)

        self.shape = shape
        self.delta = delta  # span of peak's position
        self.by_tail = by_tail  # use the tail of peak for approximation
        self.show = show

    @property
    def color(self) -> str:
        return COLOR_INTENSITY['shape']

    def calculate(self, peak: 'AnalytePeak') -> Percent:

        params = approx_peak(
            peak=peak,
            shape=self.shape,
            delta=self.delta,
            by_tail=self.by_tail,
            show=self.show,
        )

        value = params['intensity']

        # verbose
        if self.verbose:
            print(f'Peak\'s intensity: {value}')

        #
        return value
