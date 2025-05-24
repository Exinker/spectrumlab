from typing import Mapping, TYPE_CHECKING

from spectrumlab.peaks.intensity.abstract_calculator import AbstractIntensityCalculator
from spectrumlab.peaks.shape.utils import approx_peak
from spectrumlab.peaks.units import R
from spectrumlab.pictures.color import COLOR_INTENSITY, Color
from spectrumlab.types import Number


if TYPE_CHECKING:
    from spectrumlab.peaks.analyte_peak import AnalytePeak
    from spectrumlab.peaks.shape import PeakShape


class ApproxIntensityCalculator(AbstractIntensityCalculator):
    """Estimate analyte peak's intensity by approximation."""

    def __init__(
        self,
        shape: 'PeakShape',
        delta: Number = 1,
        by_tail: bool = False,
        verbose: bool = False,
        show: bool = False,
    ) -> None:
        super().__init__(verbose)

        self.shape = shape
        self.delta = delta  # span of peak's position
        self.by_tail = by_tail  # use the tail of peak for approximation
        self.show = show

        self._params = None

    @property
    def color(self) -> Color:
        return COLOR_INTENSITY['shape']

    @property
    def params(self) -> Mapping[str, float]:
        if self._params is None:
            raise ValueError('Calculate params before!')

        return self._params

    def calculate(self, peak: 'AnalytePeak') -> R:

        self._params = approx_peak(
            peak=peak,
            shape=self.shape,
            delta=self.delta,
            by_tail=self.by_tail,
            show=self.show,
        )

        value = self.params['intensity']

        if self.verbose:
            print(f'Peak\'s intensity: {value}')

        return value
