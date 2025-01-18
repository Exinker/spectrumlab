from typing import TYPE_CHECKING

from spectrumlab.grid import Grid, InterpolationKind, integrate_grid
from spectrumlab.peak.units import R
from spectrumlab.picture.color import COLOR_INTENSITY, Color
from spectrumlab.types import Number

from .calculator import AbstractIntensityCalculator

if TYPE_CHECKING:
    from spectrumlab.peak.analyte_peak import AnalytePeak


class IntegralIntensityCalculator(AbstractIntensityCalculator):
    """Estimate analyte peak's intensity by integration.

    TODO: to check clipped values?
    """

    def __init__(
        self,
        interval: Number = 3,
        kind: InterpolationKind = InterpolationKind.LINEAR,
        verbose: bool = False,
    ) -> None:
        super().__init__(verbose)

        self.interval = interval
        self.kind = kind

    @property
    def color(self) -> Color:

        if self.kind == InterpolationKind.NEAREST:
            return COLOR_INTENSITY['nearest']

        if self.kind == InterpolationKind.LINEAR:
            return COLOR_INTENSITY['linear']

        raise ValueError(f'color: {self.kind} is not supported!')

    def calculate(self, peak: 'AnalytePeak') -> R:

        value = integrate_grid(
            grid=Grid(x=peak.number, y=peak.value, units=Number),
            position=peak.position,
            interval=self.interval,
            kind=self.kind,
        )

        # verbose
        if self.verbose:
            print(f'Peak\'s intensity: {value}')

        #
        return value
