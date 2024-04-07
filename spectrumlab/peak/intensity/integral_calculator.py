from typing import TYPE_CHECKING

from spectrumlab.grid import Grid, InterpolationKind, integrate_grid
from spectrumlab.peak.intensity import AbstractIntensityCalculator
from spectrumlab.picture import COLOR_INTENSITY
from spectrumlab.typing import Number, Percent

if TYPE_CHECKING:
    from spectrumlab.peak.analyte_peak import AnalytePeak


class IntegralIntensityCalculator(AbstractIntensityCalculator):
    """Estimate analyte peak's intensity by integration.

    TODO: to check clipped values?
    """

    def __init__(self, interval: Number = 3, kind: InterpolationKind = InterpolationKind.LINEAR, verbose: bool = False) -> None:
        super().__init__(verbose)

        self.interval = interval
        self.kind = kind

    @property
    def color(self) -> str:

        if self.kind == InterpolationKind.NEAREST:
            return COLOR_INTENSITY['nearest']

        if self.kind == InterpolationKind.LINEAR:
            return COLOR_INTENSITY['linear']

        raise ValueError(f'color: {self.kind} is not supported!')

    def calculate(self, peak: 'AnalytePeak') -> Percent:

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
