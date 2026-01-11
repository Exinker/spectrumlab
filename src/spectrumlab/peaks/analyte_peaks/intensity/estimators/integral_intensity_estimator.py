from typing import TYPE_CHECKING

from spectrumlab.grids import Grid, InterpolationKind, integrate_grid
from spectrumlab.peaks.analyte_peaks.intensity.estimators.base_intensity_estimator import (
    IntensityEstimatorABC,
)
from spectrumlab.picture.colors import COLOR_INTENSITY, Color
from spectrumlab.types import Number, R

if TYPE_CHECKING:
    from spectrumlab.peaks.analyte_peaks.analyte_peak import AnalytePeak


class IntegralIntensityEstimator(IntensityEstimatorABC):
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

        match self.kind:
            case InterpolationKind.NEAREST:
                return COLOR_INTENSITY['nearest']
            case InterpolationKind.LINEAR:
                return COLOR_INTENSITY['linear']

        raise ValueError(f'color: {self.kind} is not supported!')

    def calculate(self, peak: 'AnalytePeak') -> R:

        value = integrate_grid(
            grid=Grid(x=peak.number, y=peak.value, units=Number),
            position=peak.position,
            interval=self.interval,
            kind=self.kind,
        )

        if self.verbose:
            print(f'Peak\'s intensity: {value}')

        return value
