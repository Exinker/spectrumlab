from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from spectrumlab.grid import Grid, InterpolationKind, integrate_grid
from spectrumlab.picture import COLOR_INTENSITY
from spectrumlab.typing import Number

if TYPE_CHECKING:
    from spectrumlab.peak.analyte_peak import AnalytePeak


@dataclass(frozen=True)
class IntegralIntensityConfig:
    interval: float = field(default=3)
    kind: InterpolationKind = field(default=InterpolationKind.LINEAR)

    @property
    def color(self) -> str:

        if self.kind == InterpolationKind.NEAREST:
            return COLOR_INTENSITY['nearest']
        if self.kind == InterpolationKind.LINEAR:
            return COLOR_INTENSITY['linear']

        raise ValueError(f'color: {self.kind} is not supported!')


def estimate_intensity_by_integral(peak: 'AnalytePeak', config: IntegralIntensityConfig, verbose: bool = False) -> float:
    """Estimate analyte peak's intensity by integration.

    TODO: to check clipped values?
    """

    # intensity
    value = integrate_grid(
        grid=Grid(x=peak.number, y=peak.value, units=Number),
        position=peak.position,
        interval=config.interval,
        kind=config.kind,
    )

    # verbose
    if verbose:
        print(f'Peak\'s intensity: {value}')

    #
    return value
