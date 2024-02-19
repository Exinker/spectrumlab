from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from spectrumlab.core.grid import Grid, InterpolationKind
from spectrumlab.core.grid import integrate_grid
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
            return '#1f77b4'
        if self.kind == InterpolationKind.LINEAR:
            return '#ff7f0e'

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
