import numpy as np

from spectrumlab.grid import (
    Grid,
    InterpolationKind,
    integrate_grid,
    interpolate_grid,
)
from spectrumlab.peak.intensity import (
    AbstractIntensityCalculator,
    AmplitudeIntensityCalculator,
    ApproxIntensityCalculator,
    IntegralIntensityCalculator,
)
from spectrumlab.types import Array, Number


def _estimate_intensity(
    grid: Grid,
    mask: Array[bool],
    position: Number,
    calculator: AbstractIntensityCalculator,
) -> float:
    """Interface to estimate analyte peak's intensity."""

    if isinstance(calculator, AmplitudeIntensityCalculator):
        f = interpolate_grid(grid, kind=InterpolationKind.NEAREST)

        return f(position)

    if isinstance(calculator, IntegralIntensityCalculator):
        return integrate_grid(
            grid=grid,
            position=position,
            interval=calculator.interval,
            kind=calculator.kind,
        )

    if isinstance(calculator, ApproxIntensityCalculator):

        norm = np.dot(grid.y[~mask], calculator.shape(grid.x[~mask], position=position, intensity=1))
        value = np.dot(grid.y[~mask], grid.y[~mask]) / norm

        return value

    raise ValueError(f'calculate_intensity: calculator {calculator} is not supported yet!')
