from scipy import integrate

from spectrumlab.grids import Grid
from spectrumlab.grids.utils.interpolators import InterpolationKind, interpolate_grid


def integrate_grid(
    grid: Grid,
    position: float,
    interval: float,
    kind: InterpolationKind = InterpolationKind.LINEAR,
) -> float:
    """Integrate the grid in given `position` and `interval`."""

    return integrate.quad(
        interpolate_grid(grid, kind=kind),
        a=position - interval/2,
        b=position + interval/2,
    )[0]
