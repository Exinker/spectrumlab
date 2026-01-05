from enum import Enum, auto
from typing import Callable

from scipy import interpolate

from spectrumlab.grids import Grid
from spectrumlab.types import Array, T


class InterpolationKind(Enum):

    NEAREST = auto()
    LINEAR = auto()


def interpolate_grid(
    grid: Grid,
    kind: InterpolationKind,
) -> Callable[[Array[T]], Array[float]]:
    """Interpolate the grid."""

    return interpolate.interp1d(
        grid.x, grid.y,
        kind={
            InterpolationKind.NEAREST: 'nearest',
            InterpolationKind.LINEAR: 'linear',
        }.get(kind),
        bounds_error=False,
        fill_value=0,
    )
