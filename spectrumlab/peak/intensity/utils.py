from enum import Enum, auto
from typing import Callable

from scipy import integrate, interpolate

from spectrumlab.alias import Array


class InterpolationKind(Enum):
    NEAREST = auto()
    LINEAR = auto()


def interpolate_grid(x_grid: Array[float], y_grid: Array[float], kind: InterpolationKind) -> Callable[[Array[float]], Array[float]]:
    """Interpolate the grid."""

    return interpolate.interp1d(
        x_grid, y_grid,
        kind={
            InterpolationKind.NEAREST: 'nearest',
            InterpolationKind.LINEAR: 'linear',
        }.get(kind),
        bounds_error=False,
        fill_value=0,
    )


def integrate_grid(x_grid: Array[float], y_grid: Array[float], position: float, interval: float, kind: InterpolationKind = InterpolationKind.LINEAR) -> float:
    """Calculate an intensity by interpolated grid."""

    f = interpolate_grid(x_grid, y_grid, kind=kind)

    return integrate.quad(
        f,
        a=position - interval/2,
        b=position + interval/2,
    )[0]
