from enum import Enum, auto

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from spectrumlab.curves import lanczos
from spectrumlab.types import Array


class InterpolationKind(Enum):

    NONE = auto()
    NEAREST = auto()
    LINEAR = auto()
    LANCZOS = auto()


def interpolate_lanczos(
    x_grid: Array[float],
    y_grid: Array[float],
    a: int = 3,
) -> Array[float]:
    """Smoothly interpolate with lanczos kernel.

    Params:
        a - window width
    """

    def inner(bias: Array[float]):
        nonlocal x_grid, y_grid, a  # noqa: F824

        x = np.arange(-a, a+1) - bias
        y = np.concatenate((y_grid[:a][::-1], y_grid[:], y_grid[-a:][::-1]))

        kernel = lanczos(x, x0=0, a=a)
        return np.convolve(y, kernel, 'valid')

    return inner


def interpolate(
    x: Array[float],
    y: Array[float],
    offset: float,
    kind: InterpolationKind = InterpolationKind.LINEAR,
) -> Array[float]:
    """Interpolate (x, y) values with selected kind of interpolation.

    Params:
        offset - shift of y values;
        kind - method of interpolation.
    """

    match kind:
        case InterpolationKind.NONE:
            return y

        case InterpolationKind.NEAREST | InterpolationKind.LINEAR:
            return interp1d(
                x, y,
                kind=kind,
                bounds_error=False,
                fill_value=1,
            )(x - offset)

        case InterpolationKind.LANCZOS:
            return interpolate_lanczos(
                x, y,
                a=3,
            )(offset)

        case _:
            raise ValueError(f'Interpolation kind: {kind} is not supported!')


if __name__ == '__main__':

    for a in [1, 2, 3]:
        x = np.linspace(-4, 4, 1000)
        y = lanczos(x, a=a)
        plt.plot(
            x, y,
            label=f'Lanczos(a={a})',
        )

    plt.grid(linestyle=':')
    plt.legend()

    plt.show()
