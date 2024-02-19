import matplotlib.pyplot as plt
import numpy as np

from spectrumlab.emulation.curve import voigt2pvoigt
from spectrumlab.emulation.line import PVoigtLineShape, VoigtLineShape
from spectrumlab.typing import PicoMeter


def transform(shape: VoigtLineShape, dx: PicoMeter = 1e-2, rx: PicoMeter = 100, show: bool = False) -> PVoigtLineShape:
    """Approx voigt shape by pvoigt shape."""
    x = np.linspace(-rx, +rx, 2*int(rx/dx) + 1)

    params_hat = voigt2pvoigt(x, x0=0, sigma=shape.sigma, gamma=shape.gamma)
    shape_hat = PVoigtLineShape(*params_hat)

    # show
    if show:
        y = shape(x, 0, 1)
        y_hat = shape_hat(x, 0, 1)

        plt.plot(
            x, y,
            color='red', linestyle='none', marker='s', markersize=3,
            label=r'voigt pvoigt',
        )
        plt.plot(
            x, y_hat,
            label=r'pvoigt pvoigt',
            color='black', linestyle='-', linewidth=1,
        )
        plt.plot(
            x, y_hat - y,
            color='black', linestyle='none', marker='s', markersize=0.5,
            label=r'error',
        )

        plt.xlabel('$x$ $[pm]$')
        plt.ylabel('$f(x)$')

        plt.grid(linestyle=':')
        plt.legend()
        plt.show()

    #
    return shape_hat
