import os

import matplotlib.pyplot as plt
import numpy as np

from spectrumlab.emulation.curve import voigt2pvoigt
from spectrumlab.emulation.line import PVoigtLineShape, VoigtLineShape
from spectrumlab.line import Line
from spectrumlab.typing import PicoMeter


def transform(line: Line, shape: VoigtLineShape, dx: PicoMeter = 1e-1, rx: PicoMeter = 10, show: bool = False, save: bool = False) -> PVoigtLineShape:
    """Approx voigt shape by pvoigt shape."""
    x = np.linspace(-rx, +rx, 2*int(rx/dx) + 1)

    params_hat = voigt2pvoigt(x, x0=0, sigma=shape.sigma, gamma=shape.gamma)
    shape_hat = PVoigtLineShape(*params_hat)

    # show
    if show:
        y = shape(x, 0, 1)
        y_hat = shape_hat(x, 0, 1)

        #
        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

        content = '\n'.join([
            f'{line}:',
            '',
            f'Doppler: {shape.g:.3f} [pm]',
            f'Collision: {shape.l:.3f} [pm]',
            f'FWHM: {shape.fwhm:.4f} [pm]',
        ])
        plt.text(
            0.05, 0.95,
            content,
            transform=ax.transAxes,
            ha='left', va='top',
        )

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

        plt.grid(color='grey', linestyle=':')
        plt.legend()

        # save
        if save:
            filedir = os.path.join('.', 'img')
            if not os.path.isdir(filedir):
                os.mkdir(filedir)

            filepath = os.path.join(filedir, f'{line}.png')
            plt.savefig(filepath)

        #
        plt.show()

    #
    return shape_hat
