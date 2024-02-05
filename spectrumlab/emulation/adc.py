"""
Analog-to-digital converter (ADC) for emulation.

Author: Vaschenko Pavel
 Email: vaschenko@vmk.ru
  Date: 2023.02.01
"""
from dataclasses import dataclass, field
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from spectrumlab.alias import Array


@dataclass
class ADC:
    resolution: int
    log: bool = field(default=False)

    # --------        handlers        --------
    def quantize(self, y: Array[float], show: bool = False) -> Array[float]:
        if self.log:
            y = np.log2(y)

        y0 = min(y)

        step = (max(y) - min(y)) / (2**self.resolution - 1)
        y_hat = y0 + step*np.floor((y - y0)/step + 1/2)

        #
        if show:
            fig, (ax_left, ax_right) = plt.subplots(ncols=2, figsize=(12, 4), tight_layout=True)

            ax_left.plot(
                y,
                color='black',
                label='$y$',
            )

            ax_left.plot(
                y_hat,
                color='red',
                label=r'$\hat{y}$',
            )
            ax_left.set_xlabel('x')
            ax_left.set_ylabel('y')
            ax_left.grid(color='grey', linestyle=':')
            ax_left.legend()

            ax_right.plot(
                y - y_hat,
                color='black',
            )
            ax_right.set_xlabel('x')
            ax_right.set_ylabel(r'$y - \hat{y}$')
            ax_right.grid(color='grey', linestyle=':')

            plt.show()

        return y_hat


if __name__ == '__main__':
    def _calculate_y(x: Array[float], kind: Literal['sin', 'linear', 'cubic']) -> Array[float]:
        if kind == 'sin':
            return np.sin(x)
        if kind == 'linear':
            return 1 + x
        if kind == 'cubic':
            return .1 + 0.01*x**2

        raise ValueError(f'kind: {kind} is not supported!')

    x = np.linspace(-2*np.pi, +2*np.pi, 1000)
    y = _calculate_y(x, kind='sin')
    y_hat = ADC(
        resolution=4,
    ).quantize(y)

    plt.plot(
        x, y,
        label=r'$y$',
    )
    plt.plot(
        x, y_hat,
        label=r'$\hat{y}$',
    )
    plt.plot(
        x, y - y_hat,
        label=r'error',
    )
    plt.grid(color='grey', linestyle=':')
    plt.legend()

    plt.show()
