from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt

from spectrumlab.alias import Array


@dataclass
class ADC:
    resolution: int
    log: bool = field(default=False)

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
                label='$\hat{y}$',
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
    resolution = 3
    kind = 'sin'

    match kind:
        case 'sin':
            x = np.linspace(0, 4*np.pi, 1000)
            y = np.sin(x)
        case 'linear':
            x = np.linspace(0, 40, 1000)
            y = 1 + x
        case 'cubic':
            x = np.linspace(0, 40, 1000)
            y = 1 + 0.01*x**2
        case _:
            raise ValueError(f'kind: {kind} is not supported!')

    y_hat = ADC(resolution).quantize(y)

    plt.plot(x, y)
    plt.plot(x, y_hat)
    plt.plot(x, y - y_hat)
    plt.grid(color='grey', linestyle=':')
    plt.show()
