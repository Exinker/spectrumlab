import matplotlib.pyplot as plt
import numpy as np

from spectrumlab.background.savitzky_golay_background import SavitzkyGolayBackgroundConfig
from spectrumlab.background.savitzky_golay_background import approximate_savitzky_golay


def test_approximate_savitzky_golay():
    tolerance = 1e-2

    n = 100
    x = np.linspace(-2*np.pi, +2*np.pi, n)
    y = np.sin(x)
    y_hat = approximate_savitzky_golay(
        y,
        mask=np.full(n, False),
        config=SavitzkyGolayBackgroundConfig(width=3, degree=1, width_min=2),
    )

    assert np.all(np.abs(y - y_hat) < tolerance)


if __name__ == '__main__':
    n = 100
    x = np.linspace(-2*np.pi, +2*np.pi, n)
    y = np.sin(x)
    y_hat = approximate_savitzky_golay(
        y,
        mask=np.full(n, False),
        config=SavitzkyGolayBackgroundConfig(width=3, degree=1, width_min=2),
    )

    # show
    plt.plot(
        x, y,
        color='red', linestyle='none', marker='.',
        label='$y$',
    )
    plt.plot(
        x, y_hat,
        color='black', linestyle='-',
        label=r'$\hat{y}$',
    )
    plt.plot(
        x, y - y_hat,
        color='black', linestyle=':',
        label=r'$error$',
    )
    plt.grid(color='grey', linestyle=':')
    plt.legend()
    plt.show()
