import numpy as np
import matplotlib.pyplot as plt
import pytest

from spectrumlab.background.savitzky_golay_background import SavitzkyGolayBackgroundConfig, approximate_savitzky_golay


def test_approximate_savitzky_golay(tol: float = 1e-2):
    
    n = 100
    x = np.linspace(-2*np.pi, +2*np.pi, n)
    y = np.sin(x)
    y_hat = approximate_savitzky_golay(
        y,
        mask=np.full(n, False),
        config=SavitzkyGolayBackgroundConfig(width=3, degree=1, n_counts_min=2),
    )

    assert np.all(np.abs(y - y_hat) < tol)


if __name__ == '__main__':

    n = 100
    x = np.linspace(-2*np.pi, +2*np.pi, n)
    y = np.sin(x)
    y_hat = approximate_savitzky_golay(
        y,
        mask=np.full(n, False),
        config=SavitzkyGolayBackgroundConfig(width=3, degree=1, n_counts_min=2),
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
        label='$\hat{y}$',
    )
    plt.plot(
        x, y - y_hat,
        color='black', linestyle=':',
        label='$error$',
    )
    plt.grid(color='grey', linestyle=':')
    plt.legend()
    plt.show()
