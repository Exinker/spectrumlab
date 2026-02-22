import numpy as np

from spectrumlab.types import Array


def find(values: Array[float]) -> Array[float]:
    """Find index of non-zero values."""

    return np.array([i for i, value in enumerate(values) if value])


def se(y: float | Array[float], y_hat: Array[float]) -> Array[float]:
    r"""Calculate squared error (SE) between true values $y$ and predicted values $\hat{y}$."""

    return np.square(y - y_hat)


def mse(y: float | Array[float], y_hat: Array[float]) -> float:
    r"""Calculate mean squared error (MSE) between true values $y$ and predicted values $\hat{y}$."""
    n = len(y_hat)

    xi = se(y, y_hat)
    return np.sqrt(np.sum(xi) / n**2)


def rmse(y: float | Array[float], y_hat: Array[float]) -> float:
    r"""Calculate relative mean squared error (RMSE) between true values $y$ and predicted values $\hat{y}$."""
    n = len(y_hat)

    xi = se(y, y_hat) / np.abs(y)
    return np.sqrt(np.sum(xi) / n**2)
