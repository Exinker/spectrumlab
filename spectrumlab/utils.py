
import time

import numpy as np

from spectrumlab.alias import Array


def timeit(func):
    """Timeit decorator."""
    def wraper(*args, **kwargs):
        start_at = time.perf_counter()
        results = func(*args, **kwargs)
        print(f'{func.__name__} time: {time.perf_counter() - start_at}, sec')

        return results

    return wraper


def find(values: Array[float]) -> Array[float]:
    """Find index of non-zero values."""

    return np.array([i for i, value in enumerate(values) if value])


def mse(y: Array[float], y_hat: Array[float]) -> float:
    """Calculate MSE of observes values y and prodicted values $\hat{y}$."""

    return np.sqrt(np.sum((y - y_hat)**2))


def rmse(y: Array[float], y_hat: Array[float]) -> float:
    """Calculate relative MSE of observes values y and prodicted values $\hat{y}$."""

    return np.sqrt(np.sum(((y - y_hat) / y)**2))
