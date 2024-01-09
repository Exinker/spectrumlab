import numpy as np


def distance(xi: float, xi_hat: float, is_relative: bool = False) -> float:
    """Calculate a distance (relative, in optionally) between `xi` and `xi_hat`."""
    if is_relative:
        return np.abs((xi_hat - xi) / xi)
    return np.abs(xi_hat - xi)
