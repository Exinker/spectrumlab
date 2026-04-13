from typing import Callable, TYPE_CHECKING

import numpy as np

from spectrumlab.types import Array, Number, R

if TYPE_CHECKING:
    from spectrumlab.backgrounds.savitzky_golay_background.model import SavitzkyGolayBackgroundConfig


def savitzky_golay_kernel(
    intensity: Array[R],
    mask: Array[bool],
    n_counts_min: int,
    window_max: int,
) -> Callable[[Number, int, int], R]:
    n_numbers = len(intensity)

    def inner(n: Number, window: int, degree: int) -> float:
        if window > window_max:
            return np.nan

        # index
        span = window
        while span <= window_max:
            index = np.arange(max(n - span//2, 0), min(n + span//2 + 1, n_numbers), dtype=int)
            index = index[~mask[index]]

            if index.size >= n_counts_min:
                break

            span *= 2

        # value
        if index.size >= n_counts_min:
            weight = np.exp(-((index - n)**2) / (2 * window**2))

            p = np.polyfit(index, intensity[index], degree, w=weight)
            return np.polyval(p, n)

        return np.nan

    return inner


def estimate_background(
    intensity: Array[R],
    mask: Array[bool],
    config: 'SavitzkyGolayBackgroundConfig',
) -> Array[R]:
    """Estimate background by Savitzky-Gloay approximation."""

    kernel = savitzky_golay_kernel(
        intensity,
        mask,
        n_counts_min=config.n_counts_min,
        window_max=config.window_max,
    )

    return np.array([
        kernel(n, window=config.window, degree=config.degree)
        for n, _ in enumerate(intensity)
    ])
