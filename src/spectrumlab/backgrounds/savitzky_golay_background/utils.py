import numpy as np

from spectrumlab.backgrounds.savitzky_golay_background import (
    SavitzkyGolayBackgroundConfig,
    filter_savitzky_golay,
)
from spectrumlab.types import Array, R


def estimate_background(
    intensity: Array[R],
    mask: Array[bool],
    config: SavitzkyGolayBackgroundConfig,
) -> Array[R]:
    """Estimate background by Savitzky-Gloay approximation."""

    filter = filter_savitzky_golay(
        intensity,
        mask,
        n_counts_min=config.n_counts_min,
        window_max=config.window_max,
    )

    return np.array([
        filter(n, window=config.window, degree=config.degree)
        for n, _ in enumerate(intensity)
    ])
