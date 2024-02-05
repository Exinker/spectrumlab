from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt

from spectrumlab.alias import Array
from spectrumlab.emulation.noise import Noise
from spectrumlab.spectrum import Spectrum
from spectrumlab.background.base_background import BaseBackground, BaseBackgroundConfig
from spectrumlab.peak.blink_peak import draft_blinks, DraftBlinkPeakConfig


@dataclass
class SavitzkyGolayBackgroundConfig(BaseBackgroundConfig):
    width: int
    degree: int

    n_counts_min: int = field(default=10)


class SavitzkyGolayBackground(BaseBackground):

    def __init__(self, config: SavitzkyGolayBackgroundConfig):
        super().__init__(config)

    # --------        handlers        --------
    def fit(self, spectrum: Spectrum, noise: Noise, show: bool = False) -> Array:

        # mask blinks
        mask = np.full(spectrum.shape, False)
        width = np.full(spectrum.shape, 0)
        for t in range(spectrum.n_times):
            blinks = draft_blinks(
                spectrum=spectrum[t],
                noise=noise,
                config=DraftBlinkPeakConfig(
                    n_counts_min=1,
                    n_counts_max=500,
                    slope_max=5,
                ),
            )

            for blink in blinks:
                mask[t, blink.number] = True
                width[t, blink.number] = 2*blink.n_numbers

        for n in range(spectrum.n_numbers):
            for t in range(1, spectrum.n_times-1):
                if mask[t-1,n] and mask[t+1,n]:
                    mask[t,n] = True

        if show:
            fig, ax = plt.subplots(figsize=(18,4), tight_layout=True)

            plt.imshow(
                mask,
                origin='lower',
                cmap='gray', vmin=0, vmax=1,
            )
        
        #

        background = np.zeros(spectrum.shape)
        for t in range(spectrum.n_times):
            filter = filter_savitzky_golay(spectrum.intensity[t,:], mask[t,:])

            for n in range(spectrum.n_numbers):
                background[t,n] = filter(
                    n,
                    width=width[t,n] + self.config.width,
                    degree=self.config.degree,
                )

        #
        return background


# --------        handlers        --------
def filter_savitzky_golay(intensity: Array[float], mask: Array[bool], n_counts_min: int) -> Callable:
    """Savitzky-Gloay filter with mask."""
    n_numbers = len(intensity)

    def inner(n: int, width: int, degree: int) -> float:
        hw = width // 2

        # index
        while True:
            index = np.arange(n-hw, n+hw+1, dtype=int)
            index = index[(index >= 0) & (index < n_numbers)]
            index = index[~mask[index]]

            #
            n_counts = index.size
            if n_counts >= n_counts_min:
                break
            if  hw >= n_numbers:
                break

            hw *= 2

        #
        return np.polyval(
            np.polyfit(index, intensity[index], degree),
            n,
        )

    return inner


def approximate_savitzky_golay(intensity: Array[float], mask: Array[bool], config: SavitzkyGolayBackgroundConfig) -> Array[float]:
    """Approximate y values with Savitzky-Gloay filtration."""
    width = config.width
    degree = config.degree
    filter = filter_savitzky_golay(intensity, mask, n_counts_min=config.n_counts_min)

    return np.array([
        filter(n, width=width, degree=degree)
        for n, _ in enumerate(intensity)
    ])
