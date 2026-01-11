from typing import TYPE_CHECKING

import numpy as np

from spectrumlab.spectra import Spectrum
from spectrumlab.types import Array, R, T

if TYPE_CHECKING:
    from spectrumlab.peaks.blink_peaks import BlinkPeak


class ChunkFactory:

    def __init__(self, spectrum: Spectrum):
        self.spectrum = spectrum

    def create_from_peak(self, peak: 'BlinkPeak', threshold: float) -> 'Chunk':
        lb, ub = peak.minima

        is_clipped = self.spectrum.clipped[lb:ub]
        is_snr_low = np.abs(self.spectrum.intensity[lb:ub]) / self.spectrum.deviation[lb:ub] < threshold
        mask = ~is_clipped & ~is_snr_low

        x = self.spectrum.number[lb:ub][mask]
        y = self.spectrum.intensity[lb:ub][mask]

        return Chunk(x, y)

    def create_from_frame(self, t: int, threshold: float) -> 'Chunk':
        is_clipped = self.spectrum.clipped[t]
        is_snr_low = np.abs(self.spectrum.intensity[t]) / self.spectrum.deviation[t] < threshold
        mask = ~is_clipped & ~is_snr_low

        x = self.spectrum.number[mask]
        y = self.spectrum.intensity[mask] if self.spectrum.n_times == 1 else self.spectrum.intensity[t, mask]

        return Chunk(x, y)


class Chunk:

    factory = ChunkFactory

    def __init__(self, x: Array[T], y: Array[R]):

        self.x = x
        self.y = y
