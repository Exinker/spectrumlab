from dataclasses import dataclass, field
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from spectrumlab.backgrounds.base_background import BackgroundABC, BackgroundConfigABC
from spectrumlab.peaks.blink_peaks.draft_blinks import DraftBlinksConfig, draft_blinks
from spectrumlab.spectra import Spectrum
from spectrumlab.types import Array, Number


@dataclass
class SavitzkyGolayBackgroundConfig(BackgroundConfigABC):

    width: Number
    degree: int

    width_min: Number = field(default=10)
    width_max: Number = field(default=50)

    @property
    def n_counts_min(self) -> Number:
        return self.width_min


class SavitzkyGolayBackground(BackgroundABC):

    def __init__(self, config: SavitzkyGolayBackgroundConfig):
        super().__init__(config)

        self._mask = None
        self._width = None
        self._background = None

    @property
    def mask(self) -> Array[bool]:
        if self._mask is None:
            raise ValueError  # TODO: add custom exception!

        return self._mask

    @property
    def width(self) -> Array[Number]:
        if self._width is None:
            raise ValueError  # TODO: add custom exception!

        return self._width

    @property
    def background(self) -> Array[Number]:
        if self._background is None:
            raise ValueError  # TODO: add custom exception!

        return self._background

    def fit(self, spectrum: Spectrum, show: bool = False) -> Spectrum:

        self._fit_mask(
            spectrum=spectrum,
            show=show,
        )
        self._fit_width(
            spectrum=spectrum,
        )

        # background
        background = np.zeros(spectrum.shape)
        for t in range(spectrum.n_times):

            # setup filter
            filter = filter_savitzky_golay(
                spectrum.intensity[t, :],
                self.mask[t, :],
                n_counts_min=self.config.n_counts_min,
                width_max=self.config.width_max,
            )

            # calculate background
            for n in range(spectrum.n_numbers):
                background[t, n] = filter(n, width=self.width[t, n], degree=self.config.degree)

        return spectrum.__class__(
            intensity=background,
            wavelength=spectrum.wavelength,
            number=spectrum.number,
            clipped=spectrum.clipped,
            detector=spectrum.detector,
        )

    def _fit_mask(self, spectrum: Spectrum, show: bool) -> None:

        # mask
        mask = np.full(spectrum.shape, False)
        for t in range(spectrum.n_times):
            blinks = draft_blinks(
                spectrum=spectrum[t],
                config=DraftBlinksConfig.model_construct(
                    n_counts_min=1,
                    n_counts_max=500,
                    slope_max=5,
                ),
            )

            for blink in blinks:
                mask[t, blink.number] = True

        # mask filtration (by time)
        for n in range(spectrum.n_numbers):
            for t in range(1, spectrum.n_times-1):
                if mask[t-1, n] and mask[t+1, n]:
                    mask[t, n] = True

            # remove alone blinks
            for t in range(1, spectrum.n_times-1):
                if ~mask[t-1, n] and ~mask[t+1, n]:
                    mask[t, n] = False

        if show:
            fig, ax = plt.subplots(figsize=(18, 4), tight_layout=True)

            plt.imshow(
                mask,
                origin='lower',
                cmap='gray', vmin=0, vmax=1,
            )

        #
        self._mask = mask

    def _fit_width(self, spectrum: Spectrum) -> None:
        width_min = self.config.width_min
        width_max = self.config.width_max

        width = np.full(self.mask.shape, width_min)
        for t in range(spectrum.n_times):
            for n in range(spectrum.n_numbers):
                if self.mask[t, n]:

                    # left bound
                    lb = n
                    while (lb >= n - width_max//2) and (lb >= 0):
                        if not self.mask[t, lb]:
                            break
                        lb -= 1

                    # right bound
                    rb = n
                    while (rb < n + width_max//2) and (rb < spectrum.n_numbers):
                        if not self.mask[t, rb]:
                            break
                        rb += 1

                    # width
                    width[t, n] += (rb - lb) - 1

        self._width = width


def filter_savitzky_golay(
    intensity: Array[float],
    mask: Array[bool],
    n_counts_min: int,
    width_max: Number,
) -> Callable[[Number, float, int], float]:
    """Filter `intensity` by Savitzky-Gloay algorithm."""
    n_numbers = len(intensity)

    def inner(n: Number, width: int, degree: int) -> float:
        if width > width_max:
            return np.nan

        # index
        span = width
        while span <= width_max:
            index = np.arange(n - span//2, n + span//2, dtype=int)
            index = index[(index >= 0) & (index < n_numbers)]
            index = index[~mask[index]]

            if index.size >= n_counts_min:
                break

            span *= 2

        # value
        if index.size >= n_counts_min:
            p = np.polyfit(index, intensity[index], degree)
            return np.polyval(p, n)

        return np.nan

    return inner


def approximate_savitzky_golay(
    intensity: Array[float],
    mask: Array[bool],
    config: SavitzkyGolayBackgroundConfig,
) -> Array[float]:
    """Approximate 'intensity' by Savitzky-Gloay filtration."""
    filter = filter_savitzky_golay(
        intensity,
        mask,
        n_counts_min=config.n_counts_min,
        width_max=config.width_max,
    )

    return np.array([
        filter(n, width=config.width, degree=config.degree)
        for n, _ in enumerate(intensity)
    ])
