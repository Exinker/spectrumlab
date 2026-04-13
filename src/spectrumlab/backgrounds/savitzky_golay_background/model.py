from typing import Self

import matplotlib.pyplot as plt
import numpy as np
from pydantic import Field, model_validator
from scipy.signal import convolve2d

from spectrumlab.backgrounds.base_background import (
    BackgroundConfigABC,
    BackgroundModelABC,
)
from spectrumlab.backgrounds.savitzky_golay_background.utils import savitzky_golay_kernel
from spectrumlab.peaks.blink_peaks.draft_blinks import DraftBlinksConfig, draft_blinks
from spectrumlab.spectra import Spectrum
from spectrumlab.types import Array, Number, R


class SavitzkyGolayBackgroundConfig(BackgroundConfigABC):

    window: Number
    degree: int

    window_max: Number = Field(default=50)
    n_counts_min: Number | None = Field(None)

    @model_validator(mode='after')
    def check_model(self) -> Self:

        if self.n_counts_min is None:
            self.n_counts_min = self.window

        if self.window > self.window_max:
            raise ValueError('Window ({window}) cannot exceed window_max ({window_max})'.format(
                window=self.window,
                window_max=self.window_max,
            ))

        if self.n_counts_min > self.window:
            raise ValueError('Minimum counts ({n_counts_min}) have to be less than window ({window_max})'.format(
                n_counts_min=self.n_counts_min,
                window_max=self.window_max,
            ))

        if self.degree > self.n_counts_min - 2:
            raise ValueError('Degree ({degree}) have to be less than minimum counts - 2'.format(
                degree=self.degree,
            ))


class SavitzkyGolayBackgroundModel(BackgroundModelABC):

    def __init__(self, config: SavitzkyGolayBackgroundConfig):
        super().__init__(config=config)

        self._mask = None
        self._background = None

    @property
    def config(self) -> SavitzkyGolayBackgroundConfig:
        return self._config

    @property
    def background(self) -> Array[R]:
        if self._background is None:
            raise ValueError  # TODO: add custom exception!

        return self._background

    def fit(
        self,
        spectrum: Spectrum,
        mask: Array[bool],
    ) -> Spectrum:

        # background
        background = np.zeros(spectrum.shape)
        for t in range(spectrum.n_times):

            kernel = savitzky_golay_kernel(
                spectrum.intensity[t, :],
                mask[t, :],
                n_counts_min=self.config.n_counts_min,
                window_max=self.config.window_max,
            )

            for n in range(spectrum.n_numbers):
                background[t, n] = kernel(n, window=self.config.window, degree=self.config.degree)

        return spectrum.__class__(
            intensity=background,
            wavelength=spectrum.wavelength,
            number=spectrum.number,
            clipped=spectrum.clipped,
            detector=spectrum.detector,
        )


def build_mask(
    spectrum: Spectrum,
    config: DraftBlinksConfig | None = None,
    threshold: float = 0.4,
    show: bool = False,
) -> Array[bool]:
    config = config or DraftBlinksConfig(
        n_counts_min=3,
        n_counts_max=500,
        except_edges=True,
        noise_level=2,
    )

    # build mask
    mask = np.full(spectrum.shape, False)
    for t in range(spectrum.n_times):
        blinks = draft_blinks(
            spectrum=spectrum[t],
            config=config,
        )

        for blink in blinks:
            mask[t, blink.number] = True

    mask = _filtrate_mask_by_time(
        __mask=mask,
        threshold=threshold,
    )

    if show:
        fig, ax = plt.subplots(figsize=(18, 8), tight_layout=True)

        plt.imshow(
            mask.T,
            origin='lower',
            cmap='gray', vmin=0, vmax=1,
        )

        plt.show()

    return mask


def _filtrate_mask_by_time(
    __mask: Array[bool],
    threshold: float,
    n_times: int = 7,
) -> Array[bool]:

    time = np.arange(n_times)
    kernel = np.exp(-time / n_times).reshape(-1, 1)
    kernel /= kernel.sum()

    smoothed_mask = convolve2d(__mask.astype(float), kernel, mode='same')
    mask = smoothed_mask >= threshold

    cleaned_mask = convolve2d(mask.astype(float), np.array([1, 0, 1]).reshape(-1, 1), mode='same') > 0
    mask = mask & cleaned_mask

    return mask
