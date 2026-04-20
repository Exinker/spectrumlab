import numpy as np
from pydantic import Field

from spectrumlab.backgrounds.base_background import (
    BackgroundConfigABC,
    BackgroundModelABC,
)
from spectrumlab.backgrounds.asymmetric_least_squares_background.utils import estimate_background
from spectrumlab.peaks.blink_peaks import DraftBlinksConfig, draft_blinks
from spectrumlab.spectra import EmittedSpectrum, Spectrum
from spectrumlab.types import Array, R


class AsymmetricLeastSquaresBackgroundConfig(BackgroundConfigABC):

    lam: float = Field(1e+3)
    p: float = Field(1e-4)
    n_iters: int = Field(10)


class AsymmetricLeastSquaresBackgroundModel(BackgroundModelABC):

    def __init__(self, config: AsymmetricLeastSquaresBackgroundConfig):
        super().__init__(config=config)

        self._mask = None
        self._background = None

    @property
    def config(self) -> AsymmetricLeastSquaresBackgroundConfig:
        return self._config

    @property
    def background(self) -> Array[R]:
        if self._background is None:
            raise ValueError  # TODO: add custom exception!

        return self._background

    def fit(
        self,
        spectrum: Spectrum,
    ) -> Spectrum:

        # background
        if spectrum.n_times > 1:
            raise NotImplementedError

        else:
            is_outlier = np.full(spectrum.shape, False)

            background_hat = np.full(spectrum.shape, 0)
            background_hats = []
            background_hats.append(background_hat)
            for _ in range(2):
                mask = build_mask(
                    spectrum=spectrum.__class__(
                        intensity=spectrum.intensity - background_hats[-1],
                        deviation=spectrum.deviation,
                    ),
                    config=DraftBlinksConfig(
                        n_counts_min=1,
                        except_clipped_peak=False,
                        except_sloped_peak=False,
                        except_edges=False,
                        noise_level=4,
                    ),
                )
                mask[is_outlier] = False

                background_hat = estimate_background(
                    intensity=spectrum.intensity,
                    mask=mask,
                    config=AsymmetricLeastSquaresBackgroundConfig(
                        lam=1e+3,
                        p=1e-8,
                    ),
                )
                is_outlier[mask] = spectrum.intensity[mask] - background_hat[mask] > 5 * spectrum.deviation[mask]

                background_hats.append(background_hat)

        return spectrum.__class__(
            intensity=background_hats[-1],
            wavelength=spectrum.wavelength,
            number=spectrum.number,
            clipped=spectrum.clipped,
            detector=spectrum.detector,
        )


def build_mask(
    spectrum: Spectrum,
    config: DraftBlinksConfig | None = None,
) -> Array[bool]:
    config = config or DraftBlinksConfig(
        n_counts_min=1,
        except_clipped_peak=False,
        except_sloped_peak=False,
        except_edges=False,
        noise_level=4,
    )

    peaks = draft_blinks(
        spectrum=spectrum,
        config=config,
    )

    mask = np.full(spectrum.shape, True)
    for peak in peaks:
        mask[peak.number] = False

    return mask
