from dataclasses import dataclass
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt

from spectrumlab.picture.config import COLOR
from spectrumlab.emulation.detector import Detector
from spectrumlab.emulation.emulation import EmittedSpectrumEmulation, EmittedSpectrumEmulationConfig, emulate_emitted_spectrum
from spectrumlab.emulation.noise import EmittedSpectrumNoise
from spectrumlab.emulation.spectrum import EmittedSpectrum, AbsorbedSpectrum, HighDynamicRangeEmittedSpectrum
from spectrumlab.typing import Array, MilliSecond


# --------        HDR emission emulation        --------
@dataclass(frozen=True, slots=True)
class _HighDynamicRangeMode:
    total: MilliSecond  # total exposure time
    n_frames: tuple[int, ...]  # tuple of n_frames of the each exposure (tau)
    method: Literal['naive', 'weighted'] = 'weighted'
    base: int = 10  # base of the each exposure

    def __post_init__(self):
        assert self._validate(), f'{self} is not valid!'

    def _validate(self, tol=1e-9) -> bool:
        """Validate mode to equal total exposure time and expected."""
        total = sum([n_frames * tau for n_frames, tau in self.items()])

        return abs(total - self.total) <= tol

    def items(self) -> tuple[int, MilliSecond]:
        """Generate tuples of n_frames and tau."""

        for degree, n_frames in enumerate(self.n_frames):
            if n_frames > 0:
                tau = self.base ** (-degree)

                yield n_frames, tau


@dataclass(frozen=True, slots=True)
class HighDynamicRangeMode:
    total: MilliSecond  # total exposure time
    n_frames: tuple[int, ...]  # tuple of n_frames of the each exposure
    tau: tuple[MilliSecond, ...]  # tuple of exposures
    method: Literal['naive', 'weighted'] = 'weighted'

    # --------        handlers        --------
    def items(self) -> tuple[int, MilliSecond]:
        """Generate tuples of n_frames and tau."""

        for n_frames, tau in zip(self.n_frames, self.tau):
                yield n_frames, tau

    # --------        private        --------
    def __post_init__(self):
        assert self._validate(), f'{self} is not valid!'

    def _validate(self, tol=1e-9) -> bool:
        """Validate mode to equal total exposure time and expected."""
        total = sum([n_frames * tau for n_frames, tau in self.items()])

        return abs(total - self.total) <= tol


class HighDynamicRangeEmittedSpectrumEmulation(EmittedSpectrumEmulation):
    """High dynamic range (HDR) emitted spectrum emulation."""

    def __init__(self, config: EmittedSpectrumEmulationConfig, mode: HighDynamicRangeMode):
        super().__init__(config=config)

        self.mode = mode

    # --------        handlers        --------
    def run(self, is_noised: bool = True, is_clipped: bool = True, show: bool = False, random_state: int | None = None) -> HighDynamicRangeEmittedSpectrum:
        """Run emulation."""
        config = self.config
        detector = config.detector

        # set random state
        if random_state is not None:
            np.random.seed(random_state)

        # init spectrum
        spectrum = emulate_hdr_emitted_spectrum(
            mode=self.mode,
            number=self.number,
            intensity=self.intensity,
            detector=detector,
            is_noised=is_noised,
            is_clipped=is_clipped,
        )

        # show spectrum
        if show:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), tight_layout=True)

            if spectrum.intensity.ndim == 1:
                y2 = spectrum.intensity
            else:
                y2 = spectrum.intensity[0]
            plt.fill_between(
                spectrum.number,
                y1=config.background_level,
                y2=y2,
                step='mid', alpha=0.2, facecolor=COLOR['pink'], edgecolor='k', label=f'paek',
            )
            plt.xlim([spectrum.number.min()-1, spectrum.number.max()+1])

            plt.xlabel(r'number')
            plt.ylabel({
                EmittedSpectrum: r'$I$ [$\%$]',
                AbsorbedSpectrum: r'$A$',
            }.get(type(spectrum)))
            plt.grid(color='grey', linestyle=':')
            plt.show()

        # return spectrum
        return spectrum


# --------        handlers        --------
def emulate_hdr_emitted_spectrum(mode: HighDynamicRangeMode, number: Array, intensity: Array, detector: Detector, is_noised: bool = True, is_clipped: bool = True) -> HighDynamicRangeEmittedSpectrum:

    shorts = {}
    for n_frames, tau in mode.items():
        spe = emulate_emitted_spectrum(
            number=number,
            intensity=tau*intensity,
            noise=EmittedSpectrumNoise(
                detector=detector,
                n_frames=n_frames,
            ),
            detector=detector,
            is_noised=is_noised,
            is_clipped=is_clipped,
        )
        shorts[tau] = spe

    # 
    spectrum = HighDynamicRangeEmittedSpectrum(
        number=number,
        shorts=shorts,
        method=mode.method,
        detector=detector,
    )

    # return spectrum
    return spectrum
