from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from spectrumlab.emulations.detector import Detector
from spectrumlab.emulations.emulators import (
    EmittedSpectrumEmulationConfig,
    EmittedSpectrumEmulator,
    emitted_spectrum_factory,
)
from spectrumlab.emulations.noise import EmittedSpectrumNoise
from spectrumlab.emulations.spectrum import AbsorbedSpectrum, EmittedSpectrum, HighDynamicRangeEmittedSpectrum
from spectrumlab.picture.color import COLOR
from spectrumlab.types import Array, MilliSecond


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


def mode_factory(
    total: MilliSecond,
    n_frames: tuple[int, ...],
    tau: tuple[MilliSecond, ...],
    method: Literal['naive', 'weighted'] = 'weighted',
    tol=1e-9,
) -> 'HighDynamicRangeMode':

    expected = sum([n * t for n, t in zip(n_frames, tau)])
    assert abs(total - expected) <= tol, 'Total time is not equal to expected!'

    return HighDynamicRangeMode(
        total=total,
        n_frames=n_frames,
        tau=tau,
        method=method,
    )


class HighDynamicRangeMode:

    create = mode_factory

    def __init__(
        self,
        total: MilliSecond,  # total exposure time
        n_frames: tuple[int, ...],  # tuple of n_frames of the each exposure
        tau: tuple[MilliSecond, ...],  # tuple of exposures
        method: Literal['naive', 'weighted'] = 'weighted',
    ) -> None:
        self.total = total
        self.n_frames = n_frames
        self.tau = tau
        self.method = method

    def items(self) -> tuple[int, MilliSecond]:
        """Generate tuples of n_frames and tau."""

        for n_frames, tau in zip(self.n_frames, self.tau):
            yield n_frames, tau


class HighDynamicRangeEmittedSpectrumEmulation(EmittedSpectrumEmulator):
    """High dynamic range (HDR) emitted spectrum emulation."""

    def __init__(
        self,
        config: EmittedSpectrumEmulationConfig,
        mode: HighDynamicRangeMode,
    ) -> None:
        super().__init__(config=config)

        self.mode = mode

    def run(
        self,
        is_noised: bool = True,
        is_clipped: bool = True,
        show: bool = False,
        random_state: int | None = None,
    ) -> HighDynamicRangeEmittedSpectrum:
        """Run emulation."""
        config = self.config
        detector = config.detector

        if random_state is not None:
            np.random.seed(random_state)

        spectrum = hdr_emitted_spectrum_factory(
            mode=self.mode,
            number=self.number,
            intensity=self.intensity,
            detector=detector,
            is_noised=is_noised,
            is_clipped=is_clipped,
        )

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
                step='mid', alpha=0.2, facecolor=COLOR['pink'], edgecolor='k', label='paek',
            )
            plt.xlim([spectrum.number.min()-1, spectrum.number.max()+1])

            plt.xlabel(r'number')
            plt.ylabel({
                EmittedSpectrum: r'$I$ [$\%$]',
                AbsorbedSpectrum: r'$A$',
            }.get(type(spectrum)))
            plt.grid(color='grey', linestyle=':')
            plt.show()

        return spectrum


def hdr_emitted_spectrum_factory(
    mode: HighDynamicRangeMode,
    number: Array,
    intensity: Array,
    detector: Detector,
    is_noised: bool = True,
    is_clipped: bool = True,
) -> HighDynamicRangeEmittedSpectrum:

    shorts = {}
    for n_frames, tau in mode.items():
        spe = emitted_spectrum_factory(
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

    spectrum = HighDynamicRangeEmittedSpectrum(
        number=number,
        shorts=shorts,
        method=mode.method,
        detector=detector,
    )

    return spectrum
