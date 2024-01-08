from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from spectrumlab.alias import Array, Number
from spectrumlab.emulation.aperture import Aperture
from spectrumlab.emulation.apparatus import Apparatus
from spectrumlab.emulation.detector.linear_array_detector import Detector
from spectrumlab.emulation.emulation import emulate_emitted_spectrum
from spectrumlab.emulation.noise import Noise, EmittedSpectrumNoise
from spectrumlab.emulation.spectrum import EmittedSpectrum
from spectrumlab.line import Line


@dataclass(frozen=True)
class BaseExperimentConfig(ABC):
    n_numbers: int
    n_frames: int
    n_iters: int

    detector: Detector
    apparatus: Apparatus
    aperture: Aperture


class BaseExperiment(ABC):

    def __init__(self, config: BaseExperimentConfig):
        self.config = config

        self._lines = None
        self._number = None
        self._intensity = None
        self._background = None

    @property
    def noise(self) -> Noise:
        return EmittedSpectrumNoise(
            detector=self.config.detector,
            n_frames=self.config.n_frames,
        )

    @property
    def lines(self) -> tuple[Line]:
        if self._lines is None:
            raise Exception  # add 

        return self._lines

    @property
    def intensity(self) -> Array[float]:
        if self._intensity is None:
            raise Exception  # add 

        return self._intensity

    @property
    def number(self) -> Array[Number]:
        if self._number is None:
            raise Exception  # add 

        return self._number

    @property
    def intensity(self) -> Array[float]:
        if self._intensity is None:
            raise Exception  # add 

        return self._intensity

    @property
    def background(self) -> Array[float]:
        if self._background is None:
            raise Exception  # add 

        return self._background

    # --------        handlers        --------
    @abstractmethod
    def setup(self, seed: int | None = None, verbose: bool = False, show: bool = False) -> 'Experiment':
        pass

    def run(self, is_noised: bool = True, is_clipped: bool = True) -> EmittedSpectrum:
        config = self.config

        # spectrum
        spectrum = emulate_emitted_spectrum(
            number=self.number,
            intensity=self.intensity,
            noise=self.noise,
            detector=config.detector,
            is_noised=is_noised,
            is_clipped=is_clipped,
        )

        return spectrum


def distance(xi: float, xi_hat: float, is_relative: bool = False) -> float:
    """Calculate a distance (relative, in optionally) between `xi` and `xi_hat`."""
    if is_relative:
        return np.abs((xi_hat - xi) / xi)
    return np.abs(xi_hat - xi)
