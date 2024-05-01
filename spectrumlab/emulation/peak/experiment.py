from abc import ABC, abstractmethod
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from spectrumlab.emulation.aperture import Aperture
from spectrumlab.emulation.apparatus import Apparatus
from spectrumlab.emulation.detector import Detector
from spectrumlab.emulation.emulation import convolve, emulate_emitted_spectrum
from spectrumlab.emulation.noise import EmittedSpectrumNoise, Noise
from spectrumlab.emulation.spectrum import EmittedSpectrum
from spectrumlab.line import Line
from spectrumlab.picture.color import COLOR
from spectrumlab.types import Array, Number


@dataclass(frozen=True)
class AbstractExperimentConfig(ABC):
    n_numbers: int
    n_frames: int

    detector: Detector
    apparatus: Apparatus
    aperture: Aperture

    exposure: float | Array[float]
    position: Number | Array[Number]
    intensity: float | Array[float]


class AbstractExperiment(ABC):

    def __init__(self, config: AbstractExperimentConfig):
        self.config = config

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
    def intensity(self) -> Array[float]:
        if self._intensity is None:
            raise Exception  # add custom exception!

        return self._intensity

    @property
    def number(self) -> Array[Number]:
        if self._number is None:
            raise Exception  # add custom exception!

        return self._number

    @property
    def background(self) -> Array[float]:
        if self._background is None:
            raise Exception  # add custom exception!

        return self._background

    # --------        handlers        --------
    @abstractmethod
    def setup(self, seed: int | None = None, verbose: bool = False, show: bool = False) -> 'AbstractExperiment':
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


# --------        shifted peak experiment        --------
class ShiftedExperimentConfig(AbstractExperimentConfig):

    @property
    def n_iters(self) -> int:
        return len(self.position)


class ShiftedExperiment(AbstractExperiment):

    def __init__(self, config: ShiftedExperimentConfig):
        super().__init__(config=config)

    # --------        handlers        --------
    def setup(self, seed: int | None = None, verbose: bool = False, show: bool = False) -> 'ShiftedExperiment':
        config = self.config
        detector = self.config.detector

        # setup seed
        if seed:
            np.random.seed(seed)

        # setup intensity
        rx = 100
        dx = 1e-2
        x = np.linspace(-rx, +rx, 2*int(rx/dx) + 1)
        f = convolve(x, apparatus=config.apparatus, aperture=config.aperture, pitch=detector.pitch)

        self._number = np.arange(config.n_numbers)
        self._background = 0

        intensity = np.zeros((config.n_iters, config.n_numbers))
        for i, position in enumerate(config.position):
            intensity[i] = config.exposure * config.intensity * f(self.number - position)
        self._intensity = intensity

        # show
        if show:
            fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

            y = f(x/detector.pitch)
            plt.plot(
                x/detector.pitch, y,
                color='black',
            )

            for i, position in enumerate(config.position):
                x = self.number - position
                y = self.intensity[i] / config.exposure
                plt.plot(
                    x, y,
                    color='red', linestyle='none', marker='s', markersize=3,
                    alpha=1,
                )

            plt.grid(color='grey', linestyle=':')
            plt.show()

        return self


# --------        scaled peak experiment        --------
class ScaledExperimentConfig(AbstractExperimentConfig):

    @property
    def n_iters(self) -> int:
        return len(self.exposure)


class ScaledExperiment(AbstractExperiment):

    def __init__(self, config: ScaledExperimentConfig):
        super().__init__(config=config)

    # --------        handlers        --------
    def setup(self, seed: int | None = None, verbose: bool = False, show: bool = False) -> 'ScaledExperiment':
        config = self.config
        detector = self.config.detector

        # setup seed
        if seed:
            np.random.seed(seed)

        # setup intensity
        rx = 100
        dx = .01
        x = np.linspace(-rx, +rx, 2*int(rx/dx) + 1)
        f = convolve(x, apparatus=config.apparatus, aperture=config.aperture, pitch=detector.pitch)

        self._number = np.arange(config.n_numbers)
        self._background = 0

        intensity = np.zeros((config.n_iters, config.n_numbers))
        for i, exposure in enumerate(config.exposure):
            intensity[i] = exposure * config.intensity * f(self.number - config.position)
        self._intensity = intensity

        # show
        if show:
            fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

            x, y = x, f(x/detector.pitch)
            plt.plot(
                x/detector.pitch, y,
                color='black',
            )

            for i, exposure in enumerate(config.exposure):
                x = self.number - config.position
                y = self.intensity[i] / exposure
                plt.plot(
                    x, y,
                    color='red', linestyle='none', marker='s', markersize=3,
                    alpha=1,
                )

            plt.grid(color='grey', linestyle=':')
            plt.show()

        return self


# --------        random peak experiment        --------
class RandomPeakExperimentConfig(AbstractExperimentConfig):

    @property
    def lines(self) -> tuple[Line]:
        assert len(self.position) == len(self.intensity)

        return tuple([
            Line(symbol='NA', wavelength=position, database_intensity=intensity)
            for position, intensity in zip(self.position, self.intensity)
        ])

    @property
    def n_iters(self) -> int:
        return len(self.lines)


class RandomPeakExperiment(AbstractExperiment):

    def __init__(self, config: RandomPeakExperimentConfig):
        super().__init__(config=config)

    # --------        handlers        --------
    def setup(self, seed: int | None = None, verbose: bool = False, show: bool = False) -> 'RandomPeakExperiment':
        config = self.config
        detector = self.config.detector

        # setup seed
        if seed:
            np.random.seed(seed)

        # setup intensity
        rx = 100
        dx = .01
        x = np.linspace(-rx, +rx, 2*int(rx/dx) + 1)
        f = convolve(x, apparatus=config.apparatus, aperture=config.aperture, pitch=detector.pitch)

        self._number = np.arange(config.n_numbers)
        self._background = 0
        self._intensity = np.sum(np.array([
            config.exposure * line.intensity * f(self.number - line.wavelength)
            for line in tqdm(config.lines, total=config.n_iters, disable=not verbose)
        ]), axis=0)

        # show
        if show:
            fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

            for line in config.lines:
                plt.plot(
                    [line.wavelength, line.wavelength], [0, config.exposure*line.intensity],
                    color=COLOR['blue'], linestyle=':',
                )
            plt.plot(
                self.number, self.intensity,
            )

            plt.grid(color='grey', linestyle=':')
            plt.show()

        return self
