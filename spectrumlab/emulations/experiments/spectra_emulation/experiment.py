import abc
from dataclasses import dataclass

from spectrumlab.emulations.apertures import Aperture  # noqa: I100
from spectrumlab.emulations.apparatus import Apparatus
from spectrumlab.emulations.detectors import Detector
from spectrumlab.emulations.emulations import emulate_emitted_spectrum
from spectrumlab.emulations.noises import EmittedSpectrumNoise, Noise
from spectrumlab.emulations.spectrum import EmittedSpectrum
from spectrumlab.types import Array, Number


@dataclass(frozen=True)
class AbstractExperimentConfig(abc.ABC):
    n_numbers: int
    n_frames: int

    detector: Detector
    apparatus: Apparatus
    aperture: Aperture

    exposure: float | Array[float]
    position: Number | Array[Number]
    intensity: float | Array[float]


class AbstractExperiment(abc.ABC):

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
    @abc.abstractmethod
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
