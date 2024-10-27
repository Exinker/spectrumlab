import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from spectrumlab.emulations.emulations import convolve
from spectrumlab.emulations.experiments.spectra_emulation.experiment import AbstractExperiment, AbstractExperimentConfig
from spectrumlab.line import Line
from spectrumlab.picture.color import COLOR


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

    def setup(
        self,
        seed: int | None = None,
        verbose: bool = False,
        show: bool = False,
    ) -> 'RandomPeakExperiment':
        config = self.config
        detector = self.config.detector

        if seed:
            np.random.seed(seed)

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
