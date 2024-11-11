import matplotlib.pyplot as plt
import numpy as np

from spectrumlab.emulations.emulators import convolve
from spectrumlab.emulations.experiments.spectra_emulation.experiment import AbstractExperiment, AbstractExperimentConfig


class ScaledExperimentConfig(AbstractExperimentConfig):

    @property
    def n_iters(self) -> int:
        return len(self.exposure)


class ScaledExperiment(AbstractExperiment):

    def __init__(self, config: ScaledExperimentConfig):
        super().__init__(config=config)

    def setup(
        self,
        seed: int | None = None,
        verbose: bool = False,
        show: bool = False,
    ) -> 'ScaledExperiment':
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

        intensity = np.zeros((config.n_iters, config.n_numbers))
        for i, exposure in enumerate(config.exposure):
            intensity[i] = exposure * config.intensity * f(self.number - config.position)
        self._intensity = intensity

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
