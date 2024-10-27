import matplotlib.pyplot as plt
import numpy as np

from spectrumlab.emulations.emulations import convolve
from spectrumlab.emulations.experiments.spectra_emulation.experiment import AbstractExperiment, AbstractExperimentConfig


class ShiftedExperimentConfig(AbstractExperimentConfig):

    @property
    def n_iters(self) -> int:
        return len(self.position)


class ShiftedExperiment(AbstractExperiment):

    def __init__(self, config: ShiftedExperimentConfig):
        super().__init__(config=config)

    def setup(
        self,
        seed: int | None = None,
        verbose: bool = False,
        show: bool = False,
    ) -> 'ShiftedExperiment':
        config = self.config
        detector = self.config.detector

        if seed:
            np.random.seed(seed)

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
