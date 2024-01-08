from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pytest

from spectrumlab.alias import Array, Number
from spectrumlab.emulation.aperture import Aperture, RectangularApertureShape
from spectrumlab.emulation.apparatus import Apparatus, VoigtApparatusShape
from spectrumlab.emulation.detector.linear_array_detector import Detector
from spectrumlab.emulation.emulation import convolve
from spectrumlab.line import Line
from spectrumlab.peak.shape import VoightPeakShape, restore_shape_from_spectrum
from spectrumlab.picture.config import COLOR

from core import BaseExperimentConfig, BaseExperiment, distance


IS_NOISED = False
EXPOSURE = 100
N_ITERS = 50
THRESHOLD = 0

DETECTOR = Detector.BLPP2000
SHAPE = VoigtApparatusShape(
    width=28,
    asymmetry=+0.1,
    ratio=0.1,
)


class ExperimentConfig(BaseExperimentConfig):

    @property
    def exposure(self) -> float:
        return EXPOSURE

    @property
    def position(self) -> Array[Number]:
        return self.n_numbers//2 + np.linspace(-.5, +.5, self.n_iters)


class Experiment(BaseExperiment):

    def __init__(self, config: ExperimentConfig):
        super().__init__(config=config)

    # --------        handlers        --------
    def setup(self, mu: float, sigma: float, seed: int | None = None, verbose: bool = False, show: bool = False) -> 'Experiment':
        config = self.config
        detector = self.config.detector

        step = detector.config.width

        # setup seed
        if seed:
            np.random.seed(seed)

        # sefup lines
        # _position = np.linspace(0, config.n_numbers, config.n_iters)
        # _intensity = np.full(config.n_iters, 1)
        _position = np.random.uniform(0, config.n_numbers, size=(config.n_iters,))
        _intensity = 10**np.random.normal(mu, sigma, size=(config.n_iters,))
        self._lines = tuple([
            Line(id=0, symbol='NA', wavelength=position, database_intensity=intensity)
            for position, intensity in zip(_position, _intensity)
        ])

        # setup intensity
        rx = 100
        dx = .01
        x = np.linspace(-rx, +rx, 2*int(rx/dx) + 1)
        f = convolve(x, apparatus=config.apparatus, aperture=config.aperture, step=step)

        self._number = np.arange(config.n_numbers)
        self._background = 0

        intensity = np.zeros((config.n_numbers,))
        for line in tqdm(self.lines, total=config.n_iters, disable=not verbose):
            intensity += line.intensity * f(self.number - line.position)
        intensity *= config.exposure
        self._intensity = intensity


        # show
        if show:
            fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

            for line in self.lines:
                plt.plot(
                    [line.position, line.position], [0, config.exposure*line.intensity],
                    color=COLOR['blue'], linestyle=':',
                )
            plt.plot(
                self.number, self.intensity,
            )

            plt.grid(color='grey', linestyle=':')
            plt.show()

        return self


# --------        fixtures        --------
@pytest.fixture(scope='module')
def detector() -> Detector:
    return DETECTOR


@pytest.fixture(scope='module')
def shape() -> VoigtApparatusShape:
    return SHAPE


@pytest.fixture(scope='module')
def experiment(detector: Detector, shape: VoightPeakShape) -> Experiment:

    experiment = Experiment(
        config=ExperimentConfig(
            n_numbers=20,
            n_frames=1,
            n_iters=N_ITERS,

            detector=detector,
            apparatus=Apparatus(
                detector=detector,
                shape=shape,
            ),
            aperture=Aperture(
                detector=detector,
                shape=RectangularApertureShape(),
            ),
        ),
    )
    experiment = experiment.setup(
        mu=-1,
        sigma=.5,
        verbose=False,
        show=False,
    )

    return experiment


@pytest.fixture(scope='module')
def shape_hat(experiment: Experiment) -> VoightPeakShape:

    # spectrum
    spectrum = experiment.run(is_noised=IS_NOISED)

    # shape_hat
    shape_hat = restore_shape_from_spectrum(
        spectrum=spectrum,
        noise=experiment.noise,
        verbose=True,
        show=True,
    )

    return shape_hat


def test_params_error(detector: Detector, shape: VoigtApparatusShape, shape_hat: VoightPeakShape):
    tolerance = 1e-3  # 0.1 [%]
    step = detector.config.width

    assert distance(
        xi=shape.width,
        xi_hat=shape_hat.width*step,
        is_relative=True,
    ) < tolerance
    assert distance(
        xi=shape.asymmetry,
        xi_hat=shape_hat.asymmetry,
    ) < tolerance
    assert distance(
        xi=shape.ratio,
        xi_hat=shape_hat.ratio,
    ) < tolerance


def test_shape_error(detector: Detector, shape: VoigtApparatusShape, shape_hat: VoightPeakShape):
    tolerance = 1e-6
    step = detector.config.width

    f = partial(shape, x0=0, step=step)
    f_hat = partial(VoigtApparatusShape(
        width=shape_hat.width*step,
        asymmetry=shape.asymmetry,
        ratio=shape.ratio,
    ), x0=0, step=step)

    rx = 100
    dx = .01
    x = np.linspace(-rx, +rx, 2*int(rx/dx) + 1)

    y = f(x)
    y_hat = f_hat(x)

    #
    assert np.all(np.abs(y_hat - y) < tolerance)


if __name__ == '__main__':
    detector = DETECTOR
    shape = SHAPE
    config = ExperimentConfig(
        n_numbers=2048,
        n_frames=1,
        n_iters=N_ITERS,

        detector=detector,
        apparatus=Apparatus(
            detector=detector,
            shape=shape,
        ),
        aperture=Aperture(
            detector=detector,
            shape=RectangularApertureShape(),
        ),
    )

    # experiment
    experiment = Experiment(
        config=config,
    )
    experiment = experiment.setup(
        mu=-1,
        sigma=.5,
        verbose=False,
    )

    # spectrum
    spectrum = experiment.run(is_noised=IS_NOISED)

    # restore shape
    shape_hat = restore_shape_from_spectrum(
        spectrum=spectrum,
        noise=experiment.noise,
        verbose=True,
        show=True,
    )
