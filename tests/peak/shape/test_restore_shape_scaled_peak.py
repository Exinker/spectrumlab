from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import pytest

from spectrumlab.alias import Array, Number
from spectrumlab.emulation.aperture import Aperture, RectangularApertureShape
from spectrumlab.emulation.apparatus import Apparatus, VoigtApparatusShape
from spectrumlab.emulation.detector.linear_array_detector import Detector
from spectrumlab.emulation.emulation import convolve
from spectrumlab.peak.shape import VoightPeakShape, Grid, restore_shape_from_grid

from core import BaseExperimentConfig, BaseExperiment, distance


IS_NOISED = False
POSITION = 0.0
N_ITERS = 51
THRESHOLD = 100

DETECTOR = Detector.BLPP2000
SHAPE = VoigtApparatusShape(
    width=28,
    asymmetry=+0.1,
    ratio=0.1,
)


class ExperimentConfig(BaseExperimentConfig):

    @property
    def exposure(self) -> Array[float]:
        return np.array([2**x for x in range(self.n_iters)])

    @property
    def position(self) -> Number:
        return self.n_numbers//2 + POSITION


class Experiment(BaseExperiment):

    def __init__(self, config: ExperimentConfig):
        super().__init__(config=config)

    # --------        handlers        --------
    def setup(self, seed: int | None = None, verbose: bool = False, show: bool = False) -> 'Experiment':
        config = self.config
        detector = self.config.detector

        step = detector.config.width

        # setup seed
        if seed:
            np.random.seed(seed)

        # setup intensity
        rx = 100
        dx = .01
        x = np.linspace(-rx, +rx, 2*int(rx/dx) + 1)
        f = convolve(x, apparatus=config.apparatus, aperture=config.aperture, step=step)

        self._number = np.arange(config.n_numbers)
        self._background = 0

        intensity = np.zeros((config.n_iters, config.n_numbers))
        for i, exposure in enumerate(config.exposure):
            intensity[i] = exposure * f(self.number - config.position)
        self._intensity = intensity

        # show
        if show:
            fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

            x, y = x, f(x/step)
            plt.plot(
                x/step, y,
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
        verbose=False,
        show=False,
    )

    return experiment


@pytest.fixture(scope='module')
def shape_hat(experiment: Experiment) -> VoightPeakShape:
    config = experiment.config

    # spectrum
    spectrum = experiment.run(is_noised=IS_NOISED)

    # grid
    grid = Grid.from_frames(
        spectrum=spectrum,
        offset=np.full((config.n_iters, ), config.n_numbers//2),
        scale=config.exposure,
        background=np.full((config.n_iters, ), 0),
        threshold=THRESHOLD,
    )

    # shape_hat
    shape_hat = restore_shape_from_grid(
        grid=grid,
        show=False,
    )

    return shape_hat


# --------        tests        --------
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
    )

    # experiment
    experiment = Experiment(
        config=config,
    )
    experiment = experiment.setup(
        verbose=False,
        show=False,
    )

    # spectrum
    spectrum = experiment.run(is_noised=IS_NOISED)

    # restore shape
    grid = Grid.from_frames(
        spectrum=spectrum,
        offset=np.full((config.n_iters, ), config.position),
        scale=config.exposure,
        background=np.full((config.n_iters, ), 0),
        threshold=THRESHOLD,
    )
    shape_hat = restore_shape_from_grid(
        grid=grid,
        show=True,
    )
