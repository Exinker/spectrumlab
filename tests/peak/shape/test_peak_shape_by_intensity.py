from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import pytest
from tqdm import tqdm
from scipy import interpolate, signal

from spectrumlab.alias import Array, Number
from spectrumlab.emulation.aperture import Aperture, RectangularApertureShape
from spectrumlab.emulation.apparatus import Apparatus, VoigtApparatusShape
from spectrumlab.emulation.detector.linear_array_detector import Detector
from spectrumlab.peak.shape import VoightPeakShape
from spectrumlab.peak.shape.grid import Grid
from spectrumlab.peak.shape.peak_shape import VoightPeakShape
from spectrumlab.utils import mse

from experiment import BaseExperimentConfig, BaseExperiment


DETECTOR = Detector.BLPP2000
SHAPE = VoigtApparatusShape(
    width=25,
    asymmetry=+0.1,
    ratio=0.1,
)


class ExperimentConfig(BaseExperimentConfig):

    @property
    def exposure(self) -> Array[float]:
        return np.array([2**x for x in range(self.n_iters)])

    @property
    def position(self) -> Number:
        return self.n_numbers//2


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
        f = interpolate.interp1d(
            x,
            signal.convolve(config.apparatus(x, 0), config.aperture(x, 0), mode='same') * (x[-1] - x[0])/len(x),
            kind='linear',
            bounds_error=False,
            fill_value=0,
        )

        self._number = np.arange(config.n_numbers)
        self._background = 0

        intensity = np.zeros((config.n_iters, config.n_numbers))
        for i, exposure in enumerate(config.exposure):
            intensity[i] = exposure * f((self.number - config.position)*step)
        self._intensity = intensity

        # show
        if show:
            fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

            x, y = x, f(x)
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
            n_iters=21,

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
        show=True,
    )

    return experiment


@pytest.fixture(scope='module')
def shape_hat(experiment: Experiment) -> VoightPeakShape:
    config = experiment.config

    # spectrum
    spectrum = experiment.run(is_noised=True)

    # grid
    grid = Grid.from_frames(
        spectrum=spectrum,
        offset=np.full((config.n_iters, ), config.n_numbers//2),
        scale=config.exposure,
        background=np.full((config.n_iters, ), 0),
    )

    # shape_hat
    shape_hat = VoightPeakShape.from_grid(
        grid=grid,
        show=True,
    )

    return shape_hat


def test_params_error(detector: Detector, shape: VoigtApparatusShape, shape_hat: VoightPeakShape):
    tolerance = 1e-2  # 1 [%]
    step = detector.config.width

    def distance(xi: float, xi_hat: float, is_relative: bool = False) -> float:
        """Calculate a distance (relative, in optionally) between `xi` and `xi_hat`."""
        if is_relative:
            return np.abs((xi_hat - xi) / xi)
        return np.abs(xi_hat - xi)

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
    from functools import partial

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
    assert mse(y, y_hat) < tolerance


if __name__ == '__main__':

    detector = DETECTOR
    shape = SHAPE

    # experiment
    config = ExperimentConfig(
        n_numbers=20,
        n_frames=1,
        n_iters=21,

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

    experiment = Experiment(
        config=config,
    )
    experiment = experiment.setup(
        verbose=False,
        # show=True,
    )

    # spectrum
    spectrum = experiment.run(is_noised=True)

    # optimaze threshold
    step = detector.config.width

    n_thresholds = 101
    thresholds = np.logspace(-1, 2, n_thresholds)
    error = []
    for threshold in tqdm(thresholds):
        grid = Grid.from_frames(
            spectrum=spectrum,
            offset=np.full((config.n_iters, ), config.position),
            scale=config.exposure,
            background=np.full((config.n_iters, ), 0),
            threshold=threshold,
        )

        shape_hat = VoightPeakShape.from_grid(
            grid=grid,
            show=False,
        )

        #
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
        error.append(mse(y, y_hat))


    plt.plot(thresholds, error)
    plt.show()
