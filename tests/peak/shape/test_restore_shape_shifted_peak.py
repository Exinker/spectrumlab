from functools import partial

import numpy as np
import pytest

from spectrumlab.emulation.aperture import Aperture, RectangularApertureShape
from spectrumlab.emulation.apparatus import Apparatus, VoigtApparatusShape
from spectrumlab.emulation.detector import Detector
from spectrumlab.emulation.peak import ShiftedExperiment, ShiftedExperimentConfig
from spectrumlab.peak.shape import VoightPeakShape, Grid, restore_shape_from_grid

from core import distance
from config import *


THRESHOLD = 1


# --------        fixtures        --------
@pytest.fixture(scope='module')
def detector() -> Detector:
    return DETECTOR


@pytest.fixture(scope='module')
def shape() -> VoigtApparatusShape:
    return SHAPE


@pytest.fixture(scope='module')
def experiment(detector: Detector, shape: VoightPeakShape) -> ShiftedExperiment:

    experiment = ShiftedExperiment(
        config=ShiftedExperimentConfig(
            n_numbers=N_NUMBERS,
            n_frames=N_FRAMES,

            detector=detector,
            apparatus=Apparatus(
                detector=detector,
                shape=shape,
            ),
            aperture=Aperture(
                detector=detector,
                shape=RectangularApertureShape(),
            ),

            exposure=EXPOSURE,
            position=POSITION + np.linspace(-.5, +.5, N_ITERS, endpoint=False),
            intensity=INTENSITY,
        ),
    )
    experiment = experiment.setup(
        verbose=False,
        show=False,
    )

    return experiment


@pytest.fixture(scope='module')
def shape_hat(experiment: ShiftedExperiment) -> VoightPeakShape:
    config = experiment.config

    # spectrum
    spectrum = experiment.run(is_noised=IS_NOISED)

    # grid
    grid = Grid.from_frames(
        spectrum=spectrum,
        offset=config.position,
        scale=np.full((config.n_iters, ), config.exposure * config.intensity),
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
    config=ShiftedExperimentConfig(
        n_numbers=N_NUMBERS,
        n_frames=N_FRAMES,

        detector=detector,
        apparatus=Apparatus(
            detector=detector,
            shape=shape,
        ),
        aperture=Aperture(
            detector=detector,
            shape=RectangularApertureShape(),
        ),

        exposure=EXPOSURE,
        position=POSITION + np.linspace(-.5, +.5, N_ITERS, endpoint=False),
        intensity=INTENSITY,
    )

    # experiment
    experiment = ShiftedExperiment(
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
        offset=config.position,
        scale=np.full((config.n_iters, ), config.exposure * config.intensity),
        background=np.full((config.n_iters, ), 0),
        threshold=THRESHOLD,
    )
    shape_hat = restore_shape_from_grid(
        grid=grid,
        show=True,
    )
