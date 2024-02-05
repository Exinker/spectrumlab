from functools import partial

import numpy as np
import pytest

from spectrumlab.emulation.aperture import Aperture, RectangularApertureShape
from spectrumlab.emulation.apparatus import Apparatus, VoigtApparatusShape
from spectrumlab.emulation.detector import Detector
from spectrumlab.emulation.peak import ScaledExperiment, ScaledExperimentConfig
from spectrumlab.peak.shape import VoigtPeakShape, restore_shape_from_grid
from spectrumlab.peak.shape._grid import _Grid

from core import distance
from config import *

THRESHOLD = 100


# --------        fixtures        --------
@pytest.fixture(scope='module')
def detector() -> Detector:
    return DETECTOR


@pytest.fixture(scope='module')
def shape() -> VoigtApparatusShape:
    return SHAPE


@pytest.fixture(scope='module')
def experiment(detector: Detector, shape: VoigtPeakShape) -> ScaledExperiment:

    experiment = ScaledExperiment(
        config=ScaledExperimentConfig(
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

            exposure=np.array([2**x for x in range(N_ITERS)]),
            position=POSITION,
            intensity=INTENSITY,
        ),
    )
    experiment = experiment.setup(
        verbose=False,
        show=False,
    )

    return experiment


@pytest.fixture(scope='module')
def shape_hat(experiment: ScaledExperiment) -> VoigtPeakShape:
    config = experiment.config

    # spectrum
    spectrum = experiment.run(is_noised=IS_NOISED)

    # grid
    grid = _Grid.from_frames(
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
def test_params_error(detector: Detector, shape: VoigtApparatusShape, shape_hat: VoigtPeakShape):
    tolerance = 1e-3  # 0.1 [%]

    assert distance(
        xi=shape.width,
        xi_hat=shape_hat.width*detector.pitch,
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


def test_shape_error(detector: Detector, shape: VoigtApparatusShape, shape_hat: VoigtPeakShape):
    tolerance = 1e-6

    f = partial(shape, x0=0, pitch=detector.pitch)
    f_hat = partial(VoigtApparatusShape(
        width=shape_hat.width*detector.pitch,
        asymmetry=shape.asymmetry,
        ratio=shape.ratio,
    ), x0=0, pitch=detector.pitch)

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
    config = ScaledExperimentConfig(
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

        exposure=np.array([2**x for x in range(N_ITERS)]),
        position=POSITION,
        intensity=INTENSITY,
    )

    # experiment
    experiment = ScaledExperiment(
        config=config,
    )
    experiment = experiment.setup(
        verbose=False,
        show=False,
    )

    # spectrum
    spectrum = experiment.run(is_noised=IS_NOISED)

    # restore shape
    grid = _Grid.from_frames(
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
