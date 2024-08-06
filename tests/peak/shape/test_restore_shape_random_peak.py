from functools import partial

import numpy as np
import pytest

from spectrumlab.emulation.aperture import Aperture, RectangularApertureShape
from spectrumlab.emulation.apparatus import Apparatus, VoigtApparatusShape
from spectrumlab.emulation.detector import Detector
from spectrumlab.emulation.peak import RandomPeakExperiment, RandomPeakExperimentConfig
from spectrumlab.peak.shape import VoigtPeakShape, restore_shape_from_spectrum

from config import DETECTOR, EXPOSURE, IS_NOISED, N_FRAMES, N_ITERS, SHAPE  # noqa: I100
from core import distance


N_NUMBERS = 2048
MU = -1
SIGMA = .5


# --------        fixtures        --------
@pytest.fixture(scope='module')
def detector() -> Detector:
    return DETECTOR


@pytest.fixture(scope='module')
def shape() -> VoigtApparatusShape:
    return SHAPE


@pytest.fixture(scope='module')
def experiment(detector: Detector, shape: VoigtPeakShape) -> RandomPeakExperiment:

    experiment = RandomPeakExperiment(
        config=RandomPeakExperimentConfig(
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
            # position=np.linspace(0, N_NUMBERS, N_ITERS),
            # intensity=np.full(N_ITERS, 1),
            position=np.random.uniform(0, N_NUMBERS, size=(N_ITERS,)),
            intensity=10**np.random.normal(MU, SIGMA, size=(N_ITERS,)),
        ),
    )
    experiment = experiment.setup(
        verbose=True,
    )

    return experiment


@pytest.fixture(scope='module')
def shape_hat(experiment: RandomPeakExperiment) -> VoigtPeakShape:

    # spectrum
    spectrum = experiment.run(is_noised=IS_NOISED)

    # shape_hat
    shape_hat = restore_shape_from_spectrum(
        spectrum=spectrum,
        noise=experiment.noise,
    )

    return shape_hat


@pytest.mark.skip(reason='FIXME: исправить!')
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


@pytest.mark.skip(reason='FIXME: исправить!')
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

    # config
    config = RandomPeakExperimentConfig(
        n_numbers=N_NUMBERS,
        n_frames=N_FRAMES,

        detector=DETECTOR,
        apparatus=Apparatus(
            detector=DETECTOR,
            shape=SHAPE,
        ),
        aperture=Aperture(
            detector=DETECTOR,
            shape=RectangularApertureShape(),
        ),

        exposure=EXPOSURE,
        # position=np.linspace(0, N_NUMBERS, N_ITERS),
        # intensity=np.full(N_ITERS, 1),
        position=np.random.uniform(0, N_NUMBERS, size=(N_ITERS,)),
        intensity=10**np.random.normal(MU, SIGMA, size=(N_ITERS,)),
    )

    # experiment
    experiment = RandomPeakExperiment(  # noqa: F811
        config=config,
    )
    experiment = experiment.setup(
        verbose=True,
    )

    # spectrum
    spectrum = experiment.run(is_noised=IS_NOISED)

    # restore shape
    restore_shape_from_spectrum(
        spectrum=spectrum,
        noise=experiment.noise,
        verbose=True,
        show=True,
    )
