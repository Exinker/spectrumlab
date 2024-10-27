from functools import partial

import numpy as np
import pytest

from spectrumlab.emulations.apertures import Aperture, RectangularApertureShape
from spectrumlab.emulations.apparatus import Apparatus, VoigtApparatusShape
from spectrumlab.emulations.detectors import Detector
from spectrumlab.emulations.experiments.spectra_emulation import ShiftedExperiment, ShiftedExperimentConfig
from spectrumlab.grid import Grid
from spectrumlab.peak.shape import VoigtPeakShape, restore_shape_from_grid

from config import DETECTOR, EXPOSURE, INTENSITY, IS_NOISED, N_FRAMES, N_ITERS, N_NUMBERS, POSITION, SHAPE  # noqa: I100
from core import distance


THRESHOLD = 1


# --------        fixtures        --------
@pytest.fixture(scope='module')
def detector() -> Detector:
    return DETECTOR


@pytest.fixture(scope='module')
def shape() -> VoigtApparatusShape:
    return SHAPE


@pytest.fixture(scope='module')
def experiment(detector: Detector, shape: VoigtPeakShape) -> ShiftedExperiment:

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
def shape_hat(experiment: ShiftedExperiment) -> VoigtPeakShape:
    config = experiment.config

    # spectrum
    spectrum = experiment.run(is_noised=IS_NOISED)

    # grid
    grid = Grid.factory(spectrum=spectrum).create_from_frames(
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

    shape_hat = VoigtApparatusShape(
        width=shape_hat.width*detector.pitch,
        asymmetry=shape.asymmetry,
        ratio=shape.ratio,
    )
    f_hat = partial(shape_hat, x0=0, pitch=detector.pitch)

    rx = 100
    dx = .01
    x = np.linspace(-rx, +rx, 2*int(rx/dx) + 1)

    y = f(x)
    y_hat = f_hat(x)

    #
    assert np.all(np.abs(y_hat - y) < tolerance)


if __name__ == '__main__':

    # experiment
    config = ShiftedExperimentConfig(
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
        position=POSITION + np.linspace(-.5, +.5, N_ITERS, endpoint=False),
        intensity=INTENSITY,
    )
    experiment = ShiftedExperiment(  # noqa: F811
        config=config,
    )
    experiment = experiment.setup(
        verbose=False,
        show=False,
    )

    # spectrum
    spectrum = experiment.run(is_noised=IS_NOISED)

    # restore shape
    grid = Grid.factory(spectrum=spectrum).create_from_frames(
        offset=config.position,
        scale=np.full((config.n_iters, ), config.exposure * config.intensity),
        background=np.full((config.n_iters, ), 0),
        threshold=THRESHOLD,
    )
    restore_shape_from_grid(
        grid=grid,
        show=True,
    )
