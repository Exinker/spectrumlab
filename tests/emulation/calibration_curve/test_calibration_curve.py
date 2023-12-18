import pytest

import numpy as np

from spectrumlab.emulation.calibration_curve.calibration_curve import CalibrationCurve, CalibrationCurveConfig
from spectrumlab.emulation.detector.characteristic.aperture import RectangularApertureProfile
from spectrumlab.emulation.detector.linear_array_detector import Detector
from spectrumlab.emulation.device import Device
from spectrumlab.emulation.emulation import fetch_emulation, Emulation, SpectrumConfig, EmittedSpectrumEmulationConfig
from spectrumlab.emulation.intensity import IntegralIntensityConfig, InterpolationKind
from spectrumlab.emulation.line import VoigtLineProfile


@pytest.fixture(scope='module')
def config() -> dict:

    class Config:
        # --------        emulation config        --------
        device = Device.GRAND2_II
        detector = Detector.BLPP2000

        n_numbers = 10
        n_frames = 1

        line_width = 25  # in micron
        line_asymmetry = 0
        line_ratio = 0

        aperture_profile = RectangularApertureProfile

        # --------        intensity config        --------
        intensity = IntegralIntensityConfig(
            kind=InterpolationKind.LINEAR,
            interval=3,
        )

        # --------        calibration curve config        --------
        n_probes = 18
        n_parallels = 10

        position = 5
        concentrations = tuple(reversed([10000 * (1/(2**(i))) for i in range(n_probes)]))

        # --------        others        --------
        background_level = 0
        concentration_ratio = 10**(0)  # concentration coefficient

    return Config()  # FIXME: add experiment config


@pytest.fixture(scope='module')
def emulation(config: dict) -> Emulation:

    # emulation
    return fetch_emulation(
        config=EmittedSpectrumEmulationConfig(
            device=config.device,
            detector=config.detector,

            line_profile=VoigtLineProfile(
                width=config.line_width,
                asymmetry=config.line_asymmetry,
                ratio=config.line_ratio,
            ),
            apparatus_profile=None,
            aperture_profile=config.aperture_profile(
                detector=config.detector,
            ),

            spectrum=SpectrumConfig(
                n_numbers=config.n_numbers,
                n_frames=config.n_frames,
            ),
            concentration_ratio=config.concentration_ratio,
            background_level=config.background_level,

            # info='',
        )
    )


@pytest.fixture(scope='module')
def calibration_curve(config: dict, emulation: Emulation) -> CalibrationCurve:
    calibration_curve = CalibrationCurve(
        emulation=emulation,
        config=CalibrationCurveConfig(
            intensity_config=config.intensity,
            n_probes=config.n_probes,
            n_parallels=config.n_parallels,
        ),
    )
    calibration_curve = calibration_curve.setup(
        position=config.position,
        concentrations=config.concentrations,
    )
    calibration_curve = calibration_curve.run(
        verbose=False,
        show=False,
        write=False,
    )

    return calibration_curve


def test_calibration_curve_coeff(config: dict, calibration_curve: CalibrationCurve):
    tolerance = 1e-2
    intercept, slope = calibration_curve.coeff

    assert np.abs(slope - 1) <= tolerance
