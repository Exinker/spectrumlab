import os
import pytest

import numpy as np

from spectrumlab.emulation.calibration_curve.calibration_curve import CalibrationCurve, CalibrationCurveConfig
from spectrumlab.emulation.detector.characteristic.aperture import RectangularApertureProfile
from spectrumlab.emulation.detector.linear_array_detector import Detector
from spectrumlab.emulation.device import Device
from spectrumlab.emulation.emulation import fetch_emulation, Emulation, SpectrumConfig, EmittedSpectrumEmulationConfig
from spectrumlab.emulation.experiment import EmittedExperimentConfigNaive as ExperimentConfig
from spectrumlab.emulation.intensity import IntegralIntensityConfig, InterpolationKind
from spectrumlab.emulation.line import VoigtLineProfile


@pytest.fixture(scope='module')
def config() -> ExperimentConfig:
    filedir, _ = os.path.split(__file__)

    return ExperimentConfig.from_ini(filedir=filedir, filename='config (naive).ini')


@pytest.fixture(scope='module')
def emulation(config: ExperimentConfig) -> Emulation:
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
def calibration_curve(config: ExperimentConfig, emulation: Emulation) -> CalibrationCurve:
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


def test_calibration_curve_coeff(config: ExperimentConfig, calibration_curve: CalibrationCurve):
    tolerance = 1e-2
    intercept, slope = calibration_curve.coeff

    assert np.abs(slope - 1) <= tolerance
