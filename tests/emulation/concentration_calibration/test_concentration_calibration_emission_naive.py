import numpy as np
import pytest

from spectrumlab.emulations.concentration_calibration import ConcentrationCalibration, ConcentrationCalibrationConfig, EmittedExperimentConfigNaive as ExperimentConfig
from spectrumlab.emulations.detector import Detector
from spectrumlab.emulations.emulators import Emulation
from spectrumlab.emulations.intensity import AbstractIntensityCalculator, AmplitudeIntensityCalculator, IntegralIntensityCalculator


@pytest.fixture(scope='module')
def concentration_calibration(config: ExperimentConfig, emulation: Emulation) -> ConcentrationCalibration:
    concentration_calibration = ConcentrationCalibration(
        emulation=emulation,
        config=ConcentrationCalibrationConfig(
            intensity_calculator=config.intensity_calculator,
            n_probes=config.n_probes,
            n_parallels=config.n_parallels,
        ),
    )
    concentration_calibration = concentration_calibration.setup(
        position=config.position,
        concentrations=config.concentrations,
    )
    concentration_calibration = concentration_calibration.run(
        verbose=False,
        show=False,
        write=False,
    )

    return concentration_calibration


class TestConcentrationCalibration:
    tolerance = 1e-9

    # --------        coeff        --------
    def test_concentration_calibration_coeff(self, config: ExperimentConfig, concentration_calibration: ConcentrationCalibration):
        intercept, slope = concentration_calibration.coeff

        assert np.abs(slope - 1) <= self.tolerance

    # --------        LOD        --------
    def test_concentration_calibration_lod(self, config: ExperimentConfig, concentration_calibration: ConcentrationCalibration):
        lod = self.calcualte_lod(detector=config.detector, calculator=config.intensity_calculator)

        assert np.abs(concentration_calibration.lod.intensity - lod) < self.tolerance

    @staticmethod
    def calcualte_lod(detector: Detector, calculator: AbstractIntensityCalculator, k: float = 3) -> float:
        """Calculate theoretical LOD value."""

        if isinstance(calculator, AmplitudeIntensityCalculator):
            read_noise = 100 * detector.config.read_noise / detector.config.capacity
            deviation = read_noise

            return k * deviation

        if isinstance(calculator, IntegralIntensityCalculator):
            read_noise = 100 * detector.config.read_noise / detector.config.capacity
            deviation = np.sqrt(calculator.interval) * read_noise

            return k * deviation

        raise ValueError(f'Calculator: {calculator} is not supported yet!')
