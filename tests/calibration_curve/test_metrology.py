import pytest

import numpy as np

from spectrumlab.alias import Array
from spectrumlab.calibration_curve.metrology import LOD, LOQ


# --------        limits (LOD and LOQ)        --------
class TestLimits:
    N = 100_000_000
    COEFF = (0, 1)
    TOLERANCE = 1e-3

    @pytest.fixture(scope='module')
    def intensity(self) -> Array[float]:
        np.random.seed(42)

        return np.random.randn(self.N,)

    def test_default_lod(self, intensity: Array[float]):
        k = 3
        lod = LOD.from_deviation(
            deviation=np.std(intensity, ddof=1),
            coeff=self.COEFF,
        )

        assert np.abs(1 - lod.intensity/k) <= self.TOLERANCE

    def test_default_loq(self, intensity: Array[float]):
        k = 10
        loq = LOQ.from_deviation(
            deviation=np.std(intensity, ddof=1),
            coeff=self.COEFF,
        )

        assert np.abs(1 - loq.intensity/k) <= self.TOLERANCE

    def test_to_concentration(self, intensity: Array[float]):
        lod = LOD.from_deviation(
            deviation=np.std(intensity, ddof=1),
            coeff=self.COEFF,
        )

        assert np.abs(lod.concentration - 3) <= self.TOLERANCE
