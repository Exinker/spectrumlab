import pytest

import numpy as np

from spectrumlab.alias import Array
from spectrumlab.peak.intensity import LOD, LOQ


# --------        limits (LOD and LOQ)        --------
class TestLimits:
    N = 100_000_000
    TOLERANCE = 1e-3

    @pytest.fixture(scope='module')
    def intensity(self) -> Array[float]:
        np.random.seed(42)

        return np.random.randn(self.N,)

    def test_default_lod(self, intensity: Array[float]):
        k = 3
        lod = LOD(
            deviation=np.std(intensity, ddof=1),
            units='%',
        )

        assert np.abs(1 - lod.value/k) <= self.TOLERANCE

    def test_default_loq(self, intensity: Array[float]):
        k = 10
        loq = LOQ(
            deviation=np.std(intensity, ddof=1),
            units='%',
        )

        assert np.abs(1 - loq.value/k) <= self.TOLERANCE

    def test_to_concentration(self, intensity: Array[float]):
        lod = LOD(
            deviation=np.std(intensity, ddof=1),
            units='%',
        )
        concentration = lod.to_concentration(coeff=[0, 1])

        assert np.abs(concentration - 3) <= self.TOLERANCE
