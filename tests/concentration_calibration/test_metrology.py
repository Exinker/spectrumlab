import numpy as np
import pytest

from spectrumlab.typing import Array
from spectrumlab.concentration_calibration.metrology import LOD, LOQ


# --------        limits (LOD and LOQ)        --------
class TestLimits:
    N = 100_000_000
    MEAN = 1
    DEVIATION = .01
    COEFF = (0, 1)

    @pytest.fixture(scope='module')
    def intensity(self) -> Array[float]:
        np.random.seed(42)

        return self.MEAN + self.DEVIATION*np.random.randn(self.N)

    def test_default_lod(self, intensity: Array[float]):
        tolerance = 1e-3

        k = 3
        lod = LOD.from_json(
            data={
                'mean': np.mean(intensity),
                'deviation': np.std(intensity, ddof=1),
            },
            coeff=self.COEFF,
        )

        value = (lod.intensity - self.MEAN) / self.DEVIATION
        assert np.abs(1 - value/k) <= tolerance

    def test_default_loq(self, intensity: Array[float]):
        tolerance = 1e-3

        k = 10
        loq = LOQ.from_json(
            data={
                'mean': np.mean(intensity),
                'deviation': np.std(intensity, ddof=1),
            },
            coeff=self.COEFF,
        )

        value = (loq.intensity - self.MEAN) / self.DEVIATION
        assert np.abs(1 - value/k) <= tolerance
