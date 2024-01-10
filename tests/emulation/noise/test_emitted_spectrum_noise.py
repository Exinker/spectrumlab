import numpy as np
import pytest

from spectrumlab.emulation.detector.linear_array_detector import Detector
from spectrumlab.emulation.emulation import emulate_emitted_spectrum
from spectrumlab.emulation.noise import EmittedSpectrumNoise


@pytest.mark.parametrize(
    'detector',
    [detector for detector in Detector],
)
def test_emulate_emitted_spectrum_noise(detector: Detector):
    tolerance = 1e-1
    n_times = 10_000
    n_numbers = detector.config.n_pixels

    number = np.arange(n_numbers)
    intensity = np.linspace(0, 100, n_numbers)

    theoretical_noise = EmittedSpectrumNoise(
        detector=detector,
        n_frames=1,
    )(intensity)

    spectrum = emulate_emitted_spectrum(
        number=number,
        intensity=np.array([intensity]*n_times),
        noise=EmittedSpectrumNoise(
            detector=detector,
            n_frames=1,
        ),
        detector=detector,

        is_clipped=False,  # TODO: don't remove! It's for correct estimate a noise about 100[%]!
    )
    emulated_noise = np.std(spectrum.intensity, ddof=1, axis=0)

    assert np.all((emulated_noise - theoretical_noise)/theoretical_noise < tolerance)  # relative deviation

