import numpy as np
import pytest

from spectrumlab.emulation.detector import Detector
from spectrumlab.emulation.emulation import emulate_absorbed_spectrum
from spectrumlab.emulation.noise import EmittedSpectrumNoise, AbsorbedSpectrumNoise


@pytest.mark.parametrize(
    'detector',
    [detector for detector in Detector],
)
def test_emulate_absorbed_spectrum_noise(detector: Detector):
    tolerance = 1e-1
    n_times = 10_000
    n_numbers = detector.config.n_pixels
    base_level = 100
    base_noise = EmittedSpectrumNoise(
        detector=detector,
        n_frames=1,
    )

    number = np.arange(n_numbers)
    absorbance = np.logspace(-3, np.log10(3), n_numbers)


    theoretical_noise = AbsorbedSpectrumNoise(
        detector=detector,
        n_frames=1,
        base_level=base_level,
        base_noise=base_noise,
    )(absorbance)

    spectrum = emulate_absorbed_spectrum(
        intensity=np.array([base_level * 10**(-absorbance)]*n_times),
        number=number,
        noise=EmittedSpectrumNoise(
            detector=detector,
            n_frames=1,
        ),
        base_level=base_level,
        base_noise=base_noise,
        detector=detector,

        is_clipped=False,
    )
    emulated_noise = np.std(spectrum.intensity, ddof=1, axis=0)

    mask = ~np.isnan(emulated_noise)
    assert np.all((emulated_noise[mask] - theoretical_noise[mask])/theoretical_noise[mask] < tolerance)  # relative deviation
