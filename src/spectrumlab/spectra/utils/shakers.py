import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from spectrumlab.spectra.emitted_spectrum import EmittedSpectrum
from spectrumlab.spectra.utils.interpolators import InterpolationKind, interpolate, interpolate_lanczos
from spectrumlab.types import Array


def find_offset(x: Array[float], y: Array[float], y_base: Array[float]) -> float:
    """Find offset of y values with respect the y_base.

    Lanczos interpolation used.
    """

    def fitness(x: Array[float], y: Array[float], y_base: Array[float], offset: float) -> float:
        y_hat = interpolate_lanczos(
            x, y_base,
            a=3,
        )(offset)

        return np.sum((y - y_hat)**2)

    return minimize(lambda offset: fitness(x, y, y_base, offset), 0)['x'][0]


def find_scale(y: Array[float], y_base: Array[float]) -> float:
    """Find scale of y."""

    return np.dot(y, y_base) / np.dot(y_base, y_base)


def shake_up(
    spectrum: EmittedSpectrum,
    n_iters: int,
    show: bool = False,
) -> tuple[EmittedSpectrum, Array[float], Array[float]]:
    """Centralize and normalize a spectrum intensity data."""

    x = spectrum.number
    y_base = np.mean(spectrum.intensity, axis=0)
    y_centralized = np.array(spectrum.intensity)
    y_normalized = np.array(spectrum.intensity)

    offset = np.zeros((spectrum.n_times,))
    scale = np.ones((spectrum.n_times,))
    for _ in range(n_iters):

        # update offset
        for t, y in enumerate(y_normalized):
            offset[t] += find_offset(x, y, y_base)

        # update y_base (centralize)
        for t, y in enumerate(spectrum.intensity):
            y_centralized[t] = interpolate(
                x, y,
                offset=-offset[t], kind=InterpolationKind.LANCZOS,
            )

        y_base = np.mean(y_centralized, axis=0)

        # update y_base (normalize)
        # y_normalized = y_centralized
        for t, y in enumerate(y_centralized):
            scale[t] = find_scale(y, y_base)
            y_normalized[t] = y * scale[t]

        y_base = np.mean(y_normalized, axis=0)

        if show:
            fig, (ax_left, ax_mid, ax_right) = plt.subplots(ncols=3, figsize=(18, 4), tight_layout=True)

            x = spectrum.number
            for t, y in enumerate(spectrum.intensity):
                ax_left.plot(
                    x - offset[t], y / scale[t],
                    color='red', linestyle='none', marker='s', markersize=3,
                    alpha=1,
                )

            ax_left.set_xlabel(r'$number$')
            ax_left.set_ylabel(r'$I$ [$\%$]')
            ax_left.grid(color='grey', linestyle=':')

            ax_mid.plot(offset, color='black')
            ax_mid.set_xlabel(r'$time$')
            ax_mid.set_ylabel(r'$offset$')
            ax_mid.grid(color='grey', linestyle=':')

            ax_right.plot(scale, color='black')
            ax_right.set_xlabel(r'$time$')
            ax_right.set_ylabel(r'$scale$')
            ax_right.grid(color='grey', linestyle=':')

            plt.show()

    #
    return EmittedSpectrum(intensity=y_normalized, wavelength=spectrum.wavelength), offset, scale
