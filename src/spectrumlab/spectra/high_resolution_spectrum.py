import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

from spectrumlab.detectors import Detector
from spectrumlab.grids import Grid
from spectrumlab.spectra.base_spectrum import SpectrumABC
from spectrumlab.spectra.emitted_spectrum import EmittedSpectrum
from spectrumlab.types import Array, MicroMeter, NanoMeter, Number


def calculate_factor(ratio: float) -> int:
    """Resolution enhancement factor."""

    for n in range(1, 10):
        if n*ratio % 1 < 1e-9:
            return int(n*ratio)

    raise ValueError


class HighResolutionSpectrum(SpectrumABC):

    def __init__(
        self,
        shots: tuple[EmittedSpectrum],
        number: Array[Number],
        wavelength: Array[NanoMeter],
        move: MicroMeter,
        detector: Detector | None = None,
        **kwargs,
    ) -> None:
        n_numbers = len(number)

        for spectrum in shots:
            assert spectrum.n_times == 1
            assert spectrum.n_numbers == n_numbers

        # factor
        factor = calculate_factor(
            ratio=detector.pitch/move,
        )

        # grid
        x_grid: Array[MicroMeter] = np.linspace(0, n_numbers, n_numbers*factor + 1) * detector.pitch

        y_grid = [[] for _ in x_grid]
        for i, x in enumerate(x_grid):
            for t, spectrum in enumerate(shots):
                mask = np.abs(number*detector.pitch - x - t*move) < 1e-9

                if np.any(mask):
                    y_grid[i].extend(spectrum.intensity[mask])
        y_grid = np.array([np.nanmean(value) if value else np.nan for value in y_grid])

        w_grid = interpolate.interp1d(
            number, wavelength,
            kind='linear',
            bounds_error=False,
            fill_value=0,
        )(x_grid/detector.pitch)

        mask = np.isfinite(y_grid)
        x_grid, y_grid, w_grid = x_grid[mask], y_grid[mask], w_grid[mask]

        grid = Grid(
            x=x_grid,
            y=y_grid,
            units=MicroMeter,
        )

        #
        super().__init__(intensity=y_grid, number=x_grid/detector.pitch, wavelength=w_grid, detector=detector, **kwargs)

        self.move = move
        self.factor = factor  # points per pitch
        self.shots = shots
        self.grid = grid

    @classmethod
    def from_spectrum(cls, spectrum: EmittedSpectrum, move: MicroMeter):
        assert spectrum.n_times > 1

        shots = []
        for t in range(spectrum.n_times):
            spe = EmittedSpectrum(
                intensity=spectrum.intensity[t, :],
                wavelength=spectrum.wavelength,
                detector=spectrum.detector,
            )
            shots.append(spe)
        shots = tuple(shots)

        return cls(
            shots=shots,
            number=spectrum.number,
            wavelength=spectrum.wavelength,
            move=move,
            detector=spectrum.detector,
        )

    def show(self):
        detector = self.detector
        grid = self.grid

        for t, spectrum in enumerate(self.shots):
            x = spectrum.number*detector.pitch - t*self.move
            y = spectrum.intensity
            plt.plot(
                x, y,
                color='red', linestyle='none', marker='s', markersize=3,
                alpha=1,
            )

        plt.plot(
            grid.x, grid.y,
            color='black', linestyle='-', linewidth=1, marker='s', markersize=1.5,
            alpha=1,
        )

        plt.xlabel(r'$x$ [$\mu m$]')
        plt.ylabel(r'$f(x)$')
        plt.grid(True, linestyle=':')

        plt.show()
