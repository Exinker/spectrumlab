import numpy as np
import matplotlib.pyplot as plt

from spectrumlab.alias import Array, Number, MicroMeter, NanoMeter
from spectrumlab.core.grid import Grid
from spectrumlab.emulation.detector import Detector
from spectrumlab.spectrum import Spectrum
from spectrumlab.spectrum.base_spectrum import BaseSpectrum


def calculate_factor(ratio: float) -> int:
    """Resolution enhancement factor."""

    for n in range(1, 10):
        if n*ratio % 1 < 1e-9:
            return int(n*ratio)
        
    raise ValueError


class HighResolutionSpectrum(BaseSpectrum):

    def __init__(self, shots: tuple[Spectrum], number: Array[Number], move: MicroMeter, detector: Detector | None = None, **kwargs):
        n_numbers = len(number)
        n_moves = len(shots)

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

        mask = np.isfinite(y_grid)
        x_grid, y_grid = x_grid[mask], y_grid[mask]

        grid = Grid(
            x=x_grid,
            y=y_grid,
            step=detector.pitch/factor,
        )

        #
        super().__init__(intensity=grid.y, number=grid.x/detector.pitch, detector=detector, **kwargs)

        self.move = move
        self.factor = factor  # points per pitch
        self.shots = shots
        self.grid = grid

    # --------        handlers        --------
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


if __name__ == '__main__':
    from spectrumlab.emulation.aperture import Aperture, RectangularApertureShape
    from spectrumlab.emulation.apparatus import Apparatus, VoigtApparatusShape
    from spectrumlab.emulation.detector import Detector
    from spectrumlab.emulation.device import Device
    from spectrumlab.emulation.emulation import fetch_emulation, SpectrumConfig, EmittedSpectrumEmulationConfig
    from spectrumlab.emulation.spectrum import Spectrum, EmittedSpectrum


    import warnings
    warnings.filterwarnings('ignore')


    # emulation
    move: MicroMeter = .5
    n_times, n_numbers = 28, 50

    detector = Detector.BLPP2000
    aperture = Aperture(
        detector=detector,
        shape=RectangularApertureShape(),
    )
    apparatus = Apparatus(
        detector=detector,
        shape=VoigtApparatusShape(
            width=25,
            asymmetry=0,
            ratio=0.1,
        ),
    )

    emulation = fetch_emulation(
        config=EmittedSpectrumEmulationConfig(
            device=Device.GRAND2_I,
            detector=detector,

            line=None,
            apparatus=apparatus,
            aperture=aperture,

            spectrum=SpectrumConfig(
                n_numbers=n_numbers,
                n_frames=1,
            ),
            concentration_ratio=1,
            background_level=0,

            # info='',
        )
    )

    # intensity
    shots = []
    for t in range(n_times):
        spectrum = emulation.setup(
            position=n_numbers//2 + t*(move/detector.config.width),
            concentration=1,
        ).run(
            is_noised=True,
        )
        shots.append(spectrum)

    # spectrum
    spectrum = HighResolutionSpectrum(
        shots=shots,
        number=emulation.number,
        move=move,
        detector=emulation.detector,
    )
    spectrum.show()
