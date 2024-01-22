import numpy as np
import matplotlib.pyplot as plt

from spectrumlab.alias import Array, Number, MicroMeter, NanoMeter
from spectrumlab.core.grid import Grid
from spectrumlab.emulation.detector import Detector
from spectrumlab.spectrum import Spectrum
from spectrumlab.spectrum.base_spectrum import BaseSpectrum


def calculate_power(step: MicroMeter, move: MicroMeter, tolerance: float = 1e-9) -> int:
    power = step/move

    for n in range(1, 10):
        if n*power % 1 < tolerance:
            return int(n*power)
        
    raise ValueError


class HighResolutionSpectrum(BaseSpectrum):

    def __init__(self, intensity: Array[float], power: int, wavelength: Array[NanoMeter] | None = None, number: Array[Number] | None = None, deviation: Array[float] | None = None, clipped: Array[bool] | None = None, detector: Detector | None = None):
        super().__init__(intensity=intensity, wavelength=wavelength, number=number, deviation=deviation, clipped=clipped, detector=detector)

        self.power = power  # points per step

    # 
    @classmethod
    def from_spectrum(cls, spectrum: Spectrum, move: MicroMeter, detector: Detector, show: bool = False, tolerance: float = 1e-9) -> 'HighResolutionSpectrum':

        def inner(spectrum: Spectrum, move: MicroMeter, step: MicroMeter, power: int) -> Grid:
            n_times, n_numbers = spectrum.shape

            # x_grid
            x_grid: Array[MicroMeter] = np.linspace(0, n_numbers, n_numbers*power + 1) * step

            # y_grid
            number = np.arange(n_numbers)

            y_grid = [[] for i in range(n_numbers*power + 1)]
            for i, x in enumerate(x_grid):
                for t in range(n_times):
                    mask = np.abs(number*step - x - t*move) < tolerance

                    if np.any(mask):
                        y_grid[i].extend(spectrum.intensity[t, mask])

            y_grid = np.array([np.nanmean(value) for value in y_grid])

            # mask nan
            mask = np.isfinite(y_grid)
            x_grid = x_grid[mask]
            y_grid = y_grid[mask]

            # sort
            index = np.argsort(x_grid)
            x_grid = x_grid[index]
            y_grid = y_grid[index]

            #
            return Grid(
                x=x_grid,
                y=y_grid,
                step=step,
            )

        # 
        step = detector.config.width
        power = calculate_power(step, move)
        grid = inner(
            spectrum,
            move=move,
            step=step,
            power=power,
        )

        # show
        if show:
            n_times, n_numbers = spectrum.shape
            number = np.arange(n_numbers)

            for t in range(n_times):
                x = number*grid.step - t*move
                y = spectrum.intensity[t, :]
                plt.plot(
                    x, y,
                    color='red', linestyle='none', marker='s', markersize=3,
                    alpha=1,
                )

            x = grid.x
            y = grid.y
            plt.plot(
                x, y,
                color='black', linestyle='-', linewidth=1, marker='s', markersize=1.5,
                alpha=1,
            )

            plt.xlabel(r'$x$ [$\mu m$]')
            plt.ylabel(r'$f(x)$')
            plt.grid(True, linestyle=':')

            plt.show()

        #
        return cls(
            intensity=grid.y,
            power=power,
            number=grid.x / grid.step,
            detector=detector,
        )

    # --------        handlers        --------
    def show(self) -> None:
        raise NotImplementedError


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
    intensity = []
    for t in range(n_times):
        spectrum = emulation.setup(
            position=n_numbers//2 + t*(move/detector.config.width),
            concentration=1,
        ).run(
            is_noised=True,
        )
        intensity.append(spectrum.intensity)
    intensity = np.array(intensity)

    # spectrum
    spectrum = HighResolutionSpectrum.from_spectrum(
        spectrum=EmittedSpectrum(
            intensity=intensity,
        ),
        move=move,
        detector=detector,
        show=True,
    )
