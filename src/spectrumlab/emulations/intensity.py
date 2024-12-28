import matplotlib.pyplot as plt
import numpy as np

from spectrumlab.emulations.spectrum import AbsorbedSpectrum, EmittedSpectrum, Spectrum
from spectrumlab.grid import Grid, InterpolationKind, integrate_grid, interpolate_grid
from spectrumlab.peak.intensity import (
    AbstractIntensityCalculator,
    AmplitudeIntensityCalculator,
    ApproxIntensityCalculator,
    IntegralIntensityCalculator,
)
from spectrumlab.types import Array, Number


# --------        estimate intensity        --------
def _estimate_intensity(
    grid: Grid,
    mask: Array[bool],
    position: Number,
    calculator: AbstractIntensityCalculator,
) -> float:
    """Interface to estimate analyte peak's intensity."""

    if isinstance(calculator, AmplitudeIntensityCalculator):
        f = interpolate_grid(grid, kind=InterpolationKind.NEAREST)

        return f(position)

    if isinstance(calculator, IntegralIntensityCalculator):
        return integrate_grid(
            grid=grid,
            position=position,
            interval=calculator.interval,
            kind=calculator.kind,
        )

    if isinstance(calculator, ApproxIntensityCalculator):
        shape = calculator.shape

        return np.dot(grid.y[~mask], grid.y[~mask]) / np.dot(grid.y[~mask], shape(grid.x[~mask], position=position, intensity=1))  # noqa: E501

    raise ValueError(f'calculate_intensity: calculator {calculator} is not supported yet!')


# --------        correct intensity        --------
# @dataclass
# class Normalization:
#     """Normalization by value."""
#     coeff: float

#     def __call__(self, value: float) -> float:
#         return value / self.coeff


# @dataclass
# class KatskovLinearization:
#     """Katskov linearization.

#     Аn introduction to multi-element atomic-absorption analysis
#     Katskov D.
#     DOI: 10.15826/analitika.2018.22.4.001
#     """
#     coeff: tuple[float, float]

#     def __call__(self, value: float) -> float:
#         c1, c2 = self.coeff

#         if value > c1:
#             value = ((value + c1)**2)/(4*c1)
#         if value > c2:
#             value = ((value + c2)**2)/(4*c2)

#         return value


# IntensityCorrection: TypeAlias = Normalization | KatskovLinearization


# def _correct_intensity(value: float, calculator: AbstractIntensityCalculator) -> float:
#     if config.correction is None:
#         return value

#     #
#     if isinstance(config.correction, Normalization):
#         return config.correction(value)
#     if isinstance(config.correction, KatskovLinearization):
#         return config.correction(value)

#     raise ValueError(f'Correction method {config.correction} is not supported!')


# --------        calculate intensity        --------
def calculate_intensity(
    spectrum: Spectrum,
    background: float,
    position: Number,
    calculator: AbstractIntensityCalculator,
    ylim: tuple[float, float] | None = None,
    verbose: bool = False,
    show: bool = False,
) -> float:
    """Calculate a peak's intensity with selected calculator."""
    grid = Grid(
        x=spectrum.number,
        y=(spectrum.intensity - background).flatten(),
        units=Number,
    )
    mask = spectrum.clipped.flatten()

    # estimate intensity
    value = _estimate_intensity(
        grid=grid,
        mask=mask,
        position=position,
        calculator=calculator,
    )

    # correct intensity
    # value = _correct_intensity(
    #     value=value,
    #     calculator=calculator,
    # )

    # verbose
    if verbose:
        print(f'intensity={value:.4f}, A')

    # show
    if show:
        noise = _estimate_intensity(
            grid=Grid(
                x=spectrum.number,
                y=(spectrum.deviation ** 2).flatten(),
                units=Number,
            ),
            mask=mask,
            position=position,
            calculator=calculator,
        ) ** (1/2)

        #
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), tight_layout=True)

        if isinstance(calculator, AmplitudeIntensityCalculator):
            plt.step(
                grid.x, background + grid.y,
                where='mid',
                linestyle='-', linewidth=1, color=calculator.color,
                marker='.', markersize=5,
                label=r'$s_{k}$',
            )

            x = np.linspace(position - 1/2, position + 1/2, 101)
            f = interpolate_grid(background + grid, kind=InterpolationKind.NEAREST)
            plt.fill_between(
                x,
                background,
                f(x),
                step='mid',
                alpha=0.2, facecolor=calculator.color, edgecolor=calculator.color,
                label='область\nинтегр.',
            )

        if isinstance(calculator, IntegralIntensityCalculator):

            if calculator.kind == InterpolationKind.NEAREST:
                plt.step(
                    grid.x, background + grid.y,
                    where='mid',
                    linestyle='-', linewidth=1, color=calculator.color,
                    marker='.', markersize=5,
                    label='$s_{k}$',
                )

                x = np.linspace(position - calculator.interval/2, position + calculator.interval/2, 101)
                f = interpolate_grid(background + grid, kind=InterpolationKind.NEAREST)
                plt.fill_between(
                    x,
                    background,
                    f(x),
                    step='mid',
                    alpha=0.2, facecolor=calculator.color, edgecolor=calculator.color,
                    label='область\nинтегр.',
                )

            if calculator.kind == InterpolationKind.LINEAR:
                plt.plot(
                    grid.x, background + grid.y,
                    linestyle='-', linewidth=1, color=calculator.color,
                    marker='.', markersize=5,
                    label='$s_{k}$',
                )

                x = np.linspace(position - calculator.interval/2, position + calculator.interval/2, 101)
                f = interpolate_grid(background + grid, kind=InterpolationKind.LINEAR)
                plt.fill_between(
                    x,
                    background,
                    f(x),
                    step='mid',
                    alpha=0.2, facecolor=calculator.color, edgecolor=calculator.color,
                    label='область\nинтегр.',
                )

        if isinstance(calculator, ApproxIntensityCalculator):
            plt.plot(
                grid.x, background + grid.y,
                linestyle='none', color=calculator.color,
                marker='.', markersize=5,
                label='$s_{k}$',
            )

            x = np.linspace(min(grid.x), max(grid.x), 101)
            y = background + calculator.shape(x, position=position, intensity=value)
            plt.plot(
                x, y,
                alpha=0.2, color=calculator.color,
            )

        if ylim:
            plt.ylim(ylim)

        content = '\n'.join([
            fr'$I$: {value:.4f} [%]',
            fr'$\frac{{\Delta I}}{{I}}$: {100*noise/value:.4f} [%]',
        ])
        plt.text(
            0.05, 0.95,
            content,
            transform=ax.transAxes,
            ha='left', va='top',
        )

        plt.xlabel(r'number')
        plt.ylabel({
            EmittedSpectrum: r'$I$ [$\%$]',
            AbsorbedSpectrum: r'$A$',
        }.get(type(spectrum)))
        plt.grid(color='grey', linestyle=':')

        plt.show()

    #
    return value


def calculate_deviation(
    spectrum: Spectrum,
    background: float,
    position: Number,
    calculator: AbstractIntensityCalculator,
) -> float:
    """Calculate a standart deviation of peak's intensity with selected calculator."""
    grid = Grid(
        x=spectrum.number,
        y=(spectrum.deviation ** 2).flatten(),
        units=Number,
    )
    mask = spectrum.clipped.flatten()

    if isinstance(calculator, IntegralIntensityCalculator):
        value = _estimate_intensity(
            grid=grid,
            mask=mask,
            position=position,
            calculator=calculator,
        ) ** (1/2)

        return value

    raise ValueError(f'calculate_deviation: calculator {calculator} is not supported!')
