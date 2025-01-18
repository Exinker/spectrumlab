import matplotlib.pyplot as plt
import numpy as np

from spectrumlab.emulations.intensity import AbstractIntensityTransformer
from spectrumlab.emulations.intensity.patch_calculator import (
    EmulatedMixin,
    patch_calculator,
)
from spectrumlab.emulations.intensity.transform_intensity import transform_intensity
from spectrumlab.emulations.spectrum import (
    AbsorbedSpectrum,
    EmittedSpectrum,
    Spectrum,
)
from spectrumlab.grid import Grid, InterpolationKind, interpolate_grid
from spectrumlab.peak.intensity import (
    AbstractIntensityCalculator,
    AmplitudeIntensityCalculator,
    ApproxIntensityCalculator,
    IntegralIntensityCalculator,
)
from spectrumlab.peak.units import R
from spectrumlab.types import Number


def calculate_intensity(
    spectrum: Spectrum,
    background: float,
    position: Number,
    calculator: AbstractIntensityCalculator,
    transformer: AbstractIntensityTransformer | None = None,
    ylim: tuple[float, float] | None = None,
    verbose: bool = False,
    show: bool = False,
) -> R:
    """Calculate a peak's intensity with selected calculator."""
    if not isinstance(calculator, EmulatedMixin):
        calculator = patch_calculator(calculator)

    grid = Grid(
        x=spectrum.number,
        y=(spectrum.intensity - background).flatten(),
        units=Number,
    )
    mask = spectrum.clipped.flatten()

    value = calculator.calculate(
        grid=grid,
        mask=mask,
        position=position,
    )

    value = transform_intensity(
        value,
        transformer=transformer,
    )

    if verbose:
        print(f'intensity={value:.4f}, A')

    if show:
        noise = calculator.calculate(
            grid=Grid(
                x=spectrum.number,
                y=(spectrum.deviation ** 2).flatten(),
                units=Number,
            ),
            mask=mask,
            position=position,
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

    return value


def calculate_deviation(
    spectrum: Spectrum,
    background: float,
    position: Number,
    calculator: AbstractIntensityCalculator,
) -> R:
    """Calculate a standart deviation of peak's intensity with selected calculator."""
    if not isinstance(calculator, EmulatedMixin):
        calculator = patch_calculator(calculator)

    grid = Grid(
        x=spectrum.number,
        y=(spectrum.deviation ** 2).flatten(),
        units=Number,
    )
    mask = spectrum.clipped.flatten()

    if isinstance(calculator, IntegralIntensityCalculator):
        value = calculator.calculate(
            grid=grid,
            mask=mask,
            position=position,
        ) ** (1/2)

        return value

    raise ValueError(f'calculate_deviation: calculator {calculator} is not supported!')
