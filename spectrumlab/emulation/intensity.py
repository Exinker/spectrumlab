import matplotlib.pyplot as plt
import numpy as np

from spectrumlab.alias import Array, Number
from spectrumlab.core.grid import Grid, integrate_grid, interpolate_grid
from spectrumlab.emulation.spectrum import AbsorbedSpectrum, EmittedSpectrum, Spectrum
from spectrumlab.peak.intensity import AmplitudeIntensityConfig
from spectrumlab.peak.intensity import ApproxIntensityConfig
from spectrumlab.peak.intensity import IntegralIntensityConfig, InterpolationKind
from spectrumlab.peak.intensity import IntensityConfig


# --------        estimate intensity        --------
def _estimate_intensity(grid: Grid, mask: Array[bool], position: Number, config: IntensityConfig) -> float:
    """Interface to estimate analyte peak's intensity."""

    if isinstance(config, AmplitudeIntensityConfig):
        f = interpolate_grid(grid, kind=InterpolationKind.NEAREST)

        return f(position)

    if isinstance(config, IntegralIntensityConfig):
        return integrate_grid(
            grid=grid,
            position=position,
            interval=config.interval,
            kind=config.kind,
        )

    if isinstance(config, ApproxIntensityConfig):
        shape = config.approx_shape

        return np.dot(grid.y[~mask], grid.y[~mask]) / np.dot(grid.y[~mask], shape(grid.x[~mask], position=position, intensity=1))

    raise ValueError(f'calculate_intensity: config {config} is not supported!')


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


# def _correct_intensity(value: float, config: IntensityConfig) -> float:
#     if config.correction is None:
#         return value

#     #
#     if isinstance(config.correction, Normalization):
#         return config.correction(value)
#     if isinstance(config.correction, KatskovLinearization):
#         return config.correction(value)

#     raise ValueError(f'Correction method {config.correction} is not supported!')


# --------        calculate intensity        --------
def calculate_intensity(spectrum: Spectrum, background: float, position: Number, config: IntensityConfig, ylim: tuple[float, float] | None = None, verbose: bool = False, show: bool = False) -> float:
    """Calculate a peak's intensity with selected config."""
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
        config=config,
    )

    # correct intensity
    # value = _correct_intensity(
    #     value=value,
    #     config=config,
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
            config=config,
        ) ** (1/2)

        #
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), tight_layout=True)

        if isinstance(config, AmplitudeIntensityConfig):
            plt.step(
                grid.x, background + grid.y,
                where='mid',
                linestyle='-', linewidth=1, color=config.color,
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
                alpha=0.2, facecolor=config.color, edgecolor=config.color,
                label='область\nинтегр.',
            )

        if isinstance(config, IntegralIntensityConfig):

            if config.kind == InterpolationKind.NEAREST:
                plt.step(
                    grid.x, background + grid.y,
                    where='mid',
                    linestyle='-', linewidth=1, color=config.color,
                    marker='.', markersize=5,
                    label='$s_{k}$',
                )

                x = np.linspace(position - config.interval/2, position + config.interval/2, 101)
                f = interpolate_grid(background + grid, kind=InterpolationKind.NEAREST)
                plt.fill_between(
                    x,
                    background,
                    f(x),
                    step='mid',
                    alpha=0.2, facecolor=config.color, edgecolor=config.color,
                    label='область\nинтегр.',
                )

            if config.kind == InterpolationKind.LINEAR:
                plt.plot(
                    grid.x, background + grid.y,
                    linestyle='-', linewidth=1, color=config.color,
                    marker='.', markersize=5,
                    label='$s_{k}$',
                )

                x = np.linspace(position - config.interval/2, position + config.interval/2, 101)
                f = interpolate_grid(background + grid, kind=InterpolationKind.LINEAR)
                plt.fill_between(
                    x,
                    background,
                    f(x),
                    step='mid',
                    alpha=0.2, facecolor=config.color, edgecolor=config.color,
                    label='область\nинтегр.',
                )

        if isinstance(config, ApproxIntensityConfig):
            plt.plot(
                grid.x, background + grid.y,
                linestyle='none', color=config.color,
                marker='.', markersize=5,
                label='$s_{k}$',
            )

            x = np.linspace(min(grid.x), max(grid.x), 101)
            y = background + config.approx_shape(x, position=position, intensity=value)
            plt.plot(
                x, y,
                alpha=0.2, color=config.color,
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


def calculate_deviation(spectrum: Spectrum, background: float, position: Number, config: IntensityConfig) -> float:
    """Calculate a standart deviation of peak's intensity with selected config."""
    grid = Grid(
        x=spectrum.number,
        y=(spectrum.deviation ** 2).flatten(),
        units=Number,
    )
    mask = spectrum.clipped.flatten()

    if isinstance(config, IntegralIntensityConfig):
        value = _estimate_intensity(
            grid=grid,
            mask=mask,
            position=position,
            config=config,
        ) ** (1/2)

        return value

    raise ValueError(f'calculate_deviation: config {config} is not supported!')
