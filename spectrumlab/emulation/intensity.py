
from dataclasses import dataclass, field
from typing import Literal, Mapping, TypeAlias

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, interpolate

from spectrumlab.alias import Array, Number
from spectrumlab.picture.config import COLOR
from spectrumlab.emulation.spectrum import Spectrum, EmittedSpectrum, AbsorbedSpectrum
from spectrumlab.peak.intensity import InterpolationKind, integrate_grid, interpolate_grid
from spectrumlab.peak.intensity import IntensityConfig, AmplitudeIntensityConfig, IntegralIntensityConfig, ApproxIntensityConfig
from spectrumlab.peak.intensity.utils import integrate_grid, InterpolationKind



# --------        estimate intensity        --------
def _estimate_intensity(x_grid: Array, y_grid: Array, position: Number, config: IntensityConfig) -> float:
    """Interface to estimate analyte peak's intensity."""

    if isinstance(config, AmplitudeIntensityConfig):
        f = interpolate_grid(x_grid, y_grid, kind='nearest')

        return f(position)

    if isinstance(config, IntegralIntensityConfig):
        return integrate_grid(
            x_grid, y_grid,
            position=position,
            interval=config.interval,
            kind={
                InterpolationKind.NEAREST: 'nearest',
                InterpolationKind.LINEAR: 'linear',
            }.get(config.kind),
        )

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
    x_grid = spectrum.number
    y_grid = (spectrum.intensity - background).flatten()

    # estimate intensity
    value = _estimate_intensity(
        x_grid, y_grid,
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
            spectrum.number, (spectrum.deviation ** 2).flatten(),
            position=position,
            config=config,
        ) ** (1/2)

        #
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), tight_layout=True)

        if isinstance(config, AmplitudeIntensityConfig):
            plt.step(
                x_grid, background + y_grid,
                where='mid',
                linestyle='-', linewidth=1, color=config.color,
                marker='.', markersize=5,
                label='$s_{k}$',
            )

            x = np.linspace(position - 1/2, position + 1/2, 101)
            f = interpolate_grid(x_grid, background + y_grid, kind='nearest')
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
                    x_grid, background + y_grid,
                    where='mid',
                    linestyle='-', linewidth=1, color=config.color,
                    marker='.', markersize=5,
                    label='$s_{k}$',
                )

                x = np.linspace(position - config.interval/2, position + config.interval/2, 101)
                f = interpolate_grid(x_grid, background + y_grid, kind='nearest')
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
                    x_grid, background + y_grid,
                    linestyle='-', linewidth=1, color=config.color,
                    marker='.', markersize=5,
                    label='$s_{k}$',
                )

                x = np.linspace(position - config.interval/2, position + config.interval/2, 101)
                f = interpolate_grid(x_grid, background + y_grid, kind='linear')
                plt.fill_between(
                    x, 
                    background,
                    f(x),
                    step='mid',
                    alpha=0.2, facecolor=config.color, edgecolor=config.color,
                    label='область\nинтегр.',
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

        plt.xlabel('number')
        plt.ylabel({
            EmittedSpectrum: r'$I, \%$',
            AbsorbedSpectrum: r'$A$',
        }.get(type(spectrum)))
        plt.grid(color='grey', linestyle=':')

        plt.show()

    #
    return value


def calculate_deviation(spectrum: Spectrum, background: float, position: Number, config: IntensityConfig) -> float:
    """Calculate a standart deviation of peak's intensity with selected config."""
    x_grid = spectrum.number
    y_grid = (spectrum.deviation ** 2).flatten()

    if isinstance(config, IntegralIntensityConfig):
        value = _estimate_intensity(
            x_grid, y_grid,
            position=position,
            config=config,
        ) ** (1/2)

        return value

    raise ValueError(f'calculate_deviation: config {config} is not supported!')
