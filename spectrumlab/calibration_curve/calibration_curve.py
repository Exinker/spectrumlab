
import os
from dataclasses import dataclass, field
from typing import Literal, NewType

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import interpolate

from spectrumlab.alias import Frame, Number
from spectrumlab.picture.config import COLOR
from spectrumlab.peak.intensity import LOQ
from .metrology import DynamicRange, calculate_dynamic_range


Intercept = NewType('Intercept', float)
Slope = NewType('Slope', float)


# def _get_filename(emulation: Emulation, extension: Literal['png', 'txt']):
#     device = emulation.config.device
#     detector = emulation.config.detector
#     info = emulation.config.info

#     content = '; '.join([f'{item}' for item in [device.config.dispersion, detector.config.name, info] if item])  # FIXME: change device.config.dispersion to device.config.name
#     return f'calibration_curve ({content}).{extension}'


# @dataclass
# class CalibrationCurveConfig:
#     intensity_config: IntensityConfig = field(default=IntegralIntensityConfig())

#     threshold: float = field(default=5)
#     is_clipped: bool = field(default=True)

#     n_probes: int = field(default=20)
#     n_parallels: int = field(default=5)


# class CalibrationCurve:

#     def __init__(self, data: Frame, loq: LOQ, lol: LOL):
#         self._data = data
#         self._loq = loq
#         self._lol = lol

#         self._coeff = calculate_coeff()
#         self._dynamic_range = calculate_dynamic_range()

#     @property
#     def coeff(self) -> tuple[Intercept, Slope]:
#         """Calibration curve's coeffs (intercept, slope) in log10 scale."""
#         if self._coeff is None:
#             raise EmulationError('setup and run the calibration curve before!')

#         return self._coeff


#     def calculate_coeff(self) -> tuple[Intercept, Slope]:

#         data = self._data
        


#         x = data['concentration'][~data['mask']].apply(lambda x: np.log10(x)).groupby(level=0, sort=False).mean().astype(float).values
#         y = data['intensity'][~data['mask']].apply(lambda x: np.log10(x)).groupby(level=0, sort=False).mean().astype(float).values

#         slope, intercept = np.polyfit(x, y, deg=1)
#         slope, intercept

#     # --------        handlers        --------
#     def show(self, ref: Frame | None = None, concentration_ratio: float = 1, ylim: tuple[float, float] | None = None):
#         emulation = self.emulation
#         intercept, slope = self.coeff

#         if ref is None:
#             data = self.data.copy()
#             data['concentration'] = data['concentration'].apply(lambda x: np.log10(x) + np.log10(concentration_ratio))
#             data['intensity'] = data['intensity'].apply(lambda x: np.log10(x))
#         else:
#             data = ref.copy()
#             data = data.set_index(['probe', 'parallel'])
#             data['concentration'] = data['concentration'].apply(lambda x: np.log10(x))
#             data['intensity'] = data['intensity'].apply(lambda x: np.log10(x))

#         unicorn = self.unicorn.copy()
#         unicorn['concentration'] = unicorn['concentration'].apply(lambda x: np.log10(x) + np.log10(concentration_ratio))
#         unicorn['intensity'] = unicorn['intensity'].apply(lambda x: np.log10(x))

#         # show
#         fig, (ax_left, ax_mid, ax_right) = plt.subplots(ncols=3, figsize=(15, 15/3), sharex=True, tight_layout=True,)

#         title = '\n'.join([
#             # fr'dispersion: {emulation.config.device.config.dispersion:.4f}, nm/mm',
#             # fr'detector: {emulation.config.detector.config.name}',
#             # fr'$\alpha: {emulation.config.scattering_ratio}$',
#             # fr'$\beta: {emulation.config.background_level}$',
#             # fr'dynamic_range: {self.dynamic_range.c_min:.4f} - {self.dynamic_range.c_max:.4f} ({np.log10(self.dynamic_range.c_max) - np.log10(self.dynamic_range.c_min):.4f})',
#         ])
#         plt.suptitle(title)  # TODO: add title's config

#         #
#         plt.sca(ax_left)

#         x = unicorn['concentration']
#         y = unicorn['intensity']
#         plt.plot(
#             x, y,
#             color='black', linestyle='none', marker='s', markersize=2,
#             alpha=.5,
#             label='unicorn',
#         )

#         x = data['concentration']
#         y = data['intensity']
#         plt.scatter(
#             x, y,
#             s=20,
#             marker='s',
#             facecolors='none',
#             edgecolors=[0, 0, 0, 0],
#             alpha=.2,
#         )

#         x = data['concentration'].groupby(level=0, sort=False).mean()
#         y = data['intensity'].groupby(level=0, sort=False).mean()
#         plt.scatter(
#             x, y,
#             s=40,
#             marker='s',
#             facecolors=COLOR['yellow'] if ref is None else COLOR['green'],
#             edgecolors=COLOR['yellow'] if ref is None else COLOR['green'],
#             alpha=.5,
#             label='emulated' if ref is None else 'recorded',
#         )

#         x = np.log10(self.intensity_lod.value) - intercept
#         y = np.log10(self.intensity_lod.value)
#         plt.scatter(
#             x, y,
#             s=40,
#             marker='*',
#             facecolors='none',
#             edgecolors=COLOR['red'],
#             label='DL',
#         )

#         x = np.log10(list(self.dynamic_range))
#         y = intercept + slope*x
#         plt.plot(
#             x, y,
#             color='black',
#             linestyle=':',
#             alpha=.5,
#         )

#         if ylim:
#             plt.ylim(ylim)
#         plt.xlabel('$\log_{10}{C}$')
#         if isinstance(emulation, EmittedSpectrumEmulation):
#             plt.ylabel('$\log_{10}{I}$')
#         if isinstance(emulation, AbsorbedSpectrumEmulation):
#             plt.ylabel('$\log_{10}{A}$')
#         plt.grid(color='grey', linestyle=':')
#         plt.legend()

#         #
#         plt.sca(ax_mid)

#         x = data['concentration'].groupby(level=0, sort=False).mean()
#         y = data['intensity'].apply(lambda x: 10**(x)).groupby(level=0, sort=False).mean()
#         x_grid = unicorn['concentration']
#         y_grid = unicorn['intensity']
#         y_hat = 10**(interpolate.interp1d(
#             x_grid, y_grid,
#             kind='linear',
#             bounds_error=False,
#             fill_value=np.nan,
#         )(x))
#         dy = 100*(y - y_hat)/y
#         plt.scatter(
#             x, dy,
#             s=20,
#             marker='s',
#             facecolors=COLOR['yellow'] if ref is None else COLOR['green'],
#             edgecolors=COLOR['yellow'] if ref is None else COLOR['green'],
#             alpha=.5,
#             label='emulated' if ref is None else 'recorded',
#         )

#         lim = 1.1*np.max(np.abs(dy))
#         if lim > 50: lim = 50  # restrict lim
#         if lim < 20: lim = 20  # restrict lim
#         plt.ylim(-lim, +lim)
#         plt.xlabel('$\log_{10}{C}$')
#         plt.ylabel('$bias, \%$')
#         plt.grid(color='grey', linestyle=':')
#         plt.legend()

#         #
#         plt.sca(ax_right)

#         x = unicorn['concentration']
#         y = 100 * unicorn['deviation'] / unicorn['intensity'].apply(lambda x: 10**(x))
#         plt.plot(
#             x, y,
#             color='black', linestyle=':', marker='s', markersize=2,
#             alpha=.5,
#             label='unicorn',
#         )

#         x = data['concentration'].groupby(level=0, sort=False).mean()
#         y = 100 * data['intensity'].apply(lambda x: 10**(x)).groupby(level=0, sort=False).std(ddof=1) / data['intensity'].apply(lambda x: 10**(x)).groupby(level=0, sort=False).mean()
#         plt.scatter(
#             x, y,
#             s=40,
#             marker='s',
#             facecolors=COLOR['yellow'] if ref is None else COLOR['green'],
#             edgecolors=COLOR['yellow'] if ref is None else COLOR['green'],
#             alpha=.5,
#             label='emulated' if ref is None else 'recorded',
#         )

#         plt.ylim([-.1, 2])
#         plt.xlabel('$\log_{10}{C}$')
#         plt.ylabel('$deviation, \%$')
#         plt.grid(color='grey', linestyle=':')
#         plt.legend()

#         #
#         filedir = os.path.join('.', 'img')
#         if not os.path.isdir(filedir):
#             os.mkdir(filedir)
#         filename = _get_filename(emulation, extension='png')

#         plt.savefig(
#             os.path.join(filedir, filename)
#         )

#         plt.show()

#     def write(self):
#         emulation = self.emulation
#         data = self.data

#         #
#         filedir = os.path.join('.', 'txt')
#         if not os.path.isdir(filedir):
#             os.mkdir(filedir)
#         filename = _get_filename(emulation, extension='txt')
#         filepath = os.path.join(filedir, filename)

#         #
#         data.to_csv(
#             filepath,
#             decimal=',',
#             sep=';',
#             encoding='utf-8',
#             index=True,
#             columns=data.columns,
#         )
