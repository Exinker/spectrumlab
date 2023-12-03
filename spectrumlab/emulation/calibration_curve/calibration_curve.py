
import os
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import interpolate

from spectrumlab.alias import Frame, Number
from spectrumlab.picture.config import COLOR
from spectrumlab.emulation import Emulation, EmittedSpectrumEmulation, AbsorbedSpectrumEmulation
from spectrumlab.emulation.intensity import IntensityConfig, IntegralIntensityConfig, InterpolationKind, calculate_intensity, calculate_deviation
from .concentration_limits import DynamicRange, calculate_dynamic_range
from .exceptions import EmulationError
from .intensity_limits import IntensityLOD, calculate_intensity_LOD


def _get_filename(emulation: Emulation, extension: Literal['png', 'txt']):
    device = emulation.config.device
    detector = emulation.config.detector
    info = emulation.config.info

    content = '; '.join([f'{item}' for item in [device.config.dispersion, detector.config.name, info] if item])  # FIXME: change device.config.dispersion to device.config.name
    return f'calibration_curve ({content}).{extension}'


@dataclass
class CalibrationCurveConfig:
    intensity_config: IntensityConfig = field(default=IntegralIntensityConfig())

    threshold: float = field(default=5)
    is_clipped: bool = field(default=True)

    n_probes: int = field(default=20)
    n_parallels: int = field(default=5)


class CalibrationCurve:

    def __init__(self, emulation: Emulation, config: CalibrationCurveConfig):

        self.emulation = emulation
        self.config = config

        self._position = None
        self._concentrations = None
        self._intensity_lod = None  # intensity limit of detection (LoD)
        self._data = None
        self._unicorn = None
        self._coeff = None  # intercept, slope
        self._dynamic_range = None

    @property
    def intensity_lod(self) -> IntensityLOD:
        if self._intensity_lod is None:
            raise EmulationError('setup and run the calibration curve before!')

        return self._intensity_lod

    @property
    def dynamic_range(self) -> DynamicRange:
        if self._dynamic_range is None:
            raise EmulationError('setup and run the calibration curve before!')

        return self._dynamic_range

    @property
    def coeff(self) -> tuple[float, float]:
        """Calibration curve's coeffs (intercept, slope) in log10 scale."""
        if self._coeff is None:
            raise EmulationError('setup and run the calibration curve before!')

        return self._coeff

    @property
    def data(self) -> tuple[float, float]:
        if self._data is None:
            raise EmulationError('setup and run the calibration curve before!')

        return self._data

    @property
    def unicorn(self) -> tuple[float, float]:
        if self._unicorn is None:
            raise EmulationError('setup and run the calibration curve before!')

        return self._unicorn

    # --------        handlers        --------
    def setup(self, position: Number, concentrations: tuple[float]):
        """Setup emulation of calibration curve"""
        self._position = position
        self._concentrations = concentrations

        #
        return self

    def run(self, random_state: int | None = None, verbose: bool = True, show: bool = False, write=False):
        """Run emulation."""
        emulation = self.emulation
        config = self.config

        if any(item is None for item in [self._position, self._concentrations]):
            raise EmulationError('Setup the calibration curve before!')
        position = self._position
        concentrations = self._concentrations

        # set random state
        if random_state is not None:
            np.random.seed(random_state)

        # calculate curve's intensity limit of detection (LoD)
        self._intensity_lod = calculate_intensity_LOD(
            emulation=emulation,
            config=config.intensity_config,
        )

        # emulate curve
        data = pd.DataFrame(
            data={'concentration': None, 'intensity': None, 'mask': False},
            columns=['concentration', 'intensity', 'mask'],
            index=pd.MultiIndex.from_product([list(range(config.n_probes)), list(range(config.n_parallels))], names=['probe', 'parallel'])
        )
        unicorn = pd.DataFrame(
            data={'concentration': None, 'intensity': None, 'mask': False},
            columns=['concentration', 'intensity', 'deviation', 'mask'],
            index=pd.Index(list(range(config.n_probes)), name='probe'),
        )  # idealized data (no noise)

        for i, concentration in enumerate(tqdm(concentrations)):
            emulation = emulation.setup(position=position, concentration=concentration)

            # data
            for j in range(config.n_parallels):
                spectrum = emulation.run(is_noised=True, is_clipped=config.is_clipped)

                data.loc[(i,j), 'concentration'] = concentration
                data.loc[(i,j), 'intensity'] = calculate_intensity(
                    spectrum=spectrum,
                    background=emulation.config.background_level,
                    position=position,
                    config=config.intensity_config,
                )
                data.loc[(i,j), 'mask'] = any(spectrum.clipped)

            # unicorn
            spectrum = emulation.run(is_noised=False, is_clipped=config.is_clipped)

            unicorn.loc[i, 'concentration'] = concentration
            unicorn.loc[i, 'intensity'] = calculate_intensity(
                spectrum=spectrum,
                background=emulation.config.background_level,
                position=position,
                config=config.intensity_config,
            )
            unicorn.loc[i, 'deviation'] = calculate_deviation(
                spectrum=spectrum,
                background=emulation.config.background_level,
                position=position,
                config=config.intensity_config,
            )
            unicorn.loc[i, 'mask'] = any(spectrum.clipped)

        self._data = data
        self._unicorn = unicorn

        # emulate curve / mask non-linear part of curve
        x = unicorn['concentration'].apply(lambda x: np.log10(x))
        y = unicorn['intensity'].apply(lambda x: np.log10(x))

        coeff = 0, 1
        while True:
            values = y[~unicorn['mask']] - x[~unicorn['mask']]
            intercept, slope = np.mean(values), 1

            #
            ref = 10**(intercept + slope*x)
            predicted = 10**(y)
            if np.max((100*np.abs(ref - predicted) / ref)[~unicorn['mask']]) > config.threshold:
                unicorn.loc[max(values.index), 'mask'] = True  # mask the last of unmasked!
                coeff = intercept, slope  # update coeff

            else:
                break

        data.loc[unicorn['mask'][unicorn['mask']].index, 'mask'] = True  # mask data
        self._coeff = coeff  # update coeff

        # calculate curve's dynamic range
        self._dynamic_range = calculate_dynamic_range(
            emulation=emulation,
            intensity_lod=self._intensity_lod,
            unicorn=unicorn,
            coeff=self._coeff,
            threshold=config.threshold,
        )

        # verbose
        if verbose:
            print(self.intensity_lod)
            print(self.dynamic_range)

        # show
        if show:
            self.show()

        # write
        if write:
            self.write()

        # return self
        return self

    def show(self, ref: Frame | None = None, concentration_ratio: float = 1, ylim: tuple[float, float] | None = None):
        emulation = self.emulation
        intercept, slope = self.coeff

        if ref is None:
            data = self.data.copy()
            data['concentration'] = data['concentration'].apply(lambda x: np.log10(x) + np.log10(concentration_ratio))
            data['intensity'] = data['intensity'].apply(lambda x: np.log10(x))
        else:
            data = ref.copy()
            data = data.set_index(['probe', 'parallel'])
            data['concentration'] = data['concentration'].apply(lambda x: np.log10(x))
            data['intensity'] = data['intensity'].apply(lambda x: np.log10(x))

        unicorn = self.unicorn.copy()
        unicorn['concentration'] = unicorn['concentration'].apply(lambda x: np.log10(x) + np.log10(concentration_ratio))
        unicorn['intensity'] = unicorn['intensity'].apply(lambda x: np.log10(x))

        # show
        fig, (ax_left, ax_mid, ax_right) = plt.subplots(ncols=3, figsize=(15, 15/3), sharex=True, tight_layout=True,)

        title = '\n'.join([
            # fr'dispersion: {emulation.config.device.config.dispersion:.4f}, nm/mm',
            # fr'detector: {emulation.config.detector.config.name}',
            # fr'$\alpha: {emulation.config.scattering_ratio}$',
            # fr'$\beta: {emulation.config.background_level}$',
            # fr'dynamic_range: {self.dynamic_range.c_min:.4f} - {self.dynamic_range.c_max:.4f} ({np.log10(self.dynamic_range.c_max) - np.log10(self.dynamic_range.c_min):.4f})',
        ])
        plt.suptitle(title)  # TODO: add title's config

        #
        plt.sca(ax_left)

        x = unicorn['concentration']
        y = unicorn['intensity']
        plt.plot(
            x, y,
            color='black', linestyle='none', marker='s', markersize=2,
            alpha=.5,
            label='unicorn',
        )

        x = data['concentration']
        y = data['intensity']
        plt.scatter(
            x, y,
            s=20,
            marker='s',
            facecolors='none',
            edgecolors=[0, 0, 0, 0],
            alpha=.2,
        )

        x = data['concentration'].groupby(level=0, sort=False).mean()
        y = data['intensity'].groupby(level=0, sort=False).mean()
        plt.scatter(
            x, y,
            s=40,
            marker='s',
            facecolors=COLOR['yellow'] if ref is None else COLOR['green'],
            edgecolors=COLOR['yellow'] if ref is None else COLOR['green'],
            alpha=.5,
            label='emulated' if ref is None else 'recorded',
        )

        x = np.log10(self.intensity_lod.value) - intercept
        y = np.log10(self.intensity_lod.value)
        plt.scatter(
            x, y,
            s=40,
            marker='*',
            facecolors='none',
            edgecolors=COLOR['red'],
            label='DL',
        )

        x = np.log10(list(self.dynamic_range))
        y = intercept + slope*x
        plt.plot(
            x, y,
            color='black',
            linestyle=':',
            alpha=.5,
        )

        if ylim:
            plt.ylim(ylim)
        plt.xlabel('$\log_{10}{C}$')
        if isinstance(emulation, EmittedSpectrumEmulation):
            plt.ylabel('$\log_{10}{I}$')
        if isinstance(emulation, AbsorbedSpectrumEmulation):
            plt.ylabel('$\log_{10}{A}$')
        plt.grid(color='grey', linestyle=':')
        plt.legend()

        #
        plt.sca(ax_mid)

        x = data['concentration'].groupby(level=0, sort=False).mean()
        y = data['intensity'].apply(lambda x: 10**(x)).groupby(level=0, sort=False).mean()
        x_grid = unicorn['concentration']
        y_grid = unicorn['intensity']
        y_hat = 10**(interpolate.interp1d(
            x_grid, y_grid,
            kind='linear',
            bounds_error=False,
            fill_value=np.nan,
        )(x))
        dy = 100*(y - y_hat)/y
        plt.scatter(
            x, dy,
            s=20,
            marker='s',
            facecolors=COLOR['yellow'] if ref is None else COLOR['green'],
            edgecolors=COLOR['yellow'] if ref is None else COLOR['green'],
            alpha=.5,
            label='emulated' if ref is None else 'recorded',
        )

        lim = 1.1*np.max(np.abs(dy))
        if lim > 50: lim = 50  # restrict lim
        if lim < 20: lim = 20  # restrict lim
        plt.ylim(-lim, +lim)
        plt.xlabel('$\log_{10}{C}$')
        plt.ylabel('$bias, \%$')
        plt.grid(color='grey', linestyle=':')
        plt.legend()

        #
        plt.sca(ax_right)

        x = unicorn['concentration']
        y = 100 * unicorn['deviation'] / unicorn['intensity'].apply(lambda x: 10**(x))
        plt.plot(
            x, y,
            color='black', linestyle=':', marker='s', markersize=2,
            alpha=.5,
            label='unicorn',
        )

        x = data['concentration'].groupby(level=0, sort=False).mean()
        y = 100 * data['intensity'].apply(lambda x: 10**(x)).groupby(level=0, sort=False).std(ddof=1) / data['intensity'].apply(lambda x: 10**(x)).groupby(level=0, sort=False).mean()
        plt.scatter(
            x, y,
            s=40,
            marker='s',
            facecolors=COLOR['yellow'] if ref is None else COLOR['green'],
            edgecolors=COLOR['yellow'] if ref is None else COLOR['green'],
            alpha=.5,
            label='emulated' if ref is None else 'recorded',
        )

        plt.ylim([-.1, 2])
        plt.xlabel('$\log_{10}{C}$')
        plt.ylabel('$deviation, \%$')
        plt.grid(color='grey', linestyle=':')
        plt.legend()

        #
        filedir = os.path.join('.', 'img')
        if not os.path.isdir(filedir):
            os.mkdir(filedir)
        filename = _get_filename(emulation, extension='png')

        plt.savefig(
            os.path.join(filedir, filename)
        )

        plt.show()

    def write(self):
        emulation = self.emulation
        data = self.data

        #
        filedir = os.path.join('.', 'txt')
        if not os.path.isdir(filedir):
            os.mkdir(filedir)
        filename = _get_filename(emulation, extension='txt')
        filepath = os.path.join(filedir, filename)

        #
        data.to_csv(
            filepath,
            decimal=',',
            sep=';',
            encoding='utf-8',
            index=True,
            columns=data.columns,
        )
