import os
from dataclasses import dataclass, field
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from spectrumlab.concentration_calibration import (
    AbstractConcentrationCalibration,
    Intercept,
    LOD,
    LOL,
    LOQ,
    Slope,
    estimate_lol,
)  # noqa: I100
from spectrumlab.emulations.emulators import Emulation
from spectrumlab.emulations.intensity import (
    AbstractIntensityCalculator,
    IntegralIntensityCalculator,
    calculate_deviation,
    calculate_intensity,
)
from spectrumlab.grid import InterpolationKind
from spectrumlab.picture.alpha import ALPHA
from spectrumlab.picture.color import COLOR
from spectrumlab.types import Frame, Number, Series

from .exceptions import EmulationError
from .metrology import DynamicRange, estimate_blank_deviation, estimate_blank_mean, estimate_dynamic_range


@dataclass
class ConcentrationCalibrationConfig:
    intensity_calculator: AbstractIntensityCalculator = field(default=IntegralIntensityCalculator(interval=3, kind=InterpolationKind.LINEAR))

    concentration_blank: float = field(default=0)
    threshold: float = field(default=0.05)
    is_clipped: bool = field(default=True)

    n_probes: int = field(default=20)
    n_parallels: int = field(default=5)


class ConcentrationCalibration(AbstractConcentrationCalibration):

    def __init__(self, emulation: Emulation, config: ConcentrationCalibrationConfig):
        self.emulation = emulation
        self.config = config

        self._position = None
        self._concentrations = None
        self._data = None
        self._unicorn = None
        self._coeff = None
        self._lod = None
        self._loq = None
        self._lol = None
        self._dynamic_range = None

    @property
    def position(self) -> Number:
        if self._position is None:
            raise EmulationError('Setup the concentration calibration before!')

        return self._position

    @property
    def concentrations(self) -> tuple[float]:
        if self._concentrations is None:
            raise EmulationError('Setup the concentration calibration before!')

        return self._concentrations

    @property
    def data(self) -> tuple[float, float]:
        if self._data is None:
            raise EmulationError('setup and run the concentration calibration before!')

        return self._data

    @property
    def unicorn(self) -> tuple[float, float]:
        if self._unicorn is None:
            raise EmulationError('setup and run the concentration calibration before!')

        return self._unicorn

    @property
    def coeff(self) -> tuple[Intercept, Slope]:
        """Calibration curve's coeffs in log10 scale."""
        if self._coeff is None:
            raise EmulationError('setup and run the concentration calibration before!')

        return self._coeff

    @property
    def lod(self) -> LOD:
        if self._lod is None:
            raise EmulationError('setup and run the concentration calibration before!')

        return self._lod

    @property
    def loq(self) -> LOQ:
        if self._loq is None:
            raise EmulationError('setup and run the concentration calibration before!')

        return self._loq

    @property
    def lol(self) -> LOL:
        if self._lol is None:
            raise EmulationError('setup and run the concentration calibration before!')

        return self._lol

    @property
    def dynamic_range(self) -> DynamicRange:
        if self._dynamic_range is None:
            raise EmulationError('setup and run the concentration calibration before!')

        return self._dynamic_range

    def setup(self, position: Number, concentrations: tuple[float]):
        """Setup emulation of concentration calibration"""
        self._position = position
        self._concentrations = concentrations

        #
        return self

    def run(self, random_state: int | None = None, verbose: bool = True, show: bool = False, write: bool = False):
        """Run emulation."""
        emulation = self.emulation
        config = self.config

        position = self.position
        concentrations = self.concentrations

        if random_state is not None:
            np.random.seed(random_state)

        # emulate blank
        loq = LOQ.calculate(
            mean=estimate_blank_mean(
                emulation=emulation,
                calculator=config.intensity_calculator,
            ),
            deviation=estimate_blank_deviation(
                emulation=emulation,
                calculator=config.intensity_calculator,
            ),
            k=10,
        )

        # emulate data
        data = pd.DataFrame(
            data={'concentration': None, 'intensity': None, 'mask': False},
            columns=['concentration', 'intensity', 'mask'],
            index=pd.MultiIndex.from_product([list(range(config.n_probes)), list(range(config.n_parallels))], names=['probe', 'parallel']),
        )
        unicorn = pd.DataFrame(
            data={'concentration': None, 'intensity': None, 'mask': False},
            columns=['concentration', 'intensity', 'deviation', 'mask'],
            index=pd.Index(list(range(config.n_probes)), name='probe'),
        )  # idealized data (no noise)

        for i, concentration in enumerate(tqdm(concentrations, leave=True)):
            emulation = emulation.setup(position=position, concentration=concentration)

            # data
            for j in range(config.n_parallels):
                spectrum = emulation.run(is_noised=True, is_clipped=config.is_clipped)

                data.loc[(i, j), 'concentration'] = concentration
                data.loc[(i, j), 'intensity'] = calculate_intensity(
                    spectrum=spectrum,
                    background=emulation.config.background_level,
                    position=position,
                    calculator=config.intensity_calculator,
                )

                is_traced = data.loc[(i, j), 'intensity'] < loq
                is_clipped = any(spectrum.clipped)
                data.loc[(i, j), 'mask'] = is_traced or is_clipped

            # unicorn
            spectrum = emulation.run(is_noised=False, is_clipped=config.is_clipped)

            unicorn.loc[i, 'concentration'] = concentration
            unicorn.loc[i, 'intensity'] = calculate_intensity(
                spectrum=spectrum,
                background=emulation.config.background_level,
                position=position,
                calculator=config.intensity_calculator,
            )
            unicorn.loc[i, 'deviation'] = calculate_deviation(
                spectrum=spectrum,
                background=emulation.config.background_level,
                position=position,
                calculator=config.intensity_calculator,
            )
            unicorn.loc[i, 'mask'] = any(spectrum.clipped)

        self._data = data
        self._unicorn = unicorn

        # fit
        self.fit()

        # verbose
        if verbose:
            print(self.lod)
            print(self.dynamic_range)

        if show:
            self.show()

        # write
        if write:
            self.write()

        return self

    def fit(self):
        emulation = self.emulation
        config = self.config
        data = self._data
        unicorn = self._unicorn

        # emulate curve / mask non-linear part of curve
        x = unicorn['concentration'].apply(lambda x: np.log10(x))
        y = unicorn['intensity'].apply(lambda x: np.log10(x))

        intercept, slope = 0, 1
        while True:
            values = y[~unicorn['mask']] - x[~unicorn['mask']]
            intercept, slope = np.mean(values), 1

            #
            i_true = 10**(intercept + slope*x)
            i_hat = 10**(y)
            if np.max((np.abs(i_true - i_hat) / i_true)[~unicorn['mask']]) > config.threshold:
                unicorn.loc[max(values.index), 'mask'] = True  # mask the last of unmasked!
            else:
                break

        data.loc[unicorn['mask'][unicorn['mask']].index, 'mask'] = True  # mask data

        self._coeff = intercept, slope  # update coeff

        # calculate limits
        mean = estimate_blank_mean(
            emulation=emulation,
            calculator=config.intensity_calculator,
        )
        deviation = estimate_blank_deviation(
            emulation=emulation,
            calculator=config.intensity_calculator,
        )

        self._lod = LOD.from_json(
            data={
                'mean': mean,
                'deviation': deviation,
            },
            coeff=self.coeff,
        )
        self._loq = LOQ.from_json(
            data={
                'mean': mean,
                'deviation': deviation,
            },
            coeff=self.coeff,
        )
        self._lol = estimate_lol(
            self.unicorn,
            coeff=self.coeff,
            threshold=config.threshold,
        )

        # calculate curve's dynamic range
        self._dynamic_range = estimate_dynamic_range(
            emulation=emulation,
            coeff=self.coeff,
            loq=self.loq,
            lol=self.lol,
        )

    def predict(self, intensity: Series) -> Series:
        interpect, slope = self.coeff

        return 10**((intensity.apply(lambda x: np.log10(x)) - interpect) / slope)

    def show(self, ref: Frame | None = None, concentration_ratio: float = 1, save: bool = False):
        """Show concentration calibration."""

        unicorn = self.unicorn.copy()
        unicorn['concentration'] = concentration_ratio * unicorn['concentration']

        if ref is None:
            data = self.data.copy()
            data['concentration'] = concentration_ratio * data['concentration']
        else:
            data = ref.copy()
            data['mask'] = False
            data = data.set_index(['probe', 'parallel'])

        color = self._get_color(
            mask=data['mask'].groupby(level=0, sort=False).max().astype(bool),
            color=COLOR['yellow'] if ref is None else COLOR['green'],
        )
        alpha = self._get_alpha(
            mask=data['mask'].groupby(level=0, sort=False).max().astype(bool),
            alpha=ALPHA['probe'],
        )

        fig, (ax_left, ax_mid, ax_right) = plt.subplots(ncols=3, figsize=(15, 15/3), sharex=True, tight_layout=True)

        title = ''
        plt.suptitle(title)  # TODO: add title's config

        #
        plt.sca(ax_left)

        x = unicorn['concentration']
        y = unicorn['intensity']
        plt.plot(
            x, y,
            color='black', linestyle='none', marker='s', markersize=2,
            alpha=ALPHA['default'],
            label='theoretical',
        )

        x = data['concentration']
        y = data['intensity']
        plt.scatter(
            x, y,
            s=20,
            marker='s',
            facecolors='none',
            edgecolors=[0, 0, 0, 0],
            alpha=ALPHA['parallel'],
        )

        x = data['concentration'].groupby(level=0, sort=False).mean()
        y = data['intensity'].groupby(level=0, sort=False).mean()
        plt.scatter(
            x, y,
            s=40,
            marker='s',
            facecolors=color,
            edgecolors=color,
            alpha=alpha,
            label='emulated' if ref is None else 'recorded',
        )

        x = self.lod.concentration
        y = self.lod.intensity
        plt.scatter(
            x, y,
            s=40,
            marker='*',
            facecolors='none',
            edgecolors=COLOR['red'],
            label='LoD',
        )

        x = self.dynamic_range.concentration
        y = self.dynamic_range.intensity
        plt.plot(
            x, y,
            color='black',
            linestyle=':',
            alpha=ALPHA['default'],
        )

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$C$')
        plt.ylabel(r'$R$')
        plt.grid(color='grey', linestyle=':')
        plt.legend()

        #
        plt.sca(ax_mid)

        c_true = unicorn['concentration']
        c_hat = self.predict(unicorn['intensity'])
        if ref is None:
            bias = 100*(c_hat - c_true)/c_true
            x = c_true
            y = bias
            plt.plot(
                x, y,
                color='black', linestyle='none', marker='s', markersize=2,
                alpha=ALPHA['default'],
                label='theoretical',
            )

        c_true = data['concentration'].groupby(level=0, sort=False).mean()
        if ref is None:
            c_hat = self.predict(data['intensity'].groupby(level=0, sort=False).mean())
            bias = 100*(c_hat - c_true)/c_true
            x = c_true
            y = bias
            plt.scatter(
                x, y,
                s=20,
                marker='s',
                facecolors=color,
                edgecolors=color,
                alpha=alpha,
                label='emulated' if ref is None else 'recorded',
            )

        plt.xscale('log')
        plt.xlabel(r'$C$')
        plt.ylabel(r'$bias, \%$')
        plt.grid(color='grey', linestyle=':')
        plt.legend()

        #
        plt.sca(ax_right)

        c_true = unicorn['concentration']
        c_hat = self.predict(unicorn['deviation'])
        deviation = 100 * c_hat / c_true
        x = c_true
        y = deviation
        plt.plot(
            x, y,
            color='black', linestyle='none', marker='s', markersize=2,
            alpha=ALPHA['default'],
            label='theoretical',
        )

        c_true = data['concentration'].groupby(level=0, sort=False).mean()
        c_hat = self.predict(data['intensity'])
        deviation = 100 * c_hat.groupby(level=0, sort=False).std(ddof=1) / c_true
        x = c_true
        y = deviation
        plt.scatter(
            x, y,
            s=40,
            marker='s',
            facecolors=color,
            edgecolors=color,
            alpha=alpha,
            label='emulated' if ref is None else 'recorded',
        )

        # plt.ylim([-.1, 2])
        plt.xscale('log')
        plt.xlabel(r'$C$')
        plt.ylabel(r'$deviation, \%$')
        plt.grid(color='grey', linestyle=':')
        plt.legend()

        # save
        if save:
            filedir = os.path.join('.', 'img')
            if not os.path.isdir(filedir):
                os.mkdir(filedir)
            filename = self._get_filename(extension='png')
            filepath = os.path.join(filedir, filename)

            plt.savefig(filepath)

        #
        plt.show()

    def write(self):
        data = self.data

        #
        filedir = os.path.join('.', 'txt')
        if not os.path.isdir(filedir):
            os.mkdir(filedir)
        filename = self._get_filename(extension='txt')
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

    def _get_filename(self, emulation: Emulation, extension: Literal['png', 'txt']):
        emulation = self.emulation

        device = emulation.config.device
        detector = emulation.config.detector
        info = emulation.config.info

        content = '; '.join([f'{item}' for item in [device.config.dispersion, detector.config.name, info] if item])  # FIXME: change device.config.dispersion to device.config.name  # noqa: E501
        return f'concentration_calibration ({content}).{extension}'
