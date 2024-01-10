"""
Data types for emulation experiment of .

Author: Vaschenko Pavel
 Email: vaschenko@vmk.ru
  Date: 2023.12.26
"""
import os
from configparser import ConfigParser
from dataclasses import dataclass, field

import pandas as pd

from spectrumlab.alias import Array, Frame
from spectrumlab.emulation.aperture import Aperture, RectangularApertureShape
from spectrumlab.emulation.apparatus import Apparatus, VoigtApparatusShape
from spectrumlab.emulation.detector import Detector
from spectrumlab.emulation.device import Device
from spectrumlab.emulation.intensity import IntensityConfig, IntegralIntensityConfig, InterpolationKind
from spectrumlab.emulation.line import Line, PVoigtLineShape


import warnings
warnings.filterwarnings('ignore')


@dataclass
class BaseEmittedExperimentConfig:

    # --------        emulation        --------
    device: Device
    detector: Detector

    n_numbers: int
    n_frames: int

    apparatus: Apparatus
    aperture: Aperture

    # --------        position        --------
    position: float

    # --------        intensity        --------
    intensity: IntensityConfig

    # --------        calibration curve        --------
    n_blanks: int
    n_probes: int
    n_parallels: int

    concentration_base: float = field(default=10_000)
    concentration_blank: float = field(default=0)
    concentration_ratio: float = field(default=1)

    ref: Array[float] | None = field(default=None)

    # --------        others        --------
    background_level: float = field(default=0)

    @property
    def concentrations(self) -> tuple[float]:
        return tuple(reversed([self.concentration_base * (1/(2**(i))) for i in range(self.n_probes)]))


class EmittedExperimentConfigNaive(BaseEmittedExperimentConfig):
    '''Emitted spectra (naive) experiment's config.'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # --------        fabric        --------
    @classmethod
    def from_ini(cls, filedir: str, filename: str, n_blanks: int | None = None, n_probes: int | None = None, n_parallels: int | None = None) -> 'EmittedExperimentConfigNaive':

        def _get_device(parser: ConfigParser) -> Device:
            kind = parser.get('device', 'kind')
            if kind == 'GRAND2':
                poly = parser.get('device', 'poly')
                if poly == 'I':
                    return Device.GRAND2_I
                if poly == 'II':
                    return Device.GRAND2_II
                raise ValueError(f'Device poly {poly} is not supported!')

            raise ValueError(f'Device kind {kind} is not supported!')

        def _get_detector(parser: ConfigParser) -> Detector:
            kind = parser.get('detector', 'kind')
            if kind == 'BLPP369M1':
                return Detector.BLPP369M1
            if kind == 'BLPP2000':
                return Detector.BLPP2000
            if kind == 'BLPP4000':
                return Detector.BLPP4000
            raise ValueError(f'Detector kind {kind} is not supported!')

        def _get_ref(parser: ConfigParser) -> Frame | None:
            device = _get_device(parser)

            filepath = os.path.join(filedir, 'data', device.name, f'concentration_calibration.csv')
            if os.path.isfile(filepath):
                return pd.read_csv(
                    filepath,
                    decimal=',',
                    sep=';',
                    encoding='utf-8',
                )
            else:
                return None

        # ini parser
        parser = ConfigParser(inline_comment_prefixes='#')
        parser.read(os.path.join(filedir, filename))

        #
        device = _get_device(parser)
        detector = _get_detector(parser)
        ref = _get_ref(parser)

        return EmittedExperimentConfigNaive(
            device=device,
            detector=detector,

            n_numbers=int(parser.get('spectrum', 'n_numbers')),
            n_frames=int(parser.get('spectrum', 'n_frames')),

            apparatus=Apparatus(
                detector=detector,
                shape=VoigtApparatusShape(
                    width=float(parser.get('apparatus', 'width')),
                    asymmetry=float(parser.get('apparatus', 'asymmetry')),
                    ratio=float(parser.get('apparatus', 'ratio')),
                ),
            ),
            aperture=Aperture(
                detector=detector,
                shape=RectangularApertureShape(),
            ),

            # --------        position        --------
            position=int(parser.get('spectrum', 'n_numbers'))/2,

            # --------        intensity        --------
            intensity=IntegralIntensityConfig(
                kind=InterpolationKind.LINEAR,
                interval=3,
            ),

            # --------        calibration curve        --------
            n_blanks=n_blanks or int(parser.get('calibration-curve', 'n_blanks')),
            n_probes=n_probes or int(parser.get('calibration-curve', 'n_probes')),
            n_parallels=n_parallels or int(parser.get('calibration-curve', 'n_parallels')),

            concentration_blank=float(parser.get('calibration-curve', 'concentration_blank')),
            concentration_base=float(parser.get('calibration-curve', 'concentration_base')),
            concentration_ratio=10**(float(parser.get('calibration-curve', 'concentration_ratio'))),
            ref=ref,

            # --------        others        --------
            background_level=float(parser.get('others', 'background_level')),
        )


class EmittedExperimentConfig(BaseEmittedExperimentConfig):
    '''Emitted spectra experiment's config.'''

    def __init__(self, *args, line: PVoigtLineShape, **kwargs):
        super().__init__(*args, **kwargs)

        self.line = line


class AbsorbedExperimentConfig(BaseEmittedExperimentConfig):
    '''Absorbed spectra experiment's config.'''

    def __init__(self, *args, base_level: float, base_n_frames: int, line: PVoigtLineShape, scattering_ratio: float, **kwargs):
        super().__init__(*args, **kwargs)

        self.base_level = base_level
        self.base_n_frames = base_n_frames

        self.line = line

        self.scattering_ratio = scattering_ratio

    # --------        fabric        --------
    @classmethod
    def from_ini(cls, filedir: str, filename: str, n_blanks: int | None = None, n_probes: int | None = None, n_parallels: int | None = None) -> 'EmittedExperimentConfigNaive':

        def _get_device(parser: ConfigParser) -> Device:
            kind = parser.get('device', 'kind')
            if kind == 'GRAND2':
                poly = parser.get('device', 'poly')
                if poly == 'I':
                    return Device.GRAND2_I
                if poly == 'II':
                    return Device.GRAND2_II
                raise ValueError(f'Device poly {poly} is not supported!')

            raise ValueError(f'Device kind {kind} is not supported!')

        def _get_detector(parser: ConfigParser) -> Detector:
            kind = parser.get('detector', 'kind')
            if kind == 'BLPP369M1':
                return Detector.BLPP369M1
            if kind == 'BLPP2000':
                return Detector.BLPP2000
            if kind == 'BLPP4000':
                return Detector.BLPP4000
            raise ValueError(f'Detector kind {kind} is not supported!')

        def _get_ref(parser: ConfigParser) -> Frame | None:
            device = _get_device(parser)

            filepath = os.path.join(filedir, 'data', device.name, f'concentration_calibration.csv')
            if os.path.isfile(filepath):
                return pd.read_csv(
                    filepath,
                    decimal=',',
                    sep=';',
                    encoding='utf-8',
                )
            else:
                return None

        # ini parser
        parser = ConfigParser(inline_comment_prefixes='#')
        parser.read(os.path.join(filedir, filename))

        #
        device = _get_device(parser)
        detector = _get_detector(parser)

        base_level = float(parser.get('base spectrum', 'level'))
        base_n_frames = int(parser.get('base spectrum', 'n_frames'))

        ref = _get_ref(parser)

        #
        return AbsorbedExperimentConfig(
            device=device,
            detector=detector,

            n_numbers=int(parser.get('spectrum', 'n_numbers')),
            n_frames=int(parser.get('spectrum', 'n_frames')),

            base_level=base_level,
            base_n_frames=base_n_frames,

            line = Line(
                shape=PVoigtLineShape(
                    width=float(parser.get('line', 'width')) / device.config.dispersion,  # in micron
                    asymmetry=0,
                    ratio=float(parser.get('line', 'ratio')),
                ),
            ),            
            apparatus=Apparatus(
                detector=detector,
                shape=VoigtApparatusShape(
                    width=float(parser.get('apparatus', 'width')),
                    asymmetry=float(parser.get('apparatus', 'asymmetry')),
                    ratio=float(parser.get('apparatus', 'ratio')),
                ),
            ),
            aperture=Aperture(
                detector=detector,
                shape=RectangularApertureShape(),
            ),

            # --------        position        --------
            position=int(parser.get('spectrum', 'n_numbers'))/2,

            # --------        intensity        --------
            intensity=IntegralIntensityConfig(
                kind=InterpolationKind.LINEAR,
                interval=3,
            ),

            # --------        calibration curve        --------
            n_blanks=n_blanks or int(parser.get('calibration-curve', 'n_blanks')),
            n_probes=n_probes or int(parser.get('calibration-curve', 'n_probes')),
            n_parallels=n_parallels or int(parser.get('calibration-curve', 'n_parallels')),

            concentration_blank=float(parser.get('calibration-curve', 'concentration_blank')),
            concentration_base=float(parser.get('calibration-curve', 'concentration_base')),
            concentration_ratio=10**(float(parser.get('calibration-curve', 'concentration_ratio'))),

            ref=ref,

            # --------        others        --------
            background_level=float(parser.get('others', 'background_level')),
            scattering_ratio=float(parser.get('others', 'scattering_ratio')),
        )
