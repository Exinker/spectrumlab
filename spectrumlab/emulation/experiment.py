import os
from configparser import ConfigParser
from dataclasses import dataclass, field

import pandas as pd

from spectrumlab.alias import Array, Frame, Micro
from spectrumlab.emulation.detector.characteristic.aperture import ApertureShape, RectangularApertureShape
from spectrumlab.emulation.detector.linear_array_detector import Detector
from spectrumlab.emulation.device import Device
from spectrumlab.emulation.intensity import IntensityConfig, IntegralIntensityConfig, InterpolationKind
from spectrumlab.emulation.line import LineShape, VoigtLineShape


import warnings
warnings.filterwarnings('ignore')


@dataclass
class BaseEmittedExperimentConfig:

    # --------        emulation config        --------
    device: Device
    detector: Detector

    n_numbers: int
    n_frames: int

    apparatus_shape: VoigtLineShape
    aperture_shape: ApertureShape

    # --------        intensity config        --------
    intensity: IntensityConfig

    # --------        calibration curve config        --------
    n_probes: int
    n_parallels: int

    position: float

    ref: Array[float] | None = field(default=None)

    # --------        others        --------
    background_level: float = field(default=0)
    concentration_blank: float = field(default=0)
    concentration_base: float = field(default=10_000)
    concentration_ratio: float = field(default=1)

    @property
    def concentrations(self) -> tuple[float]:
        return tuple(reversed([self.concentration_base * (1/(2**(i))) for i in range(self.n_probes)]))


class EmittedExperimentConfigNaive(BaseEmittedExperimentConfig):
    '''Experiment's config (naive).'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_ini(cls, filedir: str, filename: str, n_probes: int = 18, n_parallels: int = 5) -> 'EmittedExperimentConfigNaive':

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

            filepath = os.path.join(filedir, 'data', device.name, f'calibration_curve.csv')
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

        #
        return EmittedExperimentConfigNaive(
            device=device,
            detector=detector,

            n_numbers=int(parser.get('spectrum', 'n_numbers')),
            n_frames=int(parser.get('spectrum', 'n_frames')),

            apparatus_shape=VoigtLineShape(
                width=float(parser.get('apparatus', 'width')),
                asymmetry=float(parser.get('apparatus', 'asymmetry')),
                ratio=float(parser.get('apparatus', 'ratio')),
            ),
            aperture_shape=RectangularApertureShape,

            # --------        position config        --------
            position=int(parser.get('spectrum', 'n_numbers'))/2,

            # --------        intensity config        --------
            intensity=IntegralIntensityConfig(
                kind=InterpolationKind.LINEAR,
                interval=3,
            ),

            # --------        calibration curve config        --------
            n_probes=n_probes,
            n_parallels=n_parallels,

            ref=ref,

            # --------        others        --------
            background_level=float(parser.get('others', 'background_level')),
            concentration_blank=float(parser.get('others', 'concentration_blank')),
            concentration_base=float(parser.get('others', 'concentration_base')),
            concentration_ratio=10**(float(parser.get('others', 'concentration_ratio'))),
        )


class EmittedExperimentConfig(BaseEmittedExperimentConfig):
    '''Experiment's config.'''

    def __init__(self, *args, line_shape: VoigtLineShape, **kwargs):
        super().__init__(*args, **kwargs)

        # --------        emulation config        --------
        self.line_shape = line_shape


class AbsorbedExperimentConfig(BaseEmittedExperimentConfig):
    '''Experiment's config.'''

    def __init__(self, *args, base_level: float, base_n_frames: int, line_shape: VoigtLineShape, scattering_ratio: float, **kwargs):
        super().__init__(*args, **kwargs)

        # --------        emulation config        --------
        self.base_level = base_level
        self.base_n_frames = base_n_frames

        self.line_shape = line_shape

        self.scattering_ratio = scattering_ratio

    @classmethod
    def from_ini(cls, filedir: str, filename: str, n_probes: int = 18, n_parallels: int = 5) -> 'EmittedExperimentConfigNaive':

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

            filepath = os.path.join(filedir, 'data', device.name, f'calibration_curve.csv')
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

            line_shape = VoigtLineShape(
                width=float(parser.get('line', 'width')),
                asymmetry=0,
                ratio=float(parser.get('line', 'ratio')),
            ),
            apparatus_shape=VoigtLineShape(
                width=float(parser.get('apparatus', 'width')),
                asymmetry=float(parser.get('apparatus', 'asymmetry')),
                ratio=float(parser.get('apparatus', 'ratio')),
            ),
            aperture_shape=RectangularApertureShape,

            # --------        position config        --------
            position=int(parser.get('spectrum', 'n_numbers'))/2,

            # --------        intensity config        --------
            intensity=IntegralIntensityConfig(
                kind=InterpolationKind.LINEAR,
                interval=3,
            ),

            # --------        calibration curve config        --------
            n_probes=n_probes,
            n_parallels=n_parallels,

            ref=ref,

            # --------        others        --------
            background_level=float(parser.get('others', 'background_level')),
            concentration_blank=float(parser.get('others', 'concentration_blank')),
            concentration_base=float(parser.get('others', 'concentration_base')),
            concentration_ratio=10**(float(parser.get('others', 'concentration_ratio'))),
            scattering_ratio=float(parser.get('others', 'scattering_ratio')),
        )
