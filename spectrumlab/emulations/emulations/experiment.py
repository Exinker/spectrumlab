"""
Data types for emulation emission and absorption experiments.

Author: Vaschenko Pavel
 Email: vaschenko@vmk.ru
  Date: 2023.12.26
"""
import os
import warnings
from configparser import ConfigParser
from dataclasses import dataclass, field

from spectrumlab.emulations.apertures import Aperture, RectangularApertureShape
from spectrumlab.emulations.apparatus import Apparatus, VoigtApparatusShape
from spectrumlab.emulations.detectors import Detector
from spectrumlab.emulations.devices import Device
from spectrumlab.emulations.intensity import AbstractIntensityCalculator, IntegralIntensityCalculator
from spectrumlab.emulations.lines import Line, PVoigtLineShape
from spectrumlab.grid import InterpolationKind
from spectrumlab.types import DirPath


warnings.filterwarnings('ignore')


@dataclass
class AbstractExperimentConfig:
    device: Device
    detector: Detector
    apparatus: Apparatus
    aperture: Aperture

    position: float
    intensity_calculator: AbstractIntensityCalculator

    n_numbers: int
    n_frames: int

    concentration_ratio: float = 0
    background_level: float = field(default=0)

    @property
    def concentrations(self) -> tuple[float]:
        return tuple(reversed([self.concentration_base * (1/(2**(i))) for i in range(self.n_probes)]))


class EmittedExperimentConfigNaive(AbstractExperimentConfig):
    """Emitted spectra (naive) experiment's config."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # --------        fabric        --------
    @classmethod
    def from_ini(cls, filedir: DirPath, filename: str) -> 'EmittedExperimentConfigNaive':

        # ini parser
        parser = ConfigParser(inline_comment_prefixes='#')
        parser.read(os.path.join(filedir, filename))

        assert parser.get('emulation', 'kind') == 'emission'

        #
        device = _parse_device(parser)
        detector = _parse_detector(parser)

        return EmittedExperimentConfigNaive(
            device=device,
            detector=detector,
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

            position=int(parser.get('spectrum', 'n_numbers'))/2,
            intensity_calculator=IntegralIntensityCalculator(
                kind=InterpolationKind.LINEAR,
                interval=3,
            ),

            n_numbers=int(parser.get('spectrum', 'n_numbers')),
            n_frames=int(parser.get('spectrum', 'n_frames')),

            concentration_ratio=10**(float(parser.get('others', 'concentration_ratio'))),
            background_level=float(parser.get('others', 'background_level')),
        )


class EmittedExperimentConfig(AbstractExperimentConfig):
    """Emitted spectra experiment's config."""

    def __init__(self, *args, line: PVoigtLineShape, **kwargs):
        super().__init__(*args, **kwargs)

        self.line = line

    # --------        fabric        --------
    @classmethod
    def from_ini(cls, filedir: DirPath, filename: str) -> 'EmittedExperimentConfigNaive':

        # ini parser
        parser = ConfigParser(inline_comment_prefixes='#')
        parser.read(os.path.join(filedir, filename))

        assert parser.get('emulation', 'kind') == 'emission'

        #
        device = _parse_device(parser)
        detector = _parse_detector(parser)

        return EmittedExperimentConfig(
            device=device,
            detector=detector,
            line=PVoigtLineShape(
                width=float(parser.get('line', 'width')) / device.config.dispersion,  # in micron
                asymmetry=float(parser.get('line', 'asymmetry')),
                ratio=float(parser.get('line', 'ratio')),
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

            position=int(parser.get('spectrum', 'n_numbers'))/2,
            intensity_calculator=IntegralIntensityCalculator(
                kind=InterpolationKind.LINEAR,
                interval=3,
            ),

            n_numbers=int(parser.get('spectrum', 'n_numbers')),
            n_frames=int(parser.get('spectrum', 'n_frames')),

            concentration_ratio=10**(float(parser.get('others', 'concentration_ratio'))),
            background_level=float(parser.get('others', 'background_level')),
        )


class AbsorbedExperimentConfig(AbstractExperimentConfig):
    """Absorbed spectra experiment's config."""

    def __init__(self, *args, line: PVoigtLineShape, base_level: float, base_n_frames: int, scattering_ratio: float, **kwargs):
        super().__init__(*args, **kwargs)

        self.base_level = base_level
        self.base_n_frames = base_n_frames

        self.line = line

        self.scattering_ratio = scattering_ratio

    # --------        fabric        --------
    @classmethod
    def from_ini(cls, filedir: DirPath, filename: str) -> 'EmittedExperimentConfigNaive':

        # ini parser
        parser = ConfigParser(inline_comment_prefixes='#')
        parser.read(os.path.join(filedir, filename))

        assert parser.get('emulation', 'kind') == 'absorption'

        #
        device = _parse_device(parser)
        detector = _parse_detector(parser)

        base_level = float(parser.get('base spectrum', 'level'))
        base_n_frames = int(parser.get('base spectrum', 'n_frames'))

        #
        return AbsorbedExperimentConfig(
            device=device,
            detector=detector,
            line=Line(
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

            position=int(parser.get('spectrum', 'n_numbers'))/2,
            intensity_calculator=IntegralIntensityCalculator(
                kind=InterpolationKind.LINEAR,
                interval=3,
            ),

            n_numbers=int(parser.get('spectrum', 'n_numbers')),
            n_frames=int(parser.get('spectrum', 'n_frames')),

            base_level=base_level,
            base_n_frames=base_n_frames,
            concentration_ratio=10**(float(parser.get('others', 'concentration_ratio'))),
            background_level=float(parser.get('others', 'background_level')),
            scattering_ratio=float(parser.get('others', 'scattering_ratio')),
        )


# --------        private        --------
def _parse_device(parser: ConfigParser) -> Device:
    """Get device type."""
    kind = parser.get('device', 'kind')

    if kind == 'COLIBRI2':
        return Device.COLIBRI2
    if kind == 'GRAND2':
        poly = parser.get('device', 'poly')
        if poly == 'I':
            return Device.GRAND2_I
        if poly == 'II':
            return Device.GRAND2_II
        raise ValueError(f'Device poly {poly} is not supported!')
    if kind == 'custom':
        device = Device.dynamic
        device.config.dispersion = float(parser.get('device', 'dispersion'))
        return device

    raise ValueError(f'Device kind {kind} is not supported!')


def _parse_detector(parser: ConfigParser) -> Detector:
    """Get detector type."""
    kind = parser.get('detector', 'kind')

    if kind == 'BLPP369M1':
        return Detector.BLPP369M1
    if kind == 'BLPP2000':
        return Detector.BLPP2000
    if kind == 'BLPP4000':
        return Detector.BLPP4000

    raise ValueError(f'Detector kind {kind} is not supported!')
