"""
Data types for emulation experiment of .

Author: Vaschenko Pavel
 Email: vaschenko@vmk.ru
  Date: 2023.12.26
"""
import os
import warnings
from configparser import ConfigParser

import pandas as pd

from spectrumlab.emulation.aperture import Aperture, RectangularApertureShape
from spectrumlab.emulation.apparatus import Apparatus, VoigtApparatusShape
from spectrumlab.emulation.emulation.experiment import AbstractExperimentConfig, _parse_detector, _parse_device
from spectrumlab.emulation.intensity import IntegralIntensityCalculator
from spectrumlab.emulation.line import Line, PVoigtLineShape
from spectrumlab.grid import InterpolationKind
from spectrumlab.types import DirPath, FilePath, Frame


warnings.filterwarnings('ignore')


class EmittedExperimentConfigNaive(AbstractExperimentConfig):
    """Emitted spectra (naive) experiment's config."""

    def __init__(
            self,
            *args,
            n_blanks: int,
            n_probes: int,
            n_parallels: int,
            concentration_base: float = 10_000,
            concentration_blank: float = 0,
            ref: Frame | None = None,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.n_blanks = n_blanks
        self.n_probes = n_probes
        self.n_parallels = n_parallels
        self.concentration_base = concentration_base
        self.concentration_blank = concentration_blank
        self.ref = ref

    # --------        fabric        --------
    @classmethod
    def from_ini(cls, filedir: DirPath, filename: str, n_blanks: int | None = None, n_probes: int | None = None, n_parallels: int | None = None) -> 'EmittedExperimentConfigNaive':

        # ini parser
        parser = ConfigParser(inline_comment_prefixes='#')
        parser.read(os.path.join(filedir, filename))

        assert parser.get('emulation', 'kind') == 'emission'

        #
        device = _parse_device(parser)
        detector = _parse_detector(parser)

        filepath = os.path.join(filedir, 'concentration_calibration.csv')
        ref = _load_ref(filepath)

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

            n_blanks=n_blanks or int(parser.get('concentration-calibration', 'n_blanks')),
            n_probes=n_probes or int(parser.get('concentration-calibration', 'n_probes')),
            n_parallels=n_parallels or int(parser.get('concentration-calibration', 'n_parallels')),
            concentration_blank=float(parser.get('concentration-calibration', 'concentration_blank')),
            concentration_base=float(parser.get('concentration-calibration', 'concentration_base')),
            ref=ref,

            n_numbers=int(parser.get('spectrum', 'n_numbers')),
            n_frames=int(parser.get('spectrum', 'n_frames')),

            concentration_ratio=10**(float(parser.get('others', 'concentration_ratio'))),
            background_level=float(parser.get('others', 'background_level')),
        )


class EmittedExperimentConfig(AbstractExperimentConfig):
    """Emitted spectra experiment's config."""

    def __init__(
            self,
            *args,
            line: PVoigtLineShape,
            n_blanks: int,
            n_probes: int,
            n_parallels: int,
            concentration_base: float = 10_000,
            concentration_blank: float = 0,
            ref: Frame | None = None,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.line = line
        self.n_blanks = n_blanks
        self.n_probes = n_probes
        self.n_parallels = n_parallels
        self.concentration_base = concentration_base
        self.concentration_blank = concentration_blank
        self.ref = ref

    # --------        fabric        --------
    @classmethod
    def from_ini(cls, filedir: DirPath, filename: str, n_blanks: int | None = None, n_probes: int | None = None, n_parallels: int | None = None) -> 'EmittedExperimentConfigNaive':

        # ini parser
        parser = ConfigParser(inline_comment_prefixes='#')
        parser.read(os.path.join(filedir, filename))

        assert parser.get('emulation', 'kind') == 'emission'

        filepath = os.path.join(filedir, 'concentration_calibration.csv')
        if os.path.exists(filepath):
            ref = _load_ref(filepath)
        else:
            ref = None

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
                interval=float(parser.get('concentration-calibration', 'interval')),
            ),

            n_blanks=n_blanks or int(parser.get('concentration-calibration', 'n_blanks')),
            n_probes=n_probes or int(parser.get('concentration-calibration', 'n_probes')),
            n_parallels=n_parallels or int(parser.get('concentration-calibration', 'n_parallels')),
            concentration_blank=float(parser.get('concentration-calibration', 'concentration_blank')),
            concentration_base=float(parser.get('concentration-calibration', 'concentration_base')),
            ref=ref,

            n_numbers=int(parser.get('spectrum', 'n_numbers')),
            n_frames=int(parser.get('spectrum', 'n_frames')),

            concentration_ratio=10**(float(parser.get('others', 'concentration_ratio'))),
            background_level=float(parser.get('others', 'background_level')),
        )


class AbsorbedExperimentConfig(AbstractExperimentConfig):
    """Absorbed spectra experiment's config."""

    def __init__(
            self,
            *args,
            line: PVoigtLineShape,
            n_blanks: int,
            n_probes: int,
            n_parallels: int,
            concentration_base: float = 10_000,
            concentration_blank: float = 0,
            ref: Frame | None = None,
            base_level: float,
            base_n_frames: int,
            scattering_ratio: float,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.line = line
        self.n_blanks = n_blanks
        self.n_probes = n_probes
        self.n_parallels = n_parallels
        self.concentration_base = concentration_base
        self.concentration_blank = concentration_blank
        self.ref = ref
        self.base_level = base_level
        self.base_n_frames = base_n_frames
        self.scattering_ratio = scattering_ratio

    # --------        fabric        --------
    @classmethod
    def from_ini(cls, filedir: DirPath, filename: str, n_blanks: int | None = None, n_probes: int | None = None, n_parallels: int | None = None) -> 'EmittedExperimentConfigNaive':

        # ini parser
        parser = ConfigParser(inline_comment_prefixes='#')
        parser.read(os.path.join(filedir, filename))

        assert parser.get('emulation', 'kind') == 'absorption'

        #
        device = _parse_device(parser)
        detector = _parse_detector(parser)

        base_level = float(parser.get('base spectrum', 'level'))
        base_n_frames = int(parser.get('base spectrum', 'n_frames'))

        filepath = os.path.join(filedir, 'concentration_calibration.csv')
        if os.path.exists(filepath):
            ref = _load_ref(filepath)
        else:
            ref = None

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

            n_blanks=n_blanks or int(parser.get('concentration-calibration', 'n_blanks')),
            n_probes=n_probes or int(parser.get('concentration-calibration', 'n_probes')),
            n_parallels=n_parallels or int(parser.get('concentration-calibration', 'n_parallels')),
            concentration_blank=float(parser.get('concentration-calibration', 'concentration_blank')),
            concentration_base=float(parser.get('concentration-calibration', 'concentration_base')),
            ref=ref,

            n_numbers=int(parser.get('spectrum', 'n_numbers')),
            n_frames=int(parser.get('spectrum', 'n_frames')),

            base_level=base_level,
            base_n_frames=base_n_frames,
            concentration_ratio=10**(float(parser.get('others', 'concentration_ratio'))),
            background_level=float(parser.get('others', 'background_level')),
            scattering_ratio=float(parser.get('others', 'scattering_ratio')),
        )


def _load_ref(filepath: FilePath) -> Frame | None:
    """Get reference concentration calibration data."""

    if os.path.isfile(filepath):
        return pd.read_csv(
            filepath,
            decimal=',',
            sep=';',
            encoding='utf-8',
        )

    return None
