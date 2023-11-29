
import os
from configparser import ConfigParser

from spectrumlab.emulation.detector.characteristic.aperture import RectangularApertureProfile
from spectrumlab.emulation.detector.linear_array_detector import Detector
from spectrumlab.emulation.device import Device

from spectrumlab.emulation.intensity import IntensityConfig


class Config:
    """Emulation experiment's config."""

    def _get_device(self, parser: ConfigParser) -> Device:
        kind = parser.get('device', 'kind')
        if kind == 'GRAND2':
            poly = parser.get('device', 'poly')
            if poly == 'I':
                return Device.GRAND2_I
            if poly == 'II':
                return Device.GRAND2_II
            raise ValueError(f'Device poly {poly} is not supported!')

        raise ValueError(f'Device kind {kind} is not supported!')

    def _get_detector(self, parser: ConfigParser) -> Detector:
        kind = parser.get('detector', 'kind')
        if kind == 'BLPP369M1':
            return Detector.BLPP369M1
        if kind == 'BLPP2000':
            return Detector.BLPP2000
        if kind == 'BLPP4000':
            return Detector.BLPP4000
        raise ValueError(f'Detector kind {kind} is not supported!')

    def __init__(self, filedir: str, filename: str) -> None:

        # ini parser
        parser = ConfigParser(inline_comment_prefixes='#')
        parser.read(os.path.join(filedir, filename))

        #
        self.poly = parser.get('device', 'kind')

        # --------        emulation config        --------
        self.device = self._get_device(parser)
        self.detector = self._get_detector(parser)

        self.base_level = float(parser.get('base spectrum', 'level'))
        self.base_n_frames = int(parser.get('base spectrum', 'n_frames'))

        self.n_numbers = int(parser.get('spectrum', 'n_numbers'))
        self.n_frames = int(parser.get('spectrum', 'n_frames'))

        self.line_width = float(parser.get('line', 'width')) / self.device.config.dispersion  # in micron
        self.line_ratio = float(parser.get('line', 'ratio'))

        self.apparatus_width = float(parser.get('apparatus', 'width')) # in micron
        self.apparatus_asymmetry = float(parser.get('apparatus', 'asymmetry'))
        self.apparatus_ratio = float(parser.get('apparatus', 'ratio'))

        self.aperture_profile = RectangularApertureProfile

        # --------        intensity config        --------
        self.intensity = IntensityConfig(
            method='linear',
            interval=3,
        )

        # --------        calibration curve config        --------
        self.n_probes = 19
        self.n_parallels = 5

        self.position = 9.8
        self.concentrations = tuple(reversed([10000 * (1/(2**(i))) for i in range(self.n_probes)]))

        # self.ref = pd.read_csv(
        #     os.path.join(filedir, 'data', self.device.name, f'calibration_curve.csv'),
        #     decimal=',',
        #     sep=';',
        #     encoding='utf-8',
        # )

        # --------        others        --------
        self.scattering_ratio = float(parser.get('others', 'scattering_ratio'))  # ratio of scattered radiation, in [0, 1)
        self.background_level = float(parser.get('others', 'background_level'))  # continuous background, in A
        self.concentration_ratio = 10**(float(parser.get('others', 'concentration_ratio')))  # concentration coefficient
