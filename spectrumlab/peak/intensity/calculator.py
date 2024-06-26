import abc
from typing import TYPE_CHECKING

from spectrumlab.peak.units import U
from spectrumlab.picture.color import Color

if TYPE_CHECKING:
    from spectrumlab.peak.analyte_peak import AnalytePeak


class AbstractIntensityCalculator(abc.ABC):

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    @property
    @abc.abstractmethod
    def color(self) -> Color:
        raise NotImplementedError

    @abc.abstractmethod
    def calculate(self, peak: 'AnalytePeak') -> U:
        raise NotImplementedError
