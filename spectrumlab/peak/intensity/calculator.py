from abc import ABC, abstractmethod, abstractproperty
from typing import TYPE_CHECKING

from spectrumlab.peak.units import U
from spectrumlab.picture.color import Color

if TYPE_CHECKING:
    from spectrumlab.peak.analyte_peak import AnalytePeak


class AbstractIntensityCalculator(ABC):

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    @abstractproperty
    def color(self) -> Color:
        raise NotImplementedError

    @abstractmethod
    def calculate(self, peak: 'AnalytePeak') -> U:
        raise NotImplementedError
