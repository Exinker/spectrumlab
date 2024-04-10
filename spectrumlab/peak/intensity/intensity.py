from abc import ABC, abstractmethod, abstractproperty
from typing import TYPE_CHECKING

from spectrumlab.typing import Percent

if TYPE_CHECKING:
    from spectrumlab.peak.analyte_peak import AnalytePeak


class AbstractIntensityCalculator(ABC):

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    @abstractproperty
    def color(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def calculate(self, peak: 'AnalytePeak') -> Percent:
        raise NotImplementedError
