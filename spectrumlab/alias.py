
from enum import Enum
from typing import TypeAlias, NewType

import pandas as pd
from numpy.typing import NDArray


# --------        types        --------
Array: TypeAlias = NDArray

Index: TypeAlias = pd.Index | pd.MultiIndex
Series: TypeAlias = pd.Series
Frame: TypeAlias = pd.DataFrame


# --------        dataset        --------
class Dataset(Enum):
    train = 'train'
    valid = 'valid'
    test = 'test'


# --------        temperature units        --------
Kelvin = NewType('Kelvin', float)
Celsius = NewType('Celsius', float)


# --------        time units        --------
Second = NewType('Second', float)
MilliSecond = NewType('MilliSecond', float)
MicroSecond = NewType('MicroSecond', int)


# --------        spacial units        --------
Inch = NewType('Inch', float)

Meter = NewType('Meter', float)
CentiMeter = NewType('CentiMeter', float)  # centimetre
MilliMeter = NewType('MilliMeter', float)  # millimeter
MicroMeter = NewType('MicroMeter', float)  # micrometer
NanoMeter = NewType('NanoMeter', float)  # nanometer
PicoMeter = NewType('Pico', float)  # picometer

Number = NewType('Number', float)


# --------        value units        --------
Absorbance = NewType('Absorbance', float)
Electron = NewType('Electron', float)
Percent = NewType('Percent', float)


# --------        other units        --------
Symbol = NewType('Symbol', str)
