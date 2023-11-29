
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

# --------        spacial units        --------
Meter = NewType('Meter', float)
Inch = NewType('Inch', float)
Centi = NewType('Centi', float)  # centimetre
Milli = NewType('Milli', float)  # millimeter
Micro = NewType('Micro', float)  # micrometer
Nano = NewType('Nano', float)  # nanometer

Number = NewType('Number', float)
Wavelength = NewType('Wavelength', float)

# --------        intensity units        --------
Electron = NewType('Electron', float)
Percent = NewType('Percent', float)


# --------        absorbance units        --------
Absorbance = NewType('Absorbance', float)
