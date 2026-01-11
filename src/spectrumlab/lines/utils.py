import numpy as np

from spectrumlab.elements import Element, PeriodicTable
from spectrumlab.lines import Line
from spectrumlab.types import Kelvin, PicoMeter, Second


LIGHT_SPEED = 299_792_458  # light speed [m/s]
SIGMA = 2*1e-18  # collisional cross-section [m^2]
TAU = 1e-8  # level lifetime [s]
PERIODIC_TABLE = PeriodicTable()


def calculate_width_natural(line: Line, tau: Second = TAU) -> PicoMeter:
    """Calculate natural broadening."""
    wavelength = 1e-9*line.wavelength

    return 1e+12 * wavelength**2 / (2*np.pi*LIGHT_SPEED*tau)


def calculate_width_doppler(line: Line, temperature: Kelvin) -> PicoMeter:
    """Calculate doppler broadening."""
    wavelength = 1e-9*line.wavelength
    atomic_weight = PERIODIC_TABLE[line.symbol].atomic_weight

    return 1e+12 * 7.16 * 1e-7 * wavelength * np.sqrt(temperature/atomic_weight)


def calculate_width_collision(line: Line, buffer: Element, temperature: Kelvin, sigma: float = SIGMA) -> PicoMeter:
    """Calculate collision broadening."""
    wavelength = 1e-9*line.wavelength
    element = PERIODIC_TABLE[line.symbol]

    return 1.13*1e+33 * wavelength**2 * sigma * np.sqrt((1/element.atomic_weight + 1/buffer.atomic_weight)/temperature)
