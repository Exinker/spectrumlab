from abc import ABC

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spectrumlab.types import Array, C, R


class KernelABC(ABC):

    def __init__(
        self,
        intensity: Array[R],
        concentration: Array[C],
        bounds: tuple[R, R],
    ) -> None:

        self.intensity = np.array(intensity, dtype=np.float64, copy=True)
        self.concentration = np.array(concentration, dtype=np.float64, copy=True)
        self.bounds = bounds

        if len(self.intensity) != len(self.concentration):
            raise ValueError('Intensity and concentration arrays must have the same length!')

        if np.any(self.intensity <= 0) or np.any(self.concentration <= 0):
            raise ValueError("Intensity and concentration arrays must contain strictly positive values!")

        # calculate bias
        lb, ub = self.bounds
        mask = (lb <= self.intensity) & (self.intensity <= ub)
        if mask.sum() < 2:
            raise ValueError('The calibration bounds must contain at least 2 data points!')

        self.bias = np.log10(self.intensity[mask]).mean() - np.log10(self.concentration[mask]).mean()

    def show(self):

        fig, ax = plt.subplots(figsize=(6, 4))

        x = np.linspace(0, 5, 101)
        plt.plot(
            x,
            np.array(list(map(self.kernel, x))),
            color='black', linestyle=':',
        )

        plt.xlabel(r'$R$')
        plt.ylabel(r'$\hat{R}$')
        plt.grid(color='grey', linestyle=':')
        plt.show()

    def __call__(self, __value: R) -> R:

        if __value <= self.bounds[1]:
            return __value

        return self.kernel(__value)
