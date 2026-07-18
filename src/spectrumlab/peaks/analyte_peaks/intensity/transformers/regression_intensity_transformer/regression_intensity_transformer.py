import matplotlib.pyplot as plt

from spectrumlab.peaks.analyte_peaks.intensity.transformers.base_intensity_transformer import (
    IntensityTransformerABC,
)
from spectrumlab.peaks.analyte_peaks.intensity.transformers.regression_intensity_transformer.kernels.base_kernel import (
    KernelABC,
)
from spectrumlab.types import R


class RegressionIntensityTransformer(IntensityTransformerABC):

    def __init__(self, kernel: KernelABC) -> None:

        self.kernel = kernel

    def show(self) -> None:
        lb, ub = self.kernel.bounds
        intensity = self.kernel.intensity
        concentration = self.kernel.concentration

        mask = (lb <= intensity) & (intensity <= ub)
        unicorn = self.kernel.cast(concentration)

        fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

        plt.sca(ax_left)
        plt.plot(
            concentration,
            intensity,
            color='grey', linestyle='none', marker='s', markersize=4,
        )
        plt.plot(
            concentration[mask],
            intensity[mask],
            color='black', linestyle='none', marker='s', markersize=4,
        )
        plt.plot(
            concentration,
            unicorn,
            color='grey', linestyle=':',
        )
        plt.axhspan(
            lb, ub,
            alpha=.125, color='red',
        )
        plt.xlabel('$C$')
        plt.ylabel('$R$')
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(color='grey', linestyle=':')

        plt.sca(ax_right)
        x = concentration
        y = 100 * (intensity - unicorn) / unicorn
        plt.plot(
            x, y,
            color='grey', linestyle=':',
        )
        x = concentration
        y = 100 * (intensity - unicorn) / unicorn
        plt.plot(
            x, y,
            color='grey', linestyle='none', marker='s', markersize=4,
        )
        plt.plot(
            x[mask], y[mask],
            color='black', linestyle='none', marker='s', markersize=4,
        )
        plt.xlabel(r'$C$')
        plt.ylabel(r'$(R - \hat{R})/\hat{R}$, %')
        plt.xscale('log')
        plt.grid(color='grey', linestyle=':')

        plt.show()

    def predict(self, __value: R) -> R:

        return self.kernel(__value)
