import numpy as np

from spectrumlab.grid import (
    Grid,
    InterpolationKind,
    integrate_grid,
    interpolate_grid,
)
from spectrumlab.peak.intensity import (
    AbstractIntensityCalculator,
    AmplitudeIntensityCalculator,
    ApproxIntensityCalculator,
    IntegralIntensityCalculator,
)
from spectrumlab.peak.units import R
from spectrumlab.types import Array, Number


class EmulatedMixin:
    pass


class WrappedAmplitudeIntensityCalculator(AmplitudeIntensityCalculator, EmulatedMixin):

    def calculate(
        self,
        grid: Grid,
        mask: Array[bool],
        position: Number,
    ) -> R:

        f = interpolate_grid(grid, kind=InterpolationKind.NEAREST)
        value = f(position)

        return value


class WrappedApproxIntensityCalculator(ApproxIntensityCalculator, EmulatedMixin):

    def calculate(
        self,
        grid: Grid,
        mask: Array[bool],
        position: Number,
    ) -> R:

        norm = np.dot(grid.y[~mask], self.shape(grid.x[~mask], position=position, intensity=1))
        value = np.dot(grid.y[~mask], grid.y[~mask]) / norm

        return value


class WrappedIntegralIntensityCalculator(IntegralIntensityCalculator, EmulatedMixin):

    def calculate(
        self,
        grid: Grid,
        mask: Array[bool],
        position: Number,
    ) -> R:

        value = integrate_grid(
            grid=grid,
            position=position,
            interval=self.interval,
            kind=self.kind,
        )

        return value


def patch_calculator(
    __calculator: AbstractIntensityCalculator,
) -> AbstractIntensityCalculator:  # TODO: rename

    if isinstance(__calculator, AmplitudeIntensityCalculator):
        instance = WrappedAmplitudeIntensityCalculator(
            verbose=__calculator.verbose,
        )
    if isinstance(__calculator, ApproxIntensityCalculator):
        instance = WrappedApproxIntensityCalculator(
            shape=__calculator.shape,
            delta=__calculator.delta,
            by_tail=__calculator.by_tail,
            verbose=__calculator.verbose,
            show=__calculator.show,
        )
    if isinstance(__calculator, IntegralIntensityCalculator):
        instance = WrappedIntegralIntensityCalculator(
            interval=__calculator.interval,
            kind=__calculator.kind,
            verbose=__calculator.verbose,
        )

    __calculator.calculate = instance.calculate
    return __calculator
