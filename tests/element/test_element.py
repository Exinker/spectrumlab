import pytest

from spectrumlab.element import PeriodicTable
from spectrumlab.types import Symbol


@pytest.mark.parametrize(
    ['symbol', 'atomic_number'],
    [
        ('H', 1),
        ('C', 6),
    ],
)
def test_periodic_table_index(symbol: Symbol, atomic_number: int):
    table = PeriodicTable()

    assert symbol == table[symbol].symbol
    assert atomic_number == table[atomic_number].atomic_number
