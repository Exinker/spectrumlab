from decimal import Decimal

import numpy as np


PREFIX = {
    -12: 'p',
    -9: 'n',
    -6: r'\mu',
    -3: 'm',
    0: '',
    3: 'k',
    6: 'M',
    9: 'G',
}


def format_label(value: float, units: str | None = None, n_digits: int = 2, prefix: bool = False) -> str:
    """Format value and units to label.

    Examples:
    - 1e-5 -> '10.00 ⋅ 10^{-6}'
    - 1e-5, 'A' -> '10.00 ⋅ 10^{-6}, A'
    - 1e-5, 'A' -> '10.00, μA'

    FIXME: 1e-06, 'A' -> $999.99, nA$
    """
    assert n_digits >= 0, '`n_digits` have to be greater 0'

    sign, digits, exponent = Decimal(value).as_tuple()

    e = len(digits) + exponent - 1
    e_rounded = int(np.sign(e) * (3*np.floor(np.abs(e) / 3)))
    n = e - e_rounded + 1

    result = []
    if n > 0:
        result.append(
            ''.join(map(str, digits[:n])) + '.' * (n_digits != 0) + ''.join(map(str, digits[n:n + n_digits])),
        )
    else:
        m = -n
        result.append(
            '0' + '.' + '0'*(m) + ''.join(map(str, digits[m:n_digits - m + 1])),
        )

    if units:
        if prefix:
            result.append(
                r', {{{}}}{{{}}}'.format(PREFIX.get(e_rounded, '???'), units),
            )
        else:
            result.append(
                fr' \cdot 10^{{{e_rounded}}}' * (e_rounded != 0) + f', {units}',
            )

    else:
        result.append(
            fr' \cdot 10^{{{e_rounded}}}' * (e_rounded != 0),
        )

    return r'{}'.format(
        ''.join(result),
    )
