from typing import Sequence, Union

import numpy as np


class MissingType:
    """
    Sentinel type - indicates a missing value in a function call.
    """

    pass


MISSING: MissingType = MissingType()
DEPRECATED: MissingType = MissingType()
FloatSequence = Union[np.ndarray, Sequence[float]]
IntSequence = Union[np.ndarray, Sequence[int]]
