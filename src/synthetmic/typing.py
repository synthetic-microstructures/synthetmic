from typing import Sequence

import numpy as np
import numpy.typing as npt

IntArray = npt.NDArray[np.integer]
FloatArray = npt.NDArray[np.floating]
StrArray = npt.NDArray[np.str_]
NumericArray = npt.NDArray[np.integer] | npt.NDArray[np.floating]

FloatSequence = Sequence[float]
IntSequence = Sequence[int]
BoolSequence = Sequence[bool]
