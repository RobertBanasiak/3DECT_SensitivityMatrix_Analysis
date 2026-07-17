from __future__ import annotations

import numpy as np

EPS = np.finfo(float).eps
EXPECTED_ELECTRODES = 32
ELECTRODES_PER_PLANE = 8
EXPECTED_CHANNELS = EXPECTED_ELECTRODES * (EXPECTED_ELECTRODES - 1) // 2
