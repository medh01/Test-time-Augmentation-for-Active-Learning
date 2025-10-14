import numpy as np

def sigmoid_rampup(current: float, rampup_length: int, max_value: float = 1.0) -> float:
    """Smoothly ramps a value from 0 to `max_value` using a Gaussian curve."""
    if rampup_length <= 0:
        return max_value
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return max_value * np.exp(-5.0 * phase * phase)
