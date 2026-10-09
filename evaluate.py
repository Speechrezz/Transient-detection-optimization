import numpy as np
import numpy.typing as npt

def evaluate_transients(
        target: npt.NDArray[np.intp],
        detected: npt.NDArray[np.intp],
        sample_rate: float,
        tolerance_ms: float = 10,
        timing_weight: float = 0.1,
):
    tolerance = max(1, round(tolerance_ms * 0.001 * sample_rate))

    # Match each detection to at most one target.
    tp = 0
    total_delay = 0
    i = 0
    j = 0

    while i < len(target) and j < len(detected):
        delay = int(detected[j]) - int(target[i])

        if delay < 0:
            # Detection occurred before the target.
            j += 1

        elif delay > tolerance:
            # Target was missed.
            i += 1

        else:
            # Successful match.
            tp += 1
            total_delay += delay
            i += 1
            j += 1

    fp = len(detected) - tp
    fn = len(target) - tp

    # F1 score.
    denominator = 2 * tp + fp + fn

    if denominator == 0:
        f1 = 1.0
    else:
        f1 = 2.0 * tp / denominator

    # Timing penalty for matched detections.
    if tp > 0:
        mean_delay = total_delay / tp
        timing_penalty = mean_delay / tolerance
    else:
        timing_penalty = 0.0

    return (1.0 - f1) + timing_weight * timing_penalty