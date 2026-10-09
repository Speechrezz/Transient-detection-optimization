import numpy as np
import numpy.typing as npt
from dataclasses import dataclass

class ExponentialSmoother:
    def __init__(self, sample_rate: float, attack_ms: float, decay_ms: float):
        exp_factor = np.float32(-2.0 * np.pi * 1000.0 / sample_rate)

        if attack_ms < 0.01:
            self.attack_coef = np.float32(0.0)
        else:
            self.attack_coef = np.exp(exp_factor / np.float32(attack_ms))

        if decay_ms < 0.01:
            self.decay_coef = np.float32(0.0)
        else:
            self.decay_coef = np.exp(exp_factor / np.float32(decay_ms))

        print(self.attack_coef, self.decay_coef)
        self.current = np.float32(0.0)

    def processSample(self, sample: np.float32):
        is_within_threshold = np.abs(self.current - sample) < 1e-4

        if (is_within_threshold):
            self.current = sample
        else:
            time_coef = self.attack_coef if sample > self.current else self.decay_coef
            self.current = sample + time_coef * (self.current - sample)

        return self.current

    def processArray(self, array: npt.NDArray[np.float32]):
        output = np.zeros_like(array)

        for i, sample in enumerate(array):
            output[i] = self.processSample(sample)

        return output


@dataclass
class TransientParams:
    sample_rate: float
    fast_decay: float
    slow_decay: float
    slow_attack: float
    trigger_thresh: float
    rearm_ratio: float
    noise_floor: float = 0.0005 # -66 dB

class TransientDetector:
    def __init__(self, params: TransientParams):
        self.params = params

        self.thresh_upper = np.float32(params.trigger_thresh)
        self.thresh_lower = np.float32(params.rearm_ratio * params.trigger_thresh)
        self.noise_floor = np.float32(params.noise_floor)

        self.env_fast = ExponentialSmoother(params.sample_rate, 0.0, params.fast_decay)
        self.env_slow = ExponentialSmoother(params.sample_rate, params.slow_attack, params.slow_decay)

        self.is_armed = True

    def process(self, array: npt.NDArray[np.float32]):
        output = np.zeros_like(array)

        buffer_fast = self.env_fast.processArray(array)
        buffer_slow = self.env_slow.processArray(array)

        for i in range(len(array)):
            sample_fast = buffer_fast[i]
            sample_slow = buffer_slow[i]

            strength = np.maximum(np.float32(0.0), (sample_fast - sample_slow) / (sample_slow + self.noise_floor))

            is_transient = np.float32(0.0)

            if self.is_armed and strength > self.thresh_upper:
                self.is_armed = False
                is_transient = np.float32(1.0)
            elif not self.is_armed and strength < self.thresh_lower:
                self.is_armed = True

            output[i] = is_transient

        return output
