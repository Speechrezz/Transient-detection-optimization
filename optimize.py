import numpy as np
import numpy.typing as npt
from scipy.optimize import differential_evolution

from detector import TransientParams, TransientDetector
import sys
sys.path.append("build/Release")
from detector_cpp import TransientParamsCpp, TransientDetectorCpp

from evaluate import evaluate_transients
from fileio import to_transient_positions, load_dataset

# Dataset entries contain:
# - audio: mono float32 samples
# - sample_rate: integer
# - onsets: manually annotated sample positions

'''
TODO:
    - Create evaluation function
    - Create labeled dataset
    - Optimize
'''

counter = 0

def run_detector(
    audio: npt.NDArray[np.float32],
    sample_rate: float,
    fast_decay: float,
    slow_decay: float,
    slow_attack: float,
    trigger_thresh: float,
    rearm_ratio: float
):
    # print("run_detector - begin")
    params = TransientParamsCpp(
        sample_rate=sample_rate,
        fast_decay=fast_decay,
        slow_decay=slow_decay,
        slow_attack=slow_attack,
        trigger_thresh=trigger_thresh,
        rearm_ratio=rearm_ratio,
    )
    detector = TransientDetectorCpp()
    detector.prepare(params)
    transients = detector.process(audio)
    # print("run_detector - end")
    return to_transient_positions(transients)


def objective(params, dataset):
    fast_decay, slow_attack, slow_decay, trigger, ratio = params

    rearm = trigger * ratio

    scores = []

    for example in dataset:
        detected = run_detector(
            example["audio"],
            example["sample_rate"],
            fast_decay,
            slow_attack,
            slow_decay,
            trigger,
            rearm,
        )

        loss = evaluate_transients(
            example["onsets"],
            detected,
            sample_rate=example["sample_rate"],
        )

        scores.append(loss)

    return float(np.mean(scores))

bounds = [
    (1.0, 100.0),    # Fast release (ms)
    (3.0, 400.0),    # Slow attack (ms)
    (20.0, 1000.0),  # Slow release (ms)
    (0.1, 2.0),      # Trigger threshold
    (0.1, 0.8),      # Rearm/trigger ratio
]

def main():
    training_dataset = load_dataset("./dataset")
    print(training_dataset)

    result = differential_evolution(
        lambda p: objective(p, training_dataset),
        bounds=bounds,
        maxiter=150,
        popsize=20,
        tol=0.0001,
        seed=1,
        polish=False,
        disp=True,
    )

    print("Best parameters:")
    print(f"\tfast_decay={result.x[0]},")
    print(f"\tslow_attack={result.x[1]},")
    print(f"\tslow_decay={result.x[2]},")
    print(f"\ttrigger_thresh={result.x[3]},")
    print(f"\trearm_ratio={result.x[4]},")

    print("Training loss:", result.fun)

if __name__ == "__main__":
    main()