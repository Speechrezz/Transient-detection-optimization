import numpy as np
from scipy.optimize import differential_evolution

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

def objective(params, dataset):
    fast_release, slow_attack, slow_release, trigger, ratio = params

    rearm = trigger * ratio

    scores = []

    for example in dataset:
        detected = run_cpp_detector(
            example["audio"],
            example["sample_rate"],
            fast_release,
            slow_attack,
            slow_release,
            trigger,
            rearm,
        )

        loss = evaluate_onsets(
            detected,
            example["onsets"],
            example["sample_rate"],
        )

        scores.append(loss)

    return float(np.mean(scores))


bounds = [
    (1.0, 30.0),     # Fast release (ms)
    (3.0, 100.0),    # Slow attack (ms)
    (20.0, 500.0),   # Slow release (ms)
    (0.1, 2.0),      # Trigger threshold
    (0.1, 0.8),      # Rearm/trigger ratio
]

result = differential_evolution(
    lambda p: objective(p, training_dataset),
    bounds=bounds,
    maxiter=150,
    popsize=12,
    tol=0.001,
    seed=42,
    polish=False,
)

print("Best parameters:", result.x)
print("Training loss:", result.fun)
