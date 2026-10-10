# Optimizing parameters for a transient detection algorithm

I am using a classic two-envelope approach, where a transient is detected whenever the faster envelope has more "energy" than the slower envelope.

However, there are several parameters that need manual tweaking:

1. `fast_decay`: Fast envelope release time in milliseconds
2. `slow_attack`: Slow envelope attack time in milliseconds
3. `slow_decay`: Slow envelope release time in milliseconds
4. `trigger_thresh`: Normalized threshold to trigger a transient
5. `rearm_ratio`: Ratio of `trigger_thresh` which "re-arms" the transient detector

Instead of manually determining these parameter values, I am using machine learning (differential evolution) to find optimal parameter values.

Run the `optimize.py` to perform the differential evolution.