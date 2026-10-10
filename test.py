from fileio import load_audio_mono, save_audio_mono, to_transient_positions, save_transients
from detector import TransientDetector, TransientParams

import sys
sys.path.append("build/Debug")
from detector_cpp import TransientParamsCpp, TransientDetectorCpp

filename = "drums"
audio, sample_rate = load_audio_mono(f"./dataset/{filename}.wav")

# Python

params = TransientParams(
    sample_rate=sample_rate,
    fast_decay=200.0,
    slow_decay=600.0,
    slow_attack=20.0,
    trigger_thresh=1.0,
    rearm_ratio=0.2,
)

detector = TransientDetector()
detector.prepare(params)

output = detector.process(audio)

save_transients(f"./dataset/{filename}_py.npy", to_transient_positions(output))
save_audio_mono(f"./out/{filename}_py.wav", output, sample_rate)

# C++

params = TransientParamsCpp(
    sample_rate=sample_rate,
    fast_decay=87.32654238498534,
    slow_attack=255.36860534477842,
    slow_decay=686.2136946992026,
    trigger_thresh=0.9626476693698596,
    rearm_ratio=0.30611970382173337,
)

detector = TransientDetectorCpp()
detector.prepare(params)

output = detector.process(audio)

save_transients(f"./dataset/{filename}_cpp.npy", to_transient_positions(output))
save_audio_mono(f"./out/{filename}_cpp.wav", output, sample_rate)