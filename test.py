from fileio import load_audio_mono, save_audio_mono
from detector import TransientDetector, TransientParams

audio, sample_rate = load_audio_mono("./dataset/drums.wav")

print(sample_rate, type(sample_rate))

params = TransientParams(
    sample_rate=sample_rate,
    fast_decay=200.0,
    slow_decay=600.0,
    slow_attack=20.0,
    trigger_thresh=1.0,
    rearm_ratio=0.2
)

detector = TransientDetector(params)

output = detector.process(audio)

save_audio_mono("./out/drums.wav", output, sample_rate)