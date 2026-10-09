from fileio import load_audio_mono, save_audio_mono, to_transient_positions, save_transients
from detector import TransientDetector, TransientParams

filename = "drums"
audio, sample_rate = load_audio_mono(f"./dataset/{filename}.wav")

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

save_transients(f"./dataset/{filename}.npy", to_transient_positions(output))
save_audio_mono(f"./out/{filename}.wav", output, sample_rate)