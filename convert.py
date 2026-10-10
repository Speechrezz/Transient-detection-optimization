import numpy as np
from pathlib import Path
import soundfile as sf
from fileio import to_transient_positions, save_transients

def convert_targets_to_onsets(directory: str | Path):
    directory = Path(directory)

    for wav_path in sorted(directory.glob("*.target.wav")):
        # Load audio as mono float32
        audio, sample_rate = sf.read(
            wav_path,
            dtype="float32",
            always_2d=True,
        )
        audio = np.mean(audio, axis=1, dtype=np.float32)

        filename = wav_path.stem.removesuffix(".target")
        save_transients(f"./dataset/{filename}.npy", to_transient_positions(audio))

if __name__ == "__main__":
    convert_targets_to_onsets("./dataset")