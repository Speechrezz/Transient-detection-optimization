import numpy as np
import numpy.typing as npt
from pathlib import Path
import soundfile as sf

def load_audio_mono(path: str) -> tuple[npt.NDArray[np.float32], float]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)

    # Average all channels to mono
    audio = np.mean(audio, axis=1, dtype=np.float32)

    return audio, float(sample_rate)

def save_audio_mono(path: str, audio: npt.NDArray[np.float32], sample_rate: float):
    audio = np.asarray(audio, dtype=np.float32)
    sf.write(path, audio, int(sample_rate), subtype="FLOAT")

def to_transient_positions(transients: npt.NDArray[np.float32]):
    return np.flatnonzero(transients)

def save_transients(path: str, positions: npt.NDArray[np.intp]):
    np.save(path, positions)

def load_transients(path: str) -> npt.NDArray[np.intp]:
    return np.load(path)

def load_dataset(directory: str | Path) -> list[dict]:
    directory = Path(directory)
    dataset = []

    for wav_path in sorted(directory.glob("*.wav")):
        npy_path = wav_path.with_suffix(".npy")

        if not npy_path.exists():
            raise FileNotFoundError(f"Missing annotations: {npy_path}")

        # Load audio as mono float32
        audio, sample_rate = sf.read(
            wav_path,
            dtype="float32",
            always_2d=True,
        )
        audio = np.mean(audio, axis=1, dtype=np.float32)

        # Load annotated transient sample positions
        onsets = np.load(npy_path, allow_pickle=False).astype(np.intp)

        # Validate annotations
        if onsets.ndim != 1:
            raise ValueError(f"Invalid onset array: {npy_path}")

        if np.any(onsets < 0) or np.any(onsets >= len(audio)):
            raise ValueError(f"Onsets out of bounds: {npy_path}")

        dataset.append({
            "audio": audio,
            "sample_rate": sample_rate,
            "onsets": np.sort(onsets),
        })

    if not dataset:
        raise ValueError(f"No WAV files found in {directory}")

    return dataset