import numpy as np
import numpy.typing as npt
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