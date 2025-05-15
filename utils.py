from mido import MidiFile
import os
import numpy as np
import pandas as pd
from typing import List, Tuple
import librosa
import pretty_midi
import yaml
import json
import soundfile as sf
from tqdm import tqdm


def load_dataset(path: str) -> pd.DataFrame:
    track_data = []
    for track in tqdm(os.listdir(path=path)):
        track_dir = os.path.join(path, track)
        mix_path = os.path.join(track_dir, "mix.flac")
        meta_path = os.path.join(track_dir, "metadata.yaml")
        stems_dir = os.path.join(track_dir, "stems")
        midi_dir = os.path.join(track_dir, "MIDI")
        # stem read
        stem_flacs = []
        for stem in os.listdir(stems_dir):
            if stem.endswith(".flac"):
                stem_flacs.append(os.path.join(stems_dir, stem))
        # MIDI read
        midi_files = []
        for midi in os.listdir(midi_dir):
            if midi.endswith(".mid"):
                midi_files.append(os.path.join(midi_dir, midi))

        track_data.append(
            {
                "track": track,
                "mix": mix_path if os.path.isfile(mix_path) else None,
                "metadata": meta_path if os.path.isfile(meta_path) else None,
                "stems": stem_flacs,
                "midis": midi_files,
            }
        )
    return pd.DataFrame(track_data)


def read_flacs(path: str) -> np.ndarray:
    vocal, sr = librosa.load(path, sr=None, mono=False)
    if vocal.ndim > 1 and vocal.shape[0] > 1:
        vocal_mono = librosa.to_mono(vocal)
    else:
        vocal_mono = vocal

    mel = librosa.feature.melspectrogram(y=vocal_mono, sr=sr)
    db_mel = librosa.power_to_db(mel, ref=np.max)

    return db_mel


def read_instruments_class(path: str) -> dict:
    f = open(path, "r", encoding="utf-8")
    instruments = json.load(f)
    return instruments


def read_metadata(path: str) -> dict:
    raise NotImplementedError("YAML reading not implemented yet.")


def read_midi(path: str) -> List[pretty_midi.Instrument]:
    raise NotImplementedError("MIDI reading not implemented yet.")
