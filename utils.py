import os
import numpy as np
import pandas as pd
from typing import List
import librosa
import pretty_midi
import yaml
import json
from tqdm import tqdm


def load_dataset(path: str) -> pd.DataFrame:
    track_data = []
    for track in tqdm(os.listdir(path=path)):
        # directory
        track_dir = os.path.join(path, track)
        mix_path = os.path.join(track_dir, "mix.flac")
        meta_path = os.path.join(track_dir, "metadata.yaml")
        stems_dir = os.path.join(track_dir, "stems")
        midi_dir = os.path.join(track_dir, "MIDI")
        all_src_midi_dir = os.path.join(track_dir, "all_src.mid")
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
                "all_src_midi": (
                    all_src_midi_dir if os.path.isfile(all_src_midi_dir) else None
                ),
                "stems": stem_flacs,
                "midis": midi_files,
            }
        )

    return pd.DataFrame(track_data)


def read_flac(path: str) -> np.ndarray:
    vocal, sr = librosa.load(path, sr=None, mono=False)
    if vocal.ndim > 1 and vocal.shape[0] > 1:
        vocal_mono = librosa.to_mono(vocal)  # convert to mono
    else:
        vocal_mono = vocal

    mel = librosa.feature.melspectrogram(y=vocal_mono, sr=sr)  # melspectrogram
    db_mel = librosa.power_to_db(mel, ref=np.max)  # convert to decibel

    return db_mel


# structure be like {'0': {'class': 'Piano', 'name': 'Acoustic Grand Piano'}, '1': {'class': 'Piano', 'name': 'Bright Acoustic Piano'},...}
def read_instruments_class(path: str) -> dict:
    f = open(path, "r", encoding="utf-8")
    instruments = json.load(f)
    return instruments


def read_metadata(path: str) -> dict:
    f = open(path, "r", encoding="utf-8")
    metadata = yaml.safe_load(f)
    return metadata


def read_midi(path: str) -> List[pretty_midi.Instrument]:
    midi = pretty_midi.PrettyMIDI(path)
    return midi


# 把樂器種類標上標籤
def get_target_instrument_classes(mapping_path: str) -> List[str]:
    """
    Loads the instrument mapping and extracts a sorted list of unique instrument class names.
    This list defines the order of classes for the multi-hot labels.
    """
    try:
        instruments_mapping = read_instruments_class(mapping_path)

        classes = sorted(
            list(set(item["name"] for item in instruments_mapping.values()))
        )
        return classes
    except FileNotFoundError:
        print(f"Error: Instrument mapping file not found at {mapping_path}")
        return []
    except Exception as e:
        print(f"Error reading instrument mapping file: {e}")
        return []
