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
        # directory
        track_dir = os.path.join(path, track)
        mix_path = os.path.join(track_dir, "mix.flac")
        meta_path = os.path.join(track_dir, "metadata.yaml")
        stems_dir = os.path.join(track_dir, "stems")
        midi_dir = os.path.join(track_dir, "MIDI")
        all_src_midi_dir = os.path.join(track_dir, "all_src_mid")
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
    return midi.instruments


def get_instrument_activity_labels(
    midi_files: List[str], 
    mel_shape: Tuple[int, int], 
    sr: int = 44100, 
    hop_length: int = 512
) -> np.ndarray:
    """
    根據一首曲目的多個 MIDI 文件，生成一個「樂器活動標註矩陣」。
    這個矩陣的每一行代表一種樂器，每一列代表一個梅爾頻譜圖的時間幀。
    矩陣中的值是 True 或 False，表示該樂器在該時間幀是否有音符在發聲。
    """
    n_frames = mel_shape[1]
    # Collect all unique program numbers (instruments) in this track
    instrument_programs = []
    midi_instruments = []
    for midi_path in midi_files:
        midi = pretty_midi.PrettyMIDI(midi_path)
        for inst in midi.instruments:
            instrument_programs.append(inst.program)
            midi_instruments.append(inst)
    unique_programs = sorted(set(instrument_programs))
    program_to_idx = {prog: i for i, prog in enumerate(unique_programs)}
    activity = np.zeros((len(unique_programs), n_frames), dtype=bool)

    # For each instrument, mark frames where any note is active
    for inst in midi_instruments:
        idx = program_to_idx[inst.program]
        for note in inst.notes:
            start_frame = int(note.start * sr / hop_length)
            end_frame = int(note.end * sr / hop_length)
            activity[idx, start_frame:end_frame] = True

    return activity # activity: np.ndarray, 形狀為 (樂器數量, 幀數)，dtype=bool