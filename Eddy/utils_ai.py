import os
import numpy as np
import librosa
import pretty_midi
import yaml
import json

def extract_mel(audio_path, sr=22050, n_mels=128, hop_length=512):
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    db_mel = librosa.power_to_db(mel, ref=np.max)
    return db_mel

def read_instruments_class(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        instruments = json.load(f)
    return instruments

def get_target_instrument_classes(mapping_path: str):
    instruments_mapping = read_instruments_class(mapping_path)
    classes = sorted(list(set(item["name"] for item in instruments_mapping.values())))
    return classes

def read_metadata(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        metadata = yaml.safe_load(f)
    return metadata

def read_midi(path: str):
    return pretty_midi.PrettyMIDI(path)

def midi_to_label(midi, n_frames, frame_duration, instruments_mapping, instrument_classes):
    label = np.zeros((n_frames, len(instrument_classes)), dtype=np.float32)
    for instrument in midi.instruments:
        program_str = str(instrument.program)
        if program_str in instruments_mapping:
            mapped_class = instruments_mapping[program_str]["name"]
            if mapped_class in instrument_classes:
                class_idx = instrument_classes.index(mapped_class)
                for note in instrument.notes:
                    start_frame = int(note.start / frame_duration)
                    end_frame = int(note.end / frame_duration)
                    start_frame = max(0, min(start_frame, n_frames - 1))
                    end_frame = max(0, min(end_frame, n_frames - 1))
                    label[start_frame:end_frame+1, class_idx] = 1.0
    return label

def scan_tracks(dataset_root):
    tracks = []
    for track in os.listdir(dataset_root):
        track_dir = os.path.join(dataset_root, track)
        if not os.path.isdir(track_dir):
            continue
        mix_path = os.path.join(track_dir, "mix.flac")
        midi_path = os.path.join(track_dir, "all_src.mid")
        meta_path = os.path.join(track_dir, "metadata.yaml")
        if os.path.isfile(mix_path) and os.path.isfile(midi_path) and os.path.isfile(meta_path):
            tracks.append({
                "track": track,
                "mix": mix_path,
                "midi": midi_path,
                "metadata": meta_path
            })
    return tracks