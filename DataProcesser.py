import os
import librosa
import numpy as np
import pretty_midi
import yaml
import soundfile as sf
from tqdm import tqdm

SAMPLE_RATE = 16000
HOP_LENGTH = 512
N_MELS = 128
WINDOW_SECONDS = 1.0


def extract_mel(path, sr=SAMPLE_RATE):
    y, _ = librosa.load(path, sr=sr, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=1024, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel)
    return mel_db.T  # (time, mel)


def parse_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        meta = yaml.safe_load(f)
    stem_map = {}
    for stem_id, data in meta["stems"].items():
        instrument = data["inst_class"].lower()  # 轉小寫
        midi_path = f"{stem_id}.mid"  # 假設 MIDI 檔名為 S00.mid, S01.mid ...
        stem_map[instrument] = midi_path
    return stem_map


def extract_instrument_labels(midi_dir, stem_map, total_frames, frame_hz, label_map):
    num_classes = len(label_map)
    labels = np.zeros((total_frames, num_classes), dtype=np.float32)

    for instrument, midi_file in stem_map.items():
        midi_path = os.path.join(midi_dir, os.path.basename(midi_file))
        if not os.path.exists(midi_path):
            continue

        try:
            pm = pretty_midi.PrettyMIDI(midi_path)
            for inst in pm.instruments:
                # 這裡 instrument 名稱需和 label_map key 一致
                if instrument not in label_map:
                    continue
                idx = label_map[instrument]
                for note in inst.notes:
                    start_frame = int(note.start * frame_hz)
                    end_frame = int(note.end * frame_hz)
                    # 邊界檢查
                    start_frame = max(0, start_frame)
                    end_frame = min(total_frames - 1, end_frame)
                    labels[start_frame : end_frame + 1, idx] = 1.0
        except Exception as e:
            print(f"Error processing {midi_path}: {e}")
            continue

    return labels


def time_slice(data, labels, window_size):
    slices = []
    sliced_labels = []
    num_slices = data.shape[0] // window_size
    for i in range(num_slices):
        x = data[i * window_size : (i + 1) * window_size]
        y = labels[i * window_size : (i + 1) * window_size]
        y_agg = (y.sum(axis=0) > 0).astype(np.float32)
        slices.append(x)
        sliced_labels.append(y_agg)
    return slices, sliced_labels


def process_track(track_path, label_map, save_dir):
    print(f"Processing {track_path}...")

    # Feature
    mel_path = os.path.join(track_path, "mix.flac")
    mel = extract_mel(mel_path)
    frame_hz = SAMPLE_RATE / HOP_LENGTH

    # Labels
    yaml_path = os.path.join(track_path, "metadata.yaml")
    # midi_dir = os.path.join(track_path, 'MIDI')
    print(track_path)
    stem_map = parse_yaml(yaml_path)
    labels = extract_instrument_labels(
        track_path, stem_map, mel.shape[0], frame_hz, label_map
    )

    # Time slicing
    win_size = int(WINDOW_SECONDS * frame_hz)
    x_slices, y_slices = time_slice(mel, labels, win_size)

    # Save slices
    for i, (x, y) in enumerate(zip(x_slices, y_slices)):
        np.save(os.path.join(save_dir, f"{os.path.basename(track_path)}_{i}_x.npy"), x)
        np.save(os.path.join(save_dir, f"{os.path.basename(track_path)}_{i}_y.npy"), y)


def data_processor(input_dir, output_dir, label_map):
    os.makedirs(output_dir, exist_ok=True)
    for track_name in os.listdir(input_dir):
        track_path = os.path.join(input_dir, track_name)
        if os.path.isdir(track_path):
            track_output_dir = os.path.join(output_dir, track_name)
            os.makedirs(track_output_dir, exist_ok=True)
            process_track(track_path, label_map, track_output_dir)


if __name__ == "__main__":
    label_map = {
        "bass": 0,
        "brass": 1,
        "flute": 2,
        "guitar": 3,
        "keyboard": 4,
        "organ": 5,
        "piano": 6,
        "saxophone": 7,
        "strings": 8,
        "synthesizer": 9,
        "trumpet": 10,
        "trombone": 11,
        "tuba": 12,
        "violin": 13,
    }
    for split in ["test", "train", "validation"]:
        input_dir = f"slakh2100_flac_redux\\slakh2100_flac_redux\\{split}"
        output_dir = f"ProcessData\\{split}"
        data_processor(input_dir, output_dir, label_map)
