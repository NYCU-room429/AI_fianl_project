import librosa
import numpy as np
import pretty_midi
import json, yaml
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings(
    "ignore",
    message="Tempo, Key or Time signature change events found on non-zero tracks*",
    category=RuntimeWarning,
)

_instrument_data_cache = {}


def _get_or_load_instrument_data(json_path):
    """
    Loads instrument mapping data from the JSON file.
    The JSON file MUST be a single valid JSON object.
    """
    if json_path not in _instrument_data_cache:
        if not os.path.exists(json_path):
            raise FileNotFoundError(
                f"Instrument JSON definition file not found: {json_path}"
            )
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Error decoding JSON from {json_path}. Ensure it's a single valid JSON object. Details: {e}"
            )

        program_to_class = {}
        seen_classes = set()

        # Example: {"0": {"class": "Piano", "name": ...}, ..., "128": {"class": "Drums", "name": "Drums"}}
        for prog_key_str, details in data.items():
            class_name = details.get("class")
            if class_name is None:
                continue

            seen_classes.add(class_name)

            if prog_key_str == "128":
                program_to_class["Drums_specific_class"] = class_name  # e.g., "Drums"
            else:
                try:
                    prog_num_int = int(prog_key_str)
                    if 0 <= prog_num_int <= 127:
                        program_to_class[prog_num_int] = class_name
                except ValueError:
                    pass

        if "Drums" not in seen_classes:
            seen_classes.add("Drums")

        all_instrument_classes = sorted(list(seen_classes))

        _instrument_data_cache[json_path] = {
            "program_to_class_map": program_to_class,
            "all_classes": all_instrument_classes,
        }
    return _instrument_data_cache[json_path]


def get_all_class(json_path):
    """Returns a sorted list of unique 'class' names from the JSON file."""
    data = _get_or_load_instrument_data(json_path)
    return data["all_classes"]


def get_program_to_class_map(json_path):
    """Returns a dictionary mapping MIDI program numbers to 'class' names."""
    data = _get_or_load_instrument_data(json_path)
    return data["program_to_class_map"]


def get_rendered_stems(metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    rendered_stems = []
    for stem_name, stem_info in meta.get("stems", {}).items():
        if stem_info.get("audio_rendered", False):
            rendered_stems.append(stem_name)
    return rendered_stems


def gen_melgram(path, num_frame=1000, segment_sec=100, is_train=False):
    SR = 44100
    N_MELS = 128
    N_FFT = 2048

    try:
        src, sr = librosa.load(path, sr=SR)
    except Exception as e:
        # print(f"Error loading audio {path}: {e}")
        return None

    total_sec = len(src) / sr

    if total_sec < 200:
        # print(f"Audio {path} is too short ({total_sec:.2f}s), less than 200s. Skipping melgram.")
        return None

    center_sec = total_sec / 2
    start_sec = max(0, center_sec - segment_sec / 2)
    end_sec = min(total_sec, start_sec + segment_sec)
    end_sec = min(end_sec, total_sec)

    if (end_sec - start_sec) < 10.0:
        return None

    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    src_segment = src[start_sample:end_sample]

    if len(src_segment) < N_FFT:
        # print(f"Segment from {path} is too short for FFT ({len(src_segment)} samples). Skipping melgram.")
        return None

    hop_length = (
        (len(src_segment) - N_FFT) // (num_frame - 1) if num_frame > 1 else N_FFT
    )  # Avoid div by zero
    hop_length = max(1, hop_length)  # Ensure hop_length is at least 1

    mel = librosa.feature.melspectrogram(
        y=src_segment, sr=SR, n_fft=N_FFT, hop_length=hop_length, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    if mel_db.shape[1] < num_frame:
        pad_width = num_frame - mel_db.shape[1]
        mel_db = np.pad(
            mel_db,
            ((0, 0), (0, pad_width)),
            mode="constant",
            constant_values=np.min(mel_db),
        )  # Pad with min value
    elif mel_db.shape[1] > num_frame:
        mel_db = mel_db[:, :num_frame]

    if is_train:  # SpecAugment
        num_freq_mask = 2
        freq_mask_max_width = 15
        for _ in range(num_freq_mask):
            if N_MELS > freq_mask_max_width:
                f_start = np.random.randint(0, N_MELS - freq_mask_max_width)
                f_end = f_start + np.random.randint(1, freq_mask_max_width)
                mel_db[f_start:f_end, :] = np.min(mel_db)

        num_time_mask = 2
        time_mask_max_width = 75
        for _ in range(num_time_mask):
            if num_frame > time_mask_max_width:
                t_start = np.random.randint(0, num_frame - time_mask_max_width)
                t_end = t_start + np.random.randint(1, time_mask_max_width)
                mel_db[:, t_start:t_end] = np.min(mel_db)

    return mel_db


def extract_label_from_midi(
    midi_path, target_classes, program_class_mapping, num_frame=1000, segment_sec=100
):
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        # print(f"Could not parse MIDI file {midi_path}: {e}")
        return None

    total_time = midi.get_end_time()
    if total_time < 200:
        # print(f"MIDI {midi_path} is too short ({total_time:.2f}s), less than 200s. Skipping label extraction.")
        return None

    # Determine the 100s segment from the MIDI's timeline
    center_midi_time = total_time / 2
    segment_start_time = max(0, center_midi_time - segment_sec / 2)
    segment_end_time = min(total_time, segment_start_time + segment_sec)
    segment_end_time = min(segment_end_time, total_time)

    actual_segment_duration = segment_end_time - segment_start_time
    if actual_segment_duration <= 1.0:
        # print(f"MIDI segment from {midi_path} is too short ({actual_segment_duration:.2f}s). Skipping.")
        return None

    frame_duration_in_segment = actual_segment_duration / num_frame
    if frame_duration_in_segment <= 0:
        return None

    frame_instruments = [set() for _ in range(num_frame)]
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(target_classes)}

    for instr in midi.instruments:
        instrument_category = None
        if instr.is_drum:
            instrument_category = program_class_mapping.get(
                "Drums_specific_class", "Drums"
            )
        else:
            instrument_category = program_class_mapping.get(instr.program)

        if instrument_category is None:
            # print(f"Warning: Program {instr.program} (name: {pretty_midi.program_to_instrument_name(instr.program)}) in {midi_path} not in program_class_mapping or maps to None. Skipping instrument.")
            continue

        if instrument_category not in class_to_idx:
            continue

        for note in instr.notes:
            note_start_abs = note.start
            note_end_abs = note.end

            eff_note_start_in_segment = max(note_start_abs, segment_start_time)
            eff_note_end_in_segment = min(note_end_abs, segment_end_time)

            if (
                eff_note_end_in_segment <= eff_note_start_in_segment
                or eff_note_end_in_segment <= segment_start_time
                or eff_note_start_in_segment >= segment_end_time
            ):
                continue

            relative_note_start = eff_note_start_in_segment - segment_start_time
            relative_note_end = eff_note_end_in_segment - segment_start_time

            start_frame_idx = int(relative_note_start / frame_duration_in_segment)
            end_frame_idx = int(relative_note_end / frame_duration_in_segment)

            start_frame_idx = max(0, min(start_frame_idx, num_frame - 1))
            end_frame_idx = max(0, min(end_frame_idx, num_frame - 1))

            for i in range(
                start_frame_idx, end_frame_idx + 1
            ):  # Inclusive of end_frame_idx
                frame_instruments[i].add(instrument_category)

    label_array = np.zeros((num_frame, len(target_classes)), dtype=int)
    for frame_idx, instruments_in_frame_set in enumerate(frame_instruments):
        for inst_class_name_str in instruments_in_frame_set:
            label_array[frame_idx, class_to_idx[inst_class_name_str]] = 1

    return label_array


def plot(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss", marker=".")
    plt.plot(val_losses, label="Validation Loss", marker=".")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_class_based.png")  # New name for plot
    print("Saved loss plot to 'loss_class_based.png'")
