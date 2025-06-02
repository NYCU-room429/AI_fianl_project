from torch.utils.data import Dataset, DataLoader
import torch
import os
from tqdm import tqdm
from utils import (
    gen_melgram,
    get_all_class,  # Returns list of class strings
    get_program_to_class_map,  # Returns dict for mapping
    extract_label_from_midi,
    get_rendered_stems,
)
import numpy as np  # Added for potential use, not strictly necessary here


CLASS_JSON_PATH = "slakh-utils\\midi_inst_values\\general_midi_inst_0based.json"


def collect_all_path(root_path, class_json_path, is_test=False):
    target_classes_list = get_all_class(class_json_path)
    program_class_mapping = get_program_to_class_map(class_json_path)

    midi_files = []
    flac_files = []

    for track in tqdm(os.listdir(root_path), desc="Collecting all files' path"):
        track_path = os.path.join(root_path, track)
        if not os.path.isdir(track_path):
            continue

        stems_dir = os.path.join(track_path, "stems")
        midi_dir = os.path.join(track_path, "MIDI")
        metadata_path = os.path.join(track_path, "metadata.yaml")

        if not os.path.exists(metadata_path):
            # print(f"Warning: metadata.yaml not found in {track_path}, skipping rendered stems check for this track.")
            rendered_stems = []
        else:
            try:
                rendered_stems = get_rendered_stems(metadata_path)
            except Exception as e:
                # print(f"Warning: Could not read metadata.yaml from {track_path}: {e}. Skipping rendered stems check.")
                rendered_stems = []

        midi_path = os.path.join(track_path, "all_src.mid")
        flac_path = os.path.join(track_path, "mix.flac")

        if not (os.path.exists(midi_path) and os.path.exists(flac_path)):
            continue

        # Check if label can be extracted before adding
        if (
            extract_label_from_midi(
                midi_path, target_classes_list, program_class_mapping
            )
            is None
        ):
            continue

        midi_files.append(midi_path)
        flac_files.append(flac_path)

        if is_test:
            continue

        # Process individual stems if not in test mode
        if not (os.path.isdir(stems_dir) and os.path.isdir(midi_dir)):
            continue

        flac_list = sorted([f for f in os.listdir(stems_dir) if f.endswith(".flac")])
        midi_list = sorted([f for f in os.listdir(midi_dir) if f.endswith(".mid")])

        flac_basenames = [os.path.splitext(f)[0] for f in flac_list]
        midi_basenames = [os.path.splitext(f)[0] for f in midi_list]
        common_names = set(flac_basenames) & set(midi_basenames)

        for name in sorted(common_names):
            if name not in rendered_stems:
                continue

            midi_small_path = os.path.join(midi_dir, name + ".mid")
            flac_small_path = os.path.join(stems_dir, name + ".flac")

            if not (
                os.path.exists(midi_small_path) and os.path.exists(flac_small_path)
            ):
                continue

            if (
                extract_label_from_midi(
                    midi_small_path, target_classes_list, program_class_mapping
                )
                is None
            ):
                continue

            midi_files.append(midi_small_path)
            flac_files.append(flac_small_path)

    print(f"有效資料數量 (after filtering in collect_all_path): {len(midi_files)}")
    return midi_files, flac_files


class LoadDataset(Dataset):
    def __init__(self, midi_file_list, flac_file_list, class_json_path, is_train=False):
        self.midi_file_list = midi_file_list
        self.flac_file_list = flac_file_list

        self.target_classes_list = get_all_class(class_json_path)
        self.program_class_mapping = get_program_to_class_map(class_json_path)

        self.is_train = is_train
        self.class_json_path = class_json_path

    def __len__(self):
        return len(self.midi_file_list)

    def __getitem__(self, idx):
        mel = gen_melgram(self.flac_file_list[idx], is_train=self.is_train)
        if mel is None:
            return None, None

        mel_mean = mel.mean()
        mel_std = mel.std()
        if mel_std > 0:
            mel = (mel - mel_mean) / mel_std
        mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)

        label = extract_label_from_midi(
            self.midi_file_list[idx],
            target_classes=self.target_classes_list,
            program_class_mapping=self.program_class_mapping,
        )
        if label is None:
            # print(f"Warning: extract_label_from_midi returned None for {self.midi_file_list[idx]}. Returning None.")
            return None, None

        label_tensor = torch.tensor(label, dtype=torch.float32)
        return mel_tensor, label_tensor


def collate_fn_skip_nones(batch):
    filtered_batch = [
        item for item in batch if item[0] is not None and item[1] is not None
    ]

    if not filtered_batch:
        return None, None

    return torch.utils.data.dataloader.default_collate(filtered_batch)
