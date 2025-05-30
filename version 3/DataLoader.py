from torch.utils.data import Dataset, DataLoader
import torch
import os
from tqdm import tqdm
from utils import (
    gen_melgram,
    get_all_class,
    extract_label_from_midi,
    get_rendered_stems,
)


def collect_all_path(root_path, is_test=False):
    target_class = get_all_class(
        "C:\\Users\\linbr\\Desktop\\AI_fianl_project\\slakh-utils\\midi_inst_values\\general_midi_inst_0based.json"
    )
    midi_files = []
    flac_files = []

    for track in tqdm(os.listdir(root_path), desc="Collecting all files' path"):
        track_path = os.path.join(root_path, track)
        stems_dir = os.path.join(track_path, "stems")
        midi_dir = os.path.join(track_path, "MIDI")
        metadata_path = os.path.join(track_path, "metadata.yaml")
        rendered_stems = get_rendered_stems(metadata_path)

        if not os.path.isdir(track_path):
            continue

        midi_path = os.path.join(track_path, "all_src.mid")
        flac_path = os.path.join(track_path, "mix.flac")

        if extract_label_from_midi(midi_path, target_class) is None:
            continue

        midi_files.append(midi_path)
        flac_files.append(flac_path)

        if is_test:
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
            if extract_label_from_midi(midi_small_path, target_class) is None:
                continue
            midi_files.append(midi_small_path)
            flac_files.append(flac_small_path)

    print(f"有效資料數量: {len(midi_files)}")

    return midi_files, flac_files


class LoadDataset(Dataset):
    def __init__(self, midi_file_list, flac_file_list, is_train=False):
        self.midi_file_list = midi_file_list
        self.flac_file_list = flac_file_list
        self.target_class = get_all_class(
            "C:\\Users\\linbr\\Desktop\\AI_fianl_project\\slakh-utils\\midi_inst_values\\general_midi_inst_0based.json"
        )
        self.is_train = is_train
        # for midi, flac in tqdm(zip(midi_file_list, flac_file_list), desc="init data"):
        #     mel = gen_melgram(flac)
        #     label = extract_label_from_midi(midi, self.target_class)
        #     if mel is not None and label is not None:
        #         self.midi_file_list.append(midi)
        #         self.flac_file_list.append(flac)
        # print(f"有效資料數量: {len(self.midi_file_list)}")

    def __len__(self):
        return len(self.midi_file_list)

    def __getitem__(self, idx):
        mel = gen_melgram(
            self.flac_file_list[idx], is_train=self.is_train
        )  # numpy array
        mel_mean = mel.mean()
        mel_std = mel.std()
        if mel_std > 0:
            mel = (mel - mel_mean) / mel_std
        mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        label = extract_label_from_midi(self.midi_file_list[idx], self.target_class)
        label_tensor = torch.tensor(label, dtype=torch.float32)  # <--- 這裡要用 float32
        return mel_tensor, label_tensor
