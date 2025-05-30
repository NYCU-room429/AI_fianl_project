from torch.utils.data import Dataset, DataLoader
import torch
import os
from tqdm import tqdm
from utils import (
    gen_melgram,
    get_all_class, # Returns list of class strings
    get_program_to_class_map, # Returns dict for mapping
    extract_label_from_midi,
    get_rendered_stems,
)
import numpy as np # Added for potential use, not strictly necessary here


# IMPORTANT: Ensure general_midi_inst_0based.json is a single valid JSON object.
# The hardcoded path below is used multiple times. Consider defining it as a constant.
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
        
        # Check for metadata existence before trying to read
        if not os.path.exists(metadata_path):
            # print(f"Warning: metadata.yaml not found in {track_path}, skipping rendered stems check for this track.")
            rendered_stems = [] # Assume no rendered stems if metadata is missing
        else:
            try:
                rendered_stems = get_rendered_stems(metadata_path)
            except Exception as e:
                # print(f"Warning: Could not read metadata.yaml from {track_path}: {e}. Skipping rendered stems check.")
                rendered_stems = []


        # Process main mix and MIDI
        midi_path = os.path.join(track_path, "all_src.mid")
        flac_path = os.path.join(track_path, "mix.flac")

        if not (os.path.exists(midi_path) and os.path.exists(flac_path)):
            continue
        
        # Check if label can be extracted before adding
        if extract_label_from_midi(midi_path, target_classes_list, program_class_mapping) is None:
            continue
        # Also check if melgram can be generated (implicitly checks duration)
        # This check might be redundant if extract_label_from_midi already filters by duration
        # However, gen_melgram could have other reasons to return None.
        # For simplicity, relying on extract_label_from_midi's duration filter.

        midi_files.append(midi_path)
        flac_files.append(flac_path)

        if is_test: # For test set, typically only mix.flac and all_src.mid are used.
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
            if name not in rendered_stems: # Ensure stem audio was actually rendered
                continue
            
            midi_small_path = os.path.join(midi_dir, name + ".mid")
            flac_small_path = os.path.join(stems_dir, name + ".flac")

            if not (os.path.exists(midi_small_path) and os.path.exists(flac_small_path)):
                continue

            if extract_label_from_midi(midi_small_path, target_classes_list, program_class_mapping) is None:
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
        self.class_json_path = class_json_path # Store if needed elsewhere

        # Optional: Pre-filter files for which melgram or label extraction fails
        # This makes __getitem__ cleaner but __init__ slower.
        # For now, assuming collect_all_path does sufficient filtering.
        # If not, __getitem__ needs to handle None returns gracefully (e.g. with a collate_fn)

    def __len__(self):
        return len(self.midi_file_list)

    def __getitem__(self, idx):
        # gen_melgram can return None if audio is too short (e.g. < 200s based on its internal check)
        mel = gen_melgram(
            self.flac_file_list[idx], is_train=self.is_train
        )
        if mel is None:
            # This case should ideally be filtered out by collect_all_path
            # or handled by a custom collate_fn in DataLoader
            # print(f"Warning: gen_melgram returned None for {self.flac_file_list[idx]}. Returning None.")
            return None, None 

        mel_mean = mel.mean()
        mel_std = mel.std()
        if mel_std > 0:
            mel = (mel - mel_mean) / mel_std
        mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        
        # extract_label_from_midi can also return None (e.g. if MIDI too short or unparseable)
        label = extract_label_from_midi(
            self.midi_file_list[idx], 
            target_classes=self.target_classes_list,
            program_class_mapping=self.program_class_mapping
            # num_frame and segment_sec use defaults from extract_label_from_midi
        )
        if label is None:
            # print(f"Warning: extract_label_from_midi returned None for {self.midi_file_list[idx]}. Returning None.")
            return None, None

        label_tensor = torch.tensor(label, dtype=torch.float32)
        return mel_tensor, label_tensor

# Custom collate_fn to filter out None items from a batch
def collate_fn_skip_nones(batch):
    # Filter out items where mel_tensor or label_tensor is None
    filtered_batch = [item for item in batch if item[0] is not None and item[1] is not None]
    
    if not filtered_batch:
        # If all items were None, return None or empty tensors.
        # Returning None might require specific handling in the training loop.
        # It's often better to ensure the dataset doesn't produce Nones,
        # or return placeholder empty tensors.
        # For simplicity here, if batch is empty, default_collate might error or return empty list.
        # Let's return Tensors that can be identified as problematic
        # Or simply return None and let the training loop handle it
        return None, None # This signals the training loop to skip this batch.

    return torch.utils.data.dataloader.default_collate(filtered_batch)