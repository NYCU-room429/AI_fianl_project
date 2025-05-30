import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from CRNN_ai import CachedMelLabelDataset
import utils_ai
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# 參數
DATASET_ROOT = "/Users/eddy/Desktop/code/AI_fianl_project/slakh2100_flac_redux/slakh2100_flac_redux/train"
INSTRUMENT_MAPPING_PATH = "/Users/eddy/Desktop/code/AI_fianl_project/slakh-utils/midi_inst_values/general_midi_inst_0based.json"
MEL_SAVE_DIR = "mel_cache"
N_MELS = 128
MAX_FRAMES = 1000
TEST_SPLIT_RATIO = 0.2
RANDOM_STATE = 42

# 樂器類別
instruments_mapping = utils_ai.read_instruments_class(INSTRUMENT_MAPPING_PATH)
instrument_classes = utils_ai.get_target_instrument_classes(INSTRUMENT_MAPPING_PATH)
n_classes = len(instrument_classes)

# 掃描資料集
tracks = utils_ai.scan_tracks(DATASET_ROOT)
train_tracks, _ = train_test_split(tracks, test_size=TEST_SPLIT_RATIO, random_state=RANDOM_STATE)

train_set = CachedMelLabelDataset(
    train_tracks, MEL_SAVE_DIR, instrument_classes, instruments_mapping,
    n_mels=N_MELS, max_frames=MAX_FRAMES, train=True
)

POS_WEIGHT_PATH = "pos_weight.npy"
if os.path.exists(POS_WEIGHT_PATH):
    pos_weight = torch.tensor(np.load(POS_WEIGHT_PATH), dtype=torch.float32)
    print(f"[INFO] Loaded pos_weight from {POS_WEIGHT_PATH}")
    print("pos_weight:", pos_weight)
    pos_weight = np.clip(pos_weight, 0.5, 10)
    np.save(POS_WEIGHT_PATH, pos_weight)
    print(f"[INFO] Clipped and saved pos_weight to {POS_WEIGHT_PATH}")
    print("clip 後 pos_weight:", pos_weight)
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
else:
    def compute_pos_weight(dataset):
        total = np.zeros(n_classes)
        count = 0
        loader = DataLoader(dataset, batch_size=32)
        for _, label in tqdm(loader, desc="Computing pos_weight"):
            total += label.sum(dim=(0,1)).numpy()
            count += label.shape[1]
        pos_weight = (count - total) / (total + 1e-6)
        return pos_weight
    pos_weight = compute_pos_weight(train_set)
    # 修正：clip pos_weight，避免極端值和負數
    pos_weight = np.clip(pos_weight, 0.5, 10)
    np.save(POS_WEIGHT_PATH, pos_weight)
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
    print(f"[INFO] Computed and saved pos_weight to {POS_WEIGHT_PATH}")
    print("pos_weight:", pos_weight)