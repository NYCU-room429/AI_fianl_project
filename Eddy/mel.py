import utils_ai
import numpy as np
import os

DATASET_ROOT = "slakh2100_flac_redux/slakh2100_flac_redux/train"
MEL_SAVE_DIR = "mel_cache"
N_MELS = 128

os.makedirs(MEL_SAVE_DIR, exist_ok=True)
tracks = utils_ai.scan_tracks(DATASET_ROOT)
for item in tracks:
    mel = utils_ai.extract_mel(item["mix"], n_mels=N_MELS)
    np.save(os.path.join(MEL_SAVE_DIR, f"{item['track']}.npy"), mel)