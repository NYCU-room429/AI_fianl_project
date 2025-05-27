import torch
import numpy as np
import librosa
from CRNN import CRNN
import utils

# 參數（請根據你的訓練設定調整）
AUDIO_PATH = "slakh2100_flac_redux\\slakh2100_flac_redux\\validation\\Track01753\\stems\\S01.flac"
MODEL_PATH = "instrument_crnn_model.pth"
INSTRUMENT_MAPPING_PATH = "slakh-utils\\midi_inst_values\\general_midi_inst_0based.json"
N_MELS = 128
MAX_MEL_FRAMES = 5000
SR = 22050
HOP_LENGTH = 512

# 1. 載入 instrument_classes
instrument_classes = utils.get_target_instrument_classes(INSTRUMENT_MAPPING_PATH)
num_classes = len(instrument_classes)

# 2. 載入模型
# 2. 載入模型
model = CRNN(
    input_channel=1,
    n_mels=N_MELS,
    max_mel_frames=MAX_MEL_FRAMES,
    num_classes=num_classes,
)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# 3. 音檔轉 Mel spectrogram
y, sr = librosa.load(AUDIO_PATH, sr=SR)
mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
mel_db = librosa.power_to_db(mel)

# 4. Padding 或截斷到 (N_MELS, MAX_MEL_FRAMES)
if mel_db.shape[1] < MAX_MEL_FRAMES:
    pad_width = MAX_MEL_FRAMES - mel_db.shape[1]
    mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode="constant")
else:
    mel_db = mel_db[:, :MAX_MEL_FRAMES]
mel_spec = mel_db[np.newaxis, ...]  # shape: (1, N_MELS, MAX_MEL_FRAMES)

# 5. 推論
input_tensor = (
    torch.from_numpy(mel_spec).unsqueeze(0).float()
)  # (1, 1, N_MELS, MAX_MEL_FRAMES)
with torch.no_grad():
    output = model(input_tensor)
    prob = torch.sigmoid(output).cpu().numpy()[0]  # shape: (1000, N)
    prob_max = prob.max(axis=0)  # shape: (N,)
    pred_binary = (prob_max > 0.8).astype(int)  # shape: (N,)

# 6. 轉成樂器名稱
predicted_instruments = [
    name for i, name in enumerate(instrument_classes) if pred_binary[i]
]
print("Predicted multi-hot:", pred_binary)
print("Predicted instruments:", predicted_instruments)
print("Raw probabilities (max per class):", prob_max)
