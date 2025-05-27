import torch
import numpy as np
import librosa
from CRNN_ai import CRNN
import utils_ai
import pretty_midi
import os


# 參數（根據訓練設定調整）
AUDIO_PATH = "/Users/eddy/Desktop/code/AI_fianl_project/slakh2100_flac_redux/slakh2100_flac_redux/test/Track01878/mix.flac"
MODEL_PATH = "best_model_Focal.pth"
MIDI_DIR = "/Users/eddy/Desktop/code/AI_fianl_project/slakh2100_flac_redux/slakh2100_flac_redux/test/Track01878/MIDI"
INSTRUMENT_MAPPING_PATH = "/Users/eddy/Desktop/code/AI_fianl_project/slakh-utils/midi_inst_values/general_midi_inst_0based.json"
N_MELS = 128
MAX_MEL_FRAMES = 1000
SR = 22050
HOP_LENGTH = 512

# === 請填入訓練時的 mean/std/threshold ===
mean_std = np.load("mel_global_mean_std_Focal.npy")
#global_mean=-49.896114349365234, global_std=17.286714553833008 #給crnn_model.pth
global_mean, global_std = mean_std[0], mean_std[1]
best_ths = np.load("best_ths_Focal_best.npy")

# 1. 載入 instrument_classes
instruments_mapping = utils_ai.read_instruments_class(INSTRUMENT_MAPPING_PATH)
instrument_classes = utils_ai.get_target_instrument_classes(INSTRUMENT_MAPPING_PATH)
n_classes = len(instrument_classes)

# 2. 載入模型
model = CRNN(N_MELS, n_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# 3. 音檔轉 Mel spectrogram
y, sr = librosa.load(AUDIO_PATH, sr=SR)
mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
mel_db = librosa.power_to_db(mel, ref=np.max)

# 4. Padding 或截斷到 (N_MELS, MAX_MEL_FRAMES)
if mel_db.shape[1] < MAX_MEL_FRAMES:
    pad_width = MAX_MEL_FRAMES - mel_db.shape[1]
    mel_db = np.pad(mel_db, ((0,0),(0,pad_width)), mode='constant')
else:
    mel_db = mel_db[:, :MAX_MEL_FRAMES]

# 5. 標準化（與訓練一致）
mel_db = (mel_db - global_mean) / (global_std + 1e-6)

# 6. 推論
mel_spec = mel_db[np.newaxis, np.newaxis, ...]  # shape: (1, 1, N_MELS, MAX_MEL_FRAMES)
input_tensor = torch.from_numpy(mel_spec).float()
with torch.no_grad():
    output = model(input_tensor)
    prob = torch.sigmoid(output).cpu().numpy()[0]  # shape: (frames, n_classes)
    prob_max = prob.max(axis=0)                    # shape: (n_classes,)

print("best_ths:", best_ths)

# 7. 使用最佳 threshold
pred_binary = (prob_max > 0.5).astype(int)

midi_instruments = set()
for midi_file in os.listdir(MIDI_DIR):
    if midi_file.endswith(".mid"):
        midi_path = os.path.join(MIDI_DIR, midi_file)
        midi = pretty_midi.PrettyMIDI(midi_path)
        for inst in midi.instruments:
            if inst.is_drum:
                if "Drums" in instrument_classes:
                    midi_instruments.add("Drums")
            else:
                program_str = str(inst.program)
                if program_str in instruments_mapping:
                    mapped_name = instruments_mapping[program_str]["name"]
                    if mapped_name in instrument_classes:
                        midi_instruments.add(mapped_name)
print("MIDI instruments in this track:", sorted(midi_instruments))

# 8. 轉成樂器名稱
predicted_instruments = [name for i, name in enumerate(instrument_classes) if pred_binary[i]]
print("Predicted multi-hot:", pred_binary)
print("Predicted instruments:", predicted_instruments)
print("Raw probabilities (max per class):", prob_max)