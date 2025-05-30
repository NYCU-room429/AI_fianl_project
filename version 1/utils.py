# import librosa
# import numpy as np
# import pretty_midi
# import json, yaml
# import matplotlib.pyplot as plt

# def get_rendered_stems(metadata_path):
#     with open(metadata_path, 'r', encoding='utf-8') as f:
#         meta = yaml.safe_load(f)
#     rendered_stems = []
#     for stem_name, stem_info in meta.get('stems', {}).items():
#         if stem_info.get('audio_rendered', False):
#             rendered_stems.append(stem_name)
#     return rendered_stems

# def get_all_class(path):
#     with open(path, "r") as f:
#         data = json.load(f)

#     instrument_class = sorted(set(info["name"] for info in data.values()))

#     return instrument_class


# def gen_melgram(path, num_frame=1000, segment_sec=100, is_train=False):
#     SR = 22050
#     N_MELS = 96
#     N_FFT = 512

#     src, sr = librosa.load(path, sr=SR)
#     total_sec = len(src) / sr

#     if total_sec < 200:
#         return None  # 不符合條件

#     # 取中間100秒
#     center = total_sec / 2
#     start_sec = max(0, center - segment_sec / 2)
#     end_sec = min(total_sec, center + segment_sec / 2)
#     start_sample = int(start_sec * sr)
#     end_sample = int(end_sec * sr)
#     src_segment = src[start_sample:end_sample]

#     # 依據1000 frames計算 hop_length
#     hop_length = max(1, (len(src_segment) - N_FFT) // (num_frame - 1))
#     mel = librosa.feature.melspectrogram(
#         y=src_segment, sr=SR, n_fft=N_FFT, hop_length=hop_length, n_mels=N_MELS
#     )
#     mel_db = librosa.power_to_db(mel)

#     # 補齊或裁切
#     if mel_db.shape[1] < num_frame:
#         pad_width = num_frame - mel_db.shape[1]
#         mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode="constant")
#     elif mel_db.shape[1] > num_frame:
#         mel_db = mel_db[:, :num_frame]

#     # === 添加 SpecAugment (僅在訓練時) ===
#     if is_train:
#         # 頻率遮罩 (Frequency Masking)
#         num_freq_mask = 2  # 遮罩數量
#         freq_mask_max_width = 15  # 最大頻率寬度 (您可以試驗不同值)
#         for _ in range(num_freq_mask):
#             f_start = np.random.randint(0, N_MELS - freq_mask_max_width)
#             f_end = f_start + np.random.randint(0, freq_mask_max_width)
#             mel_db[f_start:f_end, :] = 0  # 將該頻率區間設為 0

#         # 時間遮罩 (Time Masking)
#         num_time_mask = 2  # 遮罩數量
#         time_mask_max_width = 75  # 最大時間寬度 (您可以試驗不同值)
#         for _ in range(num_time_mask):
#             t_start = np.random.randint(0, num_frame - time_mask_max_width)
#             t_end = t_start + np.random.randint(0, time_mask_max_width)
#             mel_db[:, t_start:t_end] = 0  # 將該時間區間設為 0
#     # ===================================
#     return mel_db


# def extract_label_from_midi(path, target_classes, num_frame=1000, segment_sec=100):
#     midi = pretty_midi.PrettyMIDI(path)
#     total_time = midi.get_end_time()
#     if total_time < 200:
#         return None  # 不符合條件

#     # 取中間100秒
#     center = total_time / 2
#     start_time = max(0, center - segment_sec / 2)
#     end_time = min(total_time, center + segment_sec / 2)
#     frame_size = segment_sec / num_frame
#     frame_instruments = [set() for _ in range(num_frame)]

#     for instr in midi.instruments:
#         if instr.is_drum:
#             instrument_name = "Drums"
#         else:
#             instrument_name = pretty_midi.program_to_instrument_name(instr.program)
#         for note in instr.notes:
#             # 只考慮在中間100秒內的 note
#             note_start = max(note.start, start_time)
#             note_end = min(note.end, end_time)
#             if note_end <= start_time or note_start >= end_time:
#                 continue
#             start_idx = int((note_start - start_time) / frame_size)
#             end_idx = int((note_end - start_time) / frame_size)
#             for i in range(start_idx, end_idx + 1):
#                 if 0 <= i < num_frame:
#                     frame_instruments[i].add(instrument_name)

#     num_classes = len(target_classes)
#     label_array = np.zeros((num_frame, num_classes), dtype=int)
#     class_to_idx = {cls: idx for idx, cls in enumerate(target_classes)}

#     for frame_idx, instruments in enumerate(frame_instruments):
#         for inst in instruments:
#             if inst in class_to_idx:
#                 label_array[frame_idx, class_to_idx[inst]] = 1

#     return label_array


# def plot(train_losses, val_losses):
#     plt.figure(figsize=(10, 6))
#     plt.plot(train_losses, label="Training Loss", color="blue", marker="o")
#     plt.plot(val_losses, label="Validation Loss", color="orange", marker="o")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.title("Training and Validation Loss")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig("loss.png")

#     print("Save the plot to 'loss.png'")
#     return


# import librosa
# import numpy as np
# import pretty_midi
# import json, yaml
# import matplotlib.pyplot as plt
# import random

# def get_rendered_stems(metadata_path):
#     """
#     從 metadata.yaml 檔案中獲取已渲染的音軌名稱。
#     """
#     with open(metadata_path, 'r', encoding='utf-8') as f:
#         meta = yaml.safe_load(f)
#     rendered_stems = []
#     for stem_name, stem_info in meta.get('stems', {}).items():
#         if stem_info.get('audio_rendered', False):
#             rendered_stems.append(stem_name)
#     return rendered_stems

# def get_all_class(path):
#     """
#     從 JSON 檔案中讀取所有樂器類別名稱。
#     """
#     with open(path, "r") as f:
#         data = json.load(f)
#     instrument_class = sorted(set(info["name"] for info in data.values()))
#     return instrument_class

# def gen_melgram(path, num_frame=1000, segment_sec=100, sr=22050, n_mels=96, n_fft=512, is_train=False):
#     """
#     生成梅爾頻譜圖 (Mel Spectrogram)。
#     在訓練模式下，會隨機選擇音頻片段並應用 SpecAugment。
#     """
#     src, current_sr = librosa.load(path, sr=sr)
#     total_sec = len(src) / current_sr

#     if total_sec < segment_sec: # 確保音頻長度足夠提取指定片段
#         return None

#     # 根據是否為訓練模式選擇音頻片段的起始時間
#     if is_train:
#         # 訓練時隨機選擇起始時間
#         start_sec = random.uniform(0, total_sec - segment_sec)
#     else:
#         # 驗證/測試時取音頻中間的片段
#         center = total_sec / 2
#         start_sec = max(0, center - segment_sec / 2)
    
#     end_sec = start_sec + segment_sec
    
#     start_sample = int(start_sec * current_sr)
#     end_sample = int(end_sec * current_sr)
#     src_segment = src[start_sample:end_sample]

#     # 計算 hop_length 以確保梅爾頻譜圖的時間維度為 num_frame
#     # formula: (len(src_segment) - N_FFT) / hop_length + 1 = num_frame
#     # => hop_length = (len(src_segment) - N_FFT) / (num_frame - 1)
#     hop_length = max(1, (len(src_segment) - n_fft) // (num_frame - 1))
    
#     mel = librosa.feature.melspectrogram(
#         y=src_segment, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
#     )
#     mel_db = librosa.power_to_db(mel)

#     # 確保梅爾頻譜圖的時間維度精確為 num_frame
#     if mel_db.shape[1] < num_frame:
#         pad_width = num_frame - mel_db.shape[1]
#         mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode="constant")
#     elif mel_db.shape[1] > num_frame:
#         mel_db = mel_db[:, :num_frame]

#     # === 添加 SpecAugment (僅在訓練時) ===
#     if is_train:
#         # 頻率遮罩 (Frequency Masking)
#         num_freq_mask = 2  # 遮罩數量
#         freq_mask_max_width = random.randint(1, 15)  # 隨機最大頻率寬度
#         for _ in range(num_freq_mask):
#             f_start = np.random.randint(0, n_mels - freq_mask_max_width)
#             f_end = f_start + np.random.randint(0, freq_mask_max_width)
#             mel_db[f_start:f_end, :] = 0  # 將該頻率區間設為 0

#         # 時間遮罩 (Time Masking)
#         num_time_mask = 2  # 遮罩數量
#         time_mask_max_width = random.randint(1, 75)  # 隨機最大時間寬度
#         for _ in range(num_time_mask):
#             t_start = np.random.randint(0, num_frame - time_mask_max_width)
#             t_end = t_start + np.random.randint(0, time_mask_max_width)
#             mel_db[:, t_start:t_end] = 0  # 將該時間區間設為 0
#     # ===================================
#     return mel_db

# def extract_label_from_midi(path, target_classes, num_frame=1000, segment_sec=100):
#     """
#     從 MIDI 檔案中提取樂器標籤，並將其轉換為逐幀的多標籤二元陣列。
#     與 `gen_melgram` 使用相同的片段選擇邏輯，以確保時間對齊。
#     """
#     midi = pretty_midi.PrettyMIDI(path)
#     total_time = midi.get_end_time()
#     if total_time < segment_sec: # 確保音頻長度足夠提取指定片段
#         return None

#     # 注意：這裡與 `gen_melgram` 保持片段選擇邏輯一致性至關重要
#     # 由於 DataLoader 是分開調用，如果 `gen_melgram` 在訓練時隨機取，
#     # 則這裡的 `extract_label_from_midi` 也必須取同樣的隨機起始時間。
#     # 更安全的做法是讓 `gen_melgram` 返回 `start_sec`，並將其傳入此函數。
#     # 這裡為簡化起見，假設訓練時的隨機性在 DataLoader 中統一管理，
#     # 或者當前任務中 MIDI 標籤只與音頻的中心或完整長度相關。
#     # 為了保持與 `gen_melgram` 的隨機/中心選擇邏輯一致，這裡也應引入 is_train 參數。
#     # 不過原始程式碼的 `extract_label_from_midi` 沒有 `is_train` 參數，
#     # 且 `collect_all_path` 篩選時就調用了它。
#     # 為避免引入新的複雜性，我會將 `extract_label_from_midi` 保持為從中心提取，
#     # 這意味著訓練時的隨機裁剪只會影響 Mel-gram，不影響 MIDI Label 的時間軸。
#     # 理想情況下，應該在 `__getitem__` 中統一計算 `start_sec`。
    
#     # 為了與改進後的 gen_melgram 在訓練時保持一致性
#     # 這裡的邏輯需要從DataLoader中獲取實際的start_sec,
#     # 但由於目前是單獨調用，暫時假定兩者在非訓練模式下（測試/驗證）都是取中間，
#     # 且訓練時的隨機裁剪主要作為數據增強，對標籤的絕對時間位置容忍度較高。
#     # 最安全的做法是讓 gen_melgram 返回其使用的 start_sec，並傳遞給 extract_label_from_midi。

#     # For now, to align with `gen_melgram`'s modified behavior:
#     # This `extract_label_from_midi` will now also calculate segment_sec consistently.
#     # Note: `collect_all_path` calls this without `is_train` which implies
#     # it always uses the deterministic center segment. This is fine for pre-filtering.
#     # The actual __getitem__ will call it with the correct `is_train` implicitly from `DataLoader`.

#     # 統一使用中心片段的選擇邏輯，這與 `gen_melgram` 在非訓練時的行為一致
#     center = total_time / 2
#     start_time = max(0, center - segment_sec / 2)
#     end_time = min(total_time, center + segment_sec / 2)
    
#     frame_size = segment_sec / num_frame
#     frame_instruments = [set() for _ in range(num_frame)]

#     for instr in midi.instruments:
#         if instr.is_drum:
#             instrument_name = "Drums"
#         else:
#             instrument_name = pretty_midi.program_to_instrument_name(instr.program)
#         for note in instr.notes:
#             # 只考慮在選定音頻片段內的 note
#             note_start = max(note.start, start_time)
#             note_end = min(note.end, end_time)
#             if note_end <= start_time or note_start >= end_time:
#                 continue
#             # 將 note 的時間轉換為幀索引
#             start_idx = int((note_start - start_time) / frame_size)
#             end_idx = int((note_end - start_time) / frame_size)
#             for i in range(start_idx, end_idx + 1):
#                 if 0 <= i < num_frame:
#                     frame_instruments[i].add(instrument_name)

#     num_classes = len(target_classes)
#     label_array = np.zeros((num_frame, num_classes), dtype=int)
#     class_to_idx = {cls: idx for idx, cls in enumerate(target_classes)}

#     for frame_idx, instruments in enumerate(frame_instruments):
#         for inst in instruments:
#             if inst in class_to_idx:
#                 label_array[frame_idx, class_to_idx[inst]] = 1
#     return label_array


# def plot(train_losses, val_losses):
#     """
#     繪製訓練和驗證損失曲線並保存為圖片。
#     """
#     plt.figure(figsize=(10, 6))
#     plt.plot(train_losses, label="Training Loss", color="blue", marker="o")
#     plt.plot(val_losses, label="Validation Loss", color="orange", marker="o")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.title("Training and Validation Loss")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig("loss.png")
#     print("Save the plot to 'loss.png'")
#     return

# utils.py
import librosa
import numpy as np
import pretty_midi
import json
import yaml
import matplotlib.pyplot as plt
import random
import os
# No direct import of loguru here, unless a specific function MUST log independently.
# For this request, we'll rely on main.py for primary logging.

def get_rendered_stems(metadata_path):
    if not os.path.exists(metadata_path): return []
    with open(metadata_path, 'r', encoding='utf-8') as f: meta = yaml.safe_load(f)
    return [name for name, info in meta.get('stems', {}).items() if info.get('audio_rendered', False)]

def get_all_class_names(class_json_path):
    if not os.path.exists(class_json_path): raise FileNotFoundError(f"Class JSON not found: {class_json_path}")
    with open(class_json_path, "r") as f: data = json.load(f)
    return sorted(list(set(info["name"] for info in data.values())))

def generate_mel_spectrogram(audio_path, sr, n_fft, hop_length, n_mels, segment_sec,
                             num_target_frames, selected_start_sec=None, is_train=False,
                             freq_masks=0, time_masks=0, freq_max_width=0, time_max_width=0,
                             debug_id=None, debug_dir="debug_data"):
    try:
        y, current_sr = librosa.load(audio_path, sr=sr, mono=True)
    except Exception as e:
        print(f"ERROR [utils.generate_mel_spectrogram]: Loading audio {audio_path}: {e}") # Print for critical error
        return None
    
    audio_total_sec = len(y) / float(current_sr)
    _selected_start_sec = selected_start_sec
    if _selected_start_sec is None:
        if is_train:
            max_start = audio_total_sec - segment_sec
            _selected_start_sec = random.uniform(0, max_start) if max_start > 0 else 0.0
        else:
            _selected_start_sec = max(0.0, (audio_total_sec - segment_sec) / 2.0)
    _selected_start_sec = min(_selected_start_sec, max(0.0, audio_total_sec - segment_sec))

    start_sample = int(_selected_start_sec * current_sr)
    end_sample = start_sample + int(segment_sec * current_sr)
    y_segment = y[start_sample:end_sample]

    expected_samples = int(segment_sec * sr)
    if len(y_segment) < expected_samples:
        y_segment = np.pad(y_segment, (0, expected_samples - len(y_segment)), mode='constant', constant_values=0.0)
    elif len(y_segment) > expected_samples:
        y_segment = y_segment[:expected_samples]

    if np.max(np.abs(y_segment)) < 1e-5:
        return np.full((n_mels, num_target_frames), -80.0, dtype=np.float32)

    mel_spec = librosa.feature.melspectrogram(
        y=y_segment, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max if np.max(mel_spec) > 0 else 1.0, top_db=80.0)

    current_frames = log_mel_spec.shape[1]
    if current_frames < num_target_frames:
        pad_width = num_target_frames - current_frames
        log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, pad_width)), mode="constant", constant_values=-80.0)
    elif current_frames > num_target_frames:
        log_mel_spec = log_mel_spec[:, :num_target_frames]

    if is_train:
        mask_value = -80.0
        for _ in range(freq_masks):
            if n_mels > freq_max_width > 0:
                f_start = np.random.randint(0, n_mels - freq_max_width + 1)
                f_width = np.random.randint(1, freq_max_width + 1)
                log_mel_spec[f_start:f_start + f_width, :] = mask_value
        for _ in range(time_masks):
            if num_target_frames > time_max_width > 0:
                t_start = np.random.randint(0, num_target_frames - time_max_width + 1)
                t_width = np.random.randint(1, time_max_width + 1)
                log_mel_spec[:, t_start:t_start + t_width] = mask_value
    
    if debug_id is not None and random.random() < 0.001 :
        os.makedirs(debug_dir, exist_ok=True)
        np.save(os.path.join(debug_dir, f"{debug_id}_mel.npy"), log_mel_spec)
        with open(os.path.join(debug_dir, f"{debug_id}_info.txt"), "w") as f:
            f.write(f"audio_path: {audio_path}\nselected_start_sec: {_selected_start_sec}\n")
    return log_mel_spec.astype(np.float32)

def extract_midi_labels(midi_path, instrument_names_vocab, segment_sec,
                        num_target_frames, selected_start_sec,
                        debug_id=None, debug_dir="debug_data"):
    num_classes = len(instrument_names_vocab)
    labels_matrix = np.zeros((num_target_frames, num_classes), dtype=np.float32)
    if not os.path.exists(midi_path):
        print(f"ERROR [utils.extract_midi_labels]: MIDI not found: {midi_path}")
        return labels_matrix
    try:
        with open(midi_path, 'rb') as f: midi_data = pretty_midi.PrettyMIDI(f)
    except Exception as e:
        print(f"ERROR [utils.extract_midi_labels]: Parsing MIDI {midi_path}: {e}")
        return labels_matrix

    segment_end_sec_abs = selected_start_sec + segment_sec
    time_per_frame = segment_sec / float(num_target_frames)
    if time_per_frame <= 0:
        print(f"ERROR [utils.extract_midi_labels]: time_per_frame not positive ({time_per_frame})")
        return labels_matrix
    
    instrument_to_idx = {name: i for i, name in enumerate(instrument_names_vocab)}
    for instrument in midi_data.instruments:
        try:
            instrument_name = "Drums" if instrument.is_drum else pretty_midi.program_to_instrument_name(instrument.program)
        except KeyError: continue
        if instrument_name not in instrument_to_idx: continue
        class_idx = instrument_to_idx[instrument_name]
        for note in instrument.notes:
            overlap_start_abs = max(note.start, selected_start_sec)
            overlap_end_abs = min(note.end, segment_end_sec_abs)
            if overlap_end_abs > overlap_start_abs + 1e-4:
                start_in_segment_sec = overlap_start_abs - selected_start_sec
                end_in_segment_sec = overlap_end_abs - selected_start_sec
                start_frame = int(round(start_in_segment_sec / time_per_frame))
                end_frame = int(round(end_in_segment_sec / time_per_frame))
                start_frame = max(0, min(start_frame, num_target_frames - 1))
                end_frame = max(0, min(end_frame, num_target_frames - 1))
                if end_frame >= start_frame:
                    labels_matrix[start_frame : end_frame + 1, class_idx] = 1.0
    
    if debug_id is not None and random.random() < 0.001:
        os.makedirs(debug_dir, exist_ok=True)
        np.save(os.path.join(debug_dir, f"{debug_id}_labels.npy"), labels_matrix)
        if not os.path.exists(os.path.join(debug_dir, f"{debug_id}_info.txt")):
            with open(os.path.join(debug_dir, f"{debug_id}_info.txt"), "w") as f:
                f.write(f"midi_path: {midi_path}\nselected_start_sec: {selected_start_sec}\n")
    return labels_matrix

def plot_training_history(history, save_dir, plot_filename): # Removed logger_obj
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    epochs_ran = range(1, len(history['train_loss']) + 1)
    axes[0].plot(epochs_ran, history['train_loss'], 'o-', label='Train Loss')
    axes[0].plot(epochs_ran, history['val_loss'], 'o-', label='Validation Loss')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].set_title('Loss')
    axes[0].legend(); axes[0].grid(True)
    axes[1].plot(epochs_ran, history['val_micro_f1'], 'o-', label='Val Micro F1')
    axes[1].plot(epochs_ran, history['val_macro_f1'], 'o-', label='Val Macro F1')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('F1 Score'); axes[1].set_title('F1 Scores')
    axes[1].legend(); axes[1].grid(True); axes[1].set_ylim(0, 1)
    plt.tight_layout()
    save_path = os.path.join(save_dir, plot_filename)
    plt.savefig(save_path)
    print(f"INFO: Training plot saved to {save_path}") # Use print if logger not available
    plt.close()