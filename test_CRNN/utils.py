import librosa
import numpy as np
import pretty_midi
import json, yaml
import matplotlib.pyplot as plt

def get_rendered_stems(metadata_path):
    with open(metadata_path, 'r', encoding='utf-8') as f:
        meta = yaml.safe_load(f)
    rendered_stems = []
    for stem_name, stem_info in meta.get('stems', {}).items():
        if stem_info.get('audio_rendered', False):
            rendered_stems.append(stem_name)
    return rendered_stems

def get_all_class(path):
    with open(path, 'r') as f:
        data = json.load(f)

    instrument_class = sorted(set(info["name"] for info in data.values()))

    return instrument_class   

def gen_melgram(path, num_frame=1000, segment_sec=100, is_train=False):
    SR = 22050
    N_MELS = 96
    N_FFT = 512

    src, sr = librosa.load(path, sr=SR)
    total_sec = len(src) / sr

    if total_sec < 200:
        return None  # 不符合條件

    # 取中間100秒
    center = total_sec / 2
    start_sec = max(0, center - segment_sec / 2)
    end_sec = min(total_sec, center + segment_sec / 2)
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    src_segment = src[start_sample:end_sample]

    # 依據1000 frames計算 hop_length
    hop_length = max(1, (len(src_segment) - N_FFT) // (num_frame - 1))
    mel = librosa.feature.melspectrogram(y=src_segment, sr=SR, n_fft=N_FFT, hop_length=hop_length, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel)

    # 補齊或裁切
    if mel_db.shape[1] < num_frame:
        pad_width = num_frame - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
    elif mel_db.shape[1] > num_frame:
        mel_db = mel_db[:, :num_frame]

    # === 添加 SpecAugment (僅在訓練時) ===
    if is_train:
        # 頻率遮罩 (Frequency Masking)
        num_freq_mask = 2 # 遮罩數量
        freq_mask_max_width = 15 # 最大頻率寬度 (您可以試驗不同值)
        for _ in range(num_freq_mask):
            f_start = np.random.randint(0, N_MELS - freq_mask_max_width)
            f_end = f_start + np.random.randint(0, freq_mask_max_width)
            mel_db[f_start:f_end, :] = 0 # 將該頻率區間設為 0

        # 時間遮罩 (Time Masking)
        num_time_mask = 2 # 遮罩數量
        time_mask_max_width = 75 # 最大時間寬度 (您可以試驗不同值)
        for _ in range(num_time_mask):
            t_start = np.random.randint(0, num_frame - time_mask_max_width)
            t_end = t_start + np.random.randint(0, time_mask_max_width)
            mel_db[:, t_start:t_end] = 0 # 將該時間區間設為 0
    # ===================================
    return mel_db

def extract_label_from_midi(path, target_classes, num_frame=1000, segment_sec=100):
    midi = pretty_midi.PrettyMIDI(path)
    total_time = midi.get_end_time()
    if total_time < 200:
        return None  # 不符合條件

    # 取中間100秒
    center = total_time / 2
    start_time = max(0, center - segment_sec / 2)
    end_time = min(total_time, center + segment_sec / 2)
    frame_size = segment_sec / num_frame
    frame_instruments = [set() for _ in range(num_frame)]

    for instr in midi.instruments:
        if instr.is_drum:
            instrument_name = 'Drums'
        else:
            instrument_name = pretty_midi.program_to_instrument_name(instr.program)
        for note in instr.notes:
            # 只考慮在中間100秒內的 note
            note_start = max(note.start, start_time)
            note_end = min(note.end, end_time)
            if note_end <= start_time or note_start >= end_time:
                continue
            start_idx = int((note_start - start_time) / frame_size)
            end_idx = int((note_end - start_time) / frame_size)
            for i in range(start_idx, end_idx + 1):
                if 0 <= i < num_frame:
                    frame_instruments[i].add(instrument_name)

    num_classes = len(target_classes)
    label_array = np.zeros((num_frame, num_classes), dtype=int)
    class_to_idx = {cls: idx for idx, cls in enumerate(target_classes)}

    for frame_idx, instruments in enumerate(frame_instruments):
        for inst in instruments:
            if inst in class_to_idx:
                label_array[frame_idx, class_to_idx[inst]] = 1

    return label_array

def plot(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue', marker='o')
    plt.plot(val_losses, label='Validation Loss', color='orange', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss.png')
    
    print("Save the plot to 'loss.png'")
    return

# 範例用法
# def main():
#     audio_path = 'D:\\AI_fianl_project\\slakh2100_flac_redux\\slakh2100_flac_redux\\train\\Track00001\\mix.flac'
#     midi_path = 'D:\\AI_fianl_project\\slakh2100_flac_redux\\slakh2100_flac_redux\\train\\Track00001\\all_src.mid'
#     json_path = 'D:\\AI_fianl_project\\slakh-utils\\midi_inst_values\\general_midi_inst_0based.json'
#     instrument_class = get_all_class(json_path)

#     mel = gen_melgram(audio_path, num_frame=1000, segment_sec=100)
#     label = extract_label_from_midi(midi_path, instrument_class, num_frame=1000, segment_sec=100)
#     if mel is not None and label is not None:
#         print(mel.shape, label.shape)
#     else:
#         print("音樂長度不足200秒，已跳過。")

# main()

# def plot_melgram(melgram, sr=22050, hop_length=128, title='Mel Spectrogram'):
#     plt.figure(figsize=(10, 4))
#     librosa.display.specshow(melgram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
#     plt.title(title)
#     plt.colorbar(format='%+2.0f dB')
#     plt.tight_layout()
#     plt.show()

