import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
from multiprocessing import Pool  # 導入 Pool 和 cpu_count

import utils

# parameter
MAX_MEL_FRAMES = 5000  # padding聲普圖用的，訓練的時候自行調整

CNN_INPUT_CHANNELS = 1
N_MELS = 128

CNN_CHANNELS = [32, 64, 128]
CNN_KERNEL_SIZE = (3, 3)
CNN_POOL_SIZE = (2, 2)

RNN_HIDDEN_SIZE = 128
RNN_NUM_LAYERS = 2
RNN_DROPOUT = 0.3

DROPOUT_RATE = 0.3  # CRNN的dropout rate，訓練的時候自行調整

INSTRUMENT_MAPPING_PATH = "slakh-utils/midi_inst_values/general_midi_inst_0based.json"  # utils生出來的json路徑


# (保留之前的 parameter 和 get_target_instrument_classes 函數)


# --- 新增的處理函數 ---
# 確保這個函數是頂層函數，不是類的方法，這樣才能被多進程序列化
def _process_single_track(args):
    """
    用於多進程處理單個音軌的函數。
    接收一個包含所有必要數據的元組，返回處理後的梅爾頻譜圖和標籤。
    """
    (
        row_data,
        max_mel_frames,
        instrument_classes_list,
        instruments_mapping,
        cnn_pool_size,
        cnn_channels,
    ) = args

    mix_path = row_data["mix"]
    midi_files = row_data["midis"]

    # 檢查音訊檔案與 MIDI 是否存在
    if mix_path is None or not os.path.exists(mix_path):
        return None  # 返回 None 表示該樣本無效
    if not midi_files:
        return None

    mel_spec_processed = None
    pooled_labels = None

    # 處理音訊，產生 Mel spectrogram
    try:
        mel_spec_db = utils.read_flac(mix_path)  # 確保 read_flac 處理了 sr
        current_frames = mel_spec_db.shape[1]
        if current_frames > max_mel_frames:
            mel_spec_processed = mel_spec_db[:, :max_mel_frames]
        else:
            padding_width = ((0, 0), (0, max_mel_frames - current_frames))
            mel_spec_processed = np.pad(
                mel_spec_db,
                padding_width,
                mode="constant",
                constant_values=-100,
            )
        mel_spec_processed = np.expand_dims(
            mel_spec_processed, axis=0
        )  # 增加 channel 維度
    except Exception as e:
        # print(f"Error processing audio for track {row_data['track']}: {e}") # 考慮是否打印錯誤，多進程打印可能很亂
        return None

    # 計算每個 frame 的時間長度（秒）
    # 這些參數與 utils.read_flac 中的 librosa 設置匹配
    hop_length = 512
    sr = 22050  # 這裡的 sr 必須與 read_flac 中實際使用的 sr 匹配
    frame_duration = hop_length / sr

    # 計算 CNN 下採樣後的時間步數
    target_pooled_time_steps = max_mel_frames // (cnn_pool_size[1] ** len(cnn_channels))

    def _generate_frame_level_labels_helper(
        midi_files, n_frames, instruments_mapping, frame_duration, instrument_classes
    ):
        frame_labels = np.zeros((n_frames, len(instrument_classes)), dtype=np.float32)
        for midi_path in midi_files:
            if not os.path.exists(midi_path):
                continue
            try:
                midi_data = utils.read_midi(midi_path)
                for instrument in midi_data.instruments:
                    program_str = str(instrument.program)
                    if program_str in instruments_mapping:
                        mapped_class = instruments_mapping[program_str]["name"]
                        if mapped_class in instrument_classes:
                            class_idx = instrument_classes.index(mapped_class)
                            for note in instrument.notes:
                                start_frame = int(note.start / frame_duration)
                                end_frame = int(note.end / frame_duration)
                                start_frame = max(0, min(start_frame, n_frames - 1))
                                end_frame = max(0, min(end_frame, n_frames - 1))
                                frame_labels[start_frame : end_frame + 1, class_idx] = (
                                    1.0
                                )
            except Exception as e:
                # print(f"Error processing MIDI for {midi_path}: {e}")
                continue
        return frame_labels

    raw_frame_labels = _generate_frame_level_labels_helper(
        midi_files,
        max_mel_frames,
        instruments_mapping,
        frame_duration,
        instrument_classes_list,
    )

    def _downsample_label_helper(label, pooled_time_steps):
        # label: (max_mel_frames, num_classes)
        label_tensor = (
            torch.from_numpy(label).float().T.unsqueeze(0).unsqueeze(0)
        )  # (1, 1, num_classes, max_mel_frames)
        # 注意: 如果這裡在子進程中報錯，可能是因為 PyTorch 在子進程的初始化問題
        # 可以考慮將這一步的 PyTorch 操作移到主進程收集結果後再做，或使用 numpy 實現池化
        try:
            pooled_tensor = torch.nn.functional.max_pool2d(
                label_tensor,
                kernel_size=(1, label_tensor.shape[-1] // pooled_time_steps),
            )
        except RuntimeError as e:
            # 可能是 PyTorch 初始化問題，或者 GPU 內存不足等
            print(
                f"PyTorch RuntimeError in _downsample_label_helper: {e}. Falling back to NumPy pooling if available."
            )
            # 退回到 NumPy 實現的池化（如果必要）
            # 這裡簡化處理，如果 PyTorch 失敗就返回 None
            return None

        pooled_np = (
            pooled_tensor.squeeze(0).squeeze(0).T.numpy()
        )  # (pooled_time_steps, num_classes)
        return pooled_np

    pooled_labels = _downsample_label_helper(raw_frame_labels, target_pooled_time_steps)

    if mel_spec_processed is None or pooled_labels is None:
        return None

    return mel_spec_processed, pooled_labels.astype(np.float32)


class MusicInstrumentDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        instrument_mapping_path: str = INSTRUMENT_MAPPING_PATH,
        max_mel_frames: int = MAX_MEL_FRAMES,
    ):
        self.dataframe = dataframe
        self.max_mel_frames = max_mel_frames

        self.instrument_classes = utils.get_target_instrument_classes(
            instrument_mapping_path
        )
        if not self.instrument_classes:
            raise ValueError(
                f"Could not load or find instrument classes from {instrument_mapping_path}"
            )

        self.mlb = MultiLabelBinarizer(
            classes=self.instrument_classes
        )  # 這個 mlb 其實沒用上，因為直接生成了多熱碼

        self.data = []
        self.labels = []
        self.load_data()
        self.n_mels = N_MELS  # 保持從常量獲取
        if len(self.data) > 0:
            self.n_mels = self.data[0].shape[1]

        print(
            f"Dataset initialized with N_MELS={self.n_mels}, MAX_MEL_FRAMES={self.max_mel_frames}, num_classes={len(self.instrument_classes)}"
        )
        # 移除 generate_frame_level_labels 和 downsample_label 類方法，因為它們現在作為 helper 函數在 _process_single_track 內部使用

    def load_data(self):
        print(
            f"Loading and preprocessing data for {len(self.dataframe)} tracks using multiprocessing..."
        )

        try:
            instruments_mapping = utils.read_instruments_class(INSTRUMENT_MAPPING_PATH)
        except Exception as e:
            print(f"Could not load instrument mapping: {e}")
            instruments_mapping = {}

        # 準備傳遞給子進程的參數列表
        tasks = []
        for index, row in self.dataframe.iterrows():
            tasks.append(
                (
                    row,
                    self.max_mel_frames,
                    self.instrument_classes,  # 傳遞給子進程，避免重新加載
                    instruments_mapping,  # 傳遞給子進程
                    CNN_POOL_SIZE,  # 傳遞 CNN 參數
                    CNN_CHANNELS,  # 傳遞 CNN 參數
                )
            )

        # 使用多進程池
        # 可以調整 processes 數量，通常是 cpu_count() - 1 或更多，根據記憶體和 CPU 核數
        num_processes = os.cpu_count() or 1
        print(f"Using {num_processes} processes for data loading.")

        results = []
        with Pool(processes=num_processes) as pool:
            # imap_unordered 可以按順序返回結果，而 imap 則保證順序
            # tqdm(iterable, ...) 用於顯示進度條
            for result in tqdm(
                pool.imap_unordered(_process_single_track, tasks),
                total=len(tasks),
                desc="Processing tracks",
            ):
                if result is not None:
                    results.append(result)

        # 收集所有有效結果
        for mel_spec_processed, pooled_labels in results:
            self.data.append(mel_spec_processed)
            self.labels.append(pooled_labels)

        print(
            f"Finished data loading and preprocessing. Found {len(self.data)} valid samples."
        )
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        print(f"Dataset shapes: X={self.data.shape}, y={self.labels.shape}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mel_spec = self.data[idx]
        label = self.labels[idx]
        mel_spec_tensor = torch.from_numpy(mel_spec)
        label_tensor = torch.from_numpy(label)
        return mel_spec_tensor, label_tensor


# CRNN Model Definition


class CRNN(nn.Module):
    def __init__(self, input_channel, n_mels, max_mel_frames, num_classes):
        super(CRNN, self).__init__()

        self.n_mels = n_mels
        self.max_mel_frames = max_mel_frames
        self.num_classes = num_classes

        # CNN Layers
        cnn_layers = []
        in_c = input_channel

        for out_c in CNN_CHANNELS:
            # padding
            padding = (CNN_KERNEL_SIZE[0] // 2, CNN_KERNEL_SIZE[1] // 2)
            cnn_layers.append(
                nn.Conv2d(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=CNN_KERNEL_SIZE,
                    padding=padding,
                )
            )
            cnn_layers.append(nn.BatchNorm2d(out_c))
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.MaxPool2d(kernel_size=CNN_POOL_SIZE))
            in_c = out_c

        self.cnn = nn.Sequential(*cnn_layers)

        # Calculate RNN Input Size
        # Create a dummy tensor to pass through CNN to get the output shape
        # Input shape: (batch_size, input_channel, n_mels, max_mel_frames)
        with torch.no_grad():
            dummy_input = torch.randn(1, input_channel, n_mels, max_mel_frames)
            cnn_output_shape = self.cnn(dummy_input).shape

        # CNN output shape: (batch_size, final_channels, pooled_n_mels, pooled_time_frames)
        rnn_input_size = (
            cnn_output_shape[1] * cnn_output_shape[2]
        )  # channels * pooled_n_mels
        pooled_time_steps = cnn_output_shape[3]  # pooled_time_frames

        print(f"Calculated CNN output shape: {cnn_output_shape}")
        print(f"Calculated RNN input sequence length: {pooled_time_steps}")
        print(f"Calculated RNN input features per step: {rnn_input_size}")

        self.rnn = nn.GRU(
            input_size=rnn_input_size,  # Features per time step
            hidden_size=RNN_HIDDEN_SIZE,
            num_layers=RNN_NUM_LAYERS,
            batch_first=True,  # Input/Output Tensors are (batch, seq, feature)
            bidirectional=True,
            dropout=RNN_DROPOUT if RNN_NUM_LAYERS > 1 else 0,
        )

        # Fully Connected Layers
        rnn_output_size = RNN_HIDDEN_SIZE * 2
        self.fc = nn.Sequential(
            nn.Linear(rnn_output_size, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # Input x shape: (batch_size, 1, N_MELS, MAX_MEL_FRAMES)

        # CNN Forward
        x = self.cnn(x)
        # CNN output shape: (batch_size, final_channels, pooled_n_mels, pooled_time_frames)

        # Reshape for RNN
        # Permute dimensions to (batch_size, pooled_time_frames, final_channels, pooled_n_mels)
        x = x.permute(0, 3, 1, 2)
        # Flatten the last two dimensions to get features per time step
        x = x.reshape(x.size(0), x.size(1), -1)
        # Shape now: (batch_size, sequence_length, input_size) for RNN

        # RNN Forward
        # rnn_out shape: (batch_size, 2 * RNN_HIDDEN_SIZE)
        rnn_out, _ = self.rnn(x)

        # Fully Connected Forward
        output = self.fc(rnn_out)
        # Output shape: (batch_size, pooled_time_steps, num_classes)

        # Apply Sigmoid activation for multi-label classification
        # output = torch.sigmoid(output)

        return output
