# import torch
# from torch.utils.data import Dataset, DataLoader
# import torch.nn as nn
# import numpy as np
# import os


# class CRNNDataset(Dataset):
#     def __init__(self, data_dir: str, transform=None):
#         self.data_dir = data_dir
#         self.transform = transform
#         self.data = []
#         self.labels = []
#         self.load_data()

#     def load_data(self):
#         raise NotImplementedError("This method should be overridden by subclasses.")

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         raise NotImplementedError("This method should be overridden by subclasses.")

# # CRNN 的參數都還沒設定！
# class CRNN(nn.Module):
#     def __init__(self, ):
#         super(CRNN, self).__init__()

#         self.cnn = nn.Sequential(
#             nn.Conv2d(in_channels = 1, out_channels = , kernel_size =, padding = ),
#             nn.BatchNorm2d(),
#             nn.ReLu(),
#             nn.MaxPool2d(kernel_size = 2)
#         )

#         self.rnn = nn.GRU(
#             input_size = ,
#             hidden_size = ,
#             num_layers = ,
#             batch_first = True,
#             bidirectional = True,           
#         )

#         self.fc = nn.Linear(in_features = ,out_features = )
    
#     def forward(self, x):
#         x = self.cnn(x)
# CRNN.py

import os
import numpy as np
import pandas as pd
from typing import List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

import utils

# parameter
MAX_MEL_FRAMES = 1000  # padding聲普圖用的，訓練的時候自行調整

CNN_INPUT_CHANNELS = 1
N_MELS = 128

CNN_CHANNELS = [32, 64, 128]
CNN_KERNEL_SIZE = (3, 3)
CNN_POOL_SIZE = (2, 2)

RNN_HIDDEN_SIZE = 128
RNN_NUM_LAYERS = 2
RNN_DROPOUT = 0.3

DROPOUT_RATE = 0.3 #CRNN的dropout rate，訓練的時候自行調整

INSTRUMENT_MAPPING_PATH = 'slakh-utils/midi_inst_values/general_midi_inst_0based.json' #utils生出來的json路徑

# 把樂器種類標上標籤
def get_target_instrument_classes(mapping_path: str) -> List[str]:
    """
    Loads the instrument mapping and extracts a sorted list of unique instrument class names.
    This list defines the order of classes for the multi-hot labels.
    """
    try:
        instruments_mapping = utils.read_instruments_class(mapping_path)

        classes = sorted(list(set(item['class'] for item in instruments_mapping.values())))
        return classes
    except FileNotFoundError:
        print(f"Error: Instrument mapping file not found at {mapping_path}")
        return []
    except Exception as e:
        print(f"Error reading instrument mapping file: {e}")
        return []


class MusicInstrumentDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        instrument_mapping_path: str = INSTRUMENT_MAPPING_PATH,
        max_mel_frames: int = MAX_MEL_FRAMES
    ):
        """
        Args:
            dataframe: DataFrame returned by utils.load_dataset.
            instrument_mapping_path: Path to the JSON instrument mapping file.
            max_mel_frames: The target number of time frames for spectrogram padding/truncation.
        """
        self.dataframe = dataframe
        self.max_mel_frames = max_mel_frames

        # Load and define target instrument classes here
        self.instrument_classes = get_target_instrument_classes(instrument_mapping_path)
        if not self.instrument_classes:
             raise ValueError(f"Could not load or find instrument classes from {instrument_mapping_path}")

        self.mlb = MultiLabelBinarizer(classes=self.instrument_classes)

        self.data = [] # Stores Mel spectrograms
        self.labels = [] # Stores multi-hot labels
        self.load_data() # Load data during initialization

        self.n_mels = N_MELS
        if len(self.data) > 0:
            self.n_mels = self.data[0].shape[1] # Shape is [1, N_MELS, MAX_MEL_FRAMES]

        print(f"Dataset initialized with N_MELS={self.n_mels}, MAX_MEL_FRAMES={self.max_mel_frames}, num_classes={len(self.instrument_classes)}")


    def load_data(self):
        """
        Processes the dataframe, reads audio and MIDI, computes features and labels.
        Note: This loads all data into memory. For large datasets, consider loading on the fly in __getitem__.
        """
        print(f"Loading and preprocessing data for {len(self.dataframe)} tracks...")

        # Load instrument mapping once
        try:
            instruments_mapping = utils.read_instruments_class(INSTRUMENT_MAPPING_PATH)
        except Exception as e:
            print(f"Could not load instrument mapping: {e}")
            instruments_mapping = {} # Use empty dict if loading fails

        for index, row in tqdm(self.dataframe.iterrows(), total=len(self.dataframe), desc="Processing tracks"):
            mix_path = row['mix']
            midi_files = row['midis']

            if mix_path is None or not os.path.exists(mix_path):
                continue

            if not midi_files:
                 continue

            # Process Audio
            try:
                # Use utils.read_flac which returns dB Mel spectrogram (N_MELS, time_frames)
                mel_spec_db = utils.read_flac(mix_path)

                # Pad or truncate the time dimension (axis=1)
                current_frames = mel_spec_db.shape[1]
                if current_frames > self.max_mel_frames:
                    # Truncate time dimension
                    mel_spec_processed = mel_spec_db[:, :self.max_mel_frames]
                else:
                    # Pad time dimension with a low dB value
                    padding_width = ((0, 0), (0, self.max_mel_frames - current_frames))
                    mel_spec_processed = np.pad(mel_spec_db, padding_width, mode='constant', constant_values=-100) # Pad with a low dB value

                # PyTorch Conv2d expects [C, H, W]. Mel spec is [Freq, Time].
                # So reshape to [1, N_MELS, max_mel_frames]
                mel_spec_processed = np.expand_dims(mel_spec_processed, axis=0) # Add channel dimension

            except Exception as e:
                # print(f"Error processing audio for track {row['track']}: {e}")
                continue # Skip this track if audio processing fails


            # Process Labels from MIDI
            track_instruments_classes = set()
            for midi_path in midi_files:
                 if not os.path.exists(midi_path):
                     continue
                 try:
                     midi_data = utils.read_midi(midi_path)
                     for instrument in midi_data.instruments:
                         program_str = str(instrument.program)
                         if program_str in instruments_mapping:
                              mapped_class = instruments_mapping[program_str]['class']
                              # Only add the class if it's in our target list
                              if mapped_class in self.instrument_classes:
                                   track_instruments_classes.add(mapped_class)
                 except Exception as e:
                    # print(f"Error processing MIDI for track {row['track']} ({midi_path}): {e}")
                    continue # Continue to next MIDI file if one fails

            # Create multi-hot label vector
            if not track_instruments_classes:
                # If no valid instruments were found in MIDI that map to target classes, create an all-zero label
                 label_vector = np.zeros(len(self.instrument_classes), dtype=np.float32)
            else:
                # Use the fitted MultiLabelBinarizer
                label_vector = self.mlb.fit_transform([list(track_instruments_classes)])[0]
                label_vector = label_vector.astype(np.float32) # Ensure float type for PyTorch

            # Append the processed data and label if audio processing was successful
            self.data.append(mel_spec_processed)
            self.labels.append(label_vector)

        print(f"Finished data loading and preprocessing. Found {len(self.data)} valid samples.")
        # Convert lists to numpy arrays
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        print(f"Dataset shapes: X={self.data.shape}, y={self.labels.shape}")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a single sample (Mel spectrogram) and its label.
        Converts numpy arrays to PyTorch tensors.
        """
        mel_spec = self.data[idx]
        label = self.labels[idx]

        # Convert numpy arrays to PyTorch tensors
        # mel_spec_tensor shape: [1, N_MELS, MAX_MEL_FRAMES] (already added channel in load_data)
        mel_spec_tensor = torch.from_numpy(mel_spec)
        label_tensor = torch.from_numpy(label)

        # No transform used in this example, but can be added here

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
             #padding
             padding = (CNN_KERNEL_SIZE[0]//2, CNN_KERNEL_SIZE[1]//2)
             cnn_layers.append(nn.Conv2d(in_channels=in_c, out_channels=out_c,
                                         kernel_size=CNN_KERNEL_SIZE, padding=padding))
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
        rnn_input_size = cnn_output_shape[1] * cnn_output_shape[2] # channels * pooled_n_mels
        pooled_time_steps = cnn_output_shape[3] # pooled_time_frames

        print(f"Calculated CNN output shape: {cnn_output_shape}")
        print(f"Calculated RNN input sequence length: {pooled_time_steps}")
        print(f"Calculated RNN input features per step: {rnn_input_size}")

        self.rnn = nn.GRU(
            input_size = rnn_input_size, # Features per time step
            hidden_size = RNN_HIDDEN_SIZE,
            num_layers = RNN_NUM_LAYERS,
            batch_first = True, # Input/Output Tensors are (batch, seq, feature)
            bidirectional = True,
            dropout = RNN_DROPOUT if RNN_NUM_LAYERS > 1 else 0,
        )

        # Fully Connected Layers
        rnn_output_size = RNN_HIDDEN_SIZE * 2
        self.fc = nn.Sequential(
            nn.Linear(rnn_output_size, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(128, num_classes)
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
        # Output shape: (batch_size, num_classes)

        # Apply Sigmoid activation for multi-label classification
        output = torch.sigmoid(output)

        return output