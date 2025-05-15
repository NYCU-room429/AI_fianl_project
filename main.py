import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import librosa
import scipy.stats as stats
import matplotlib.pyplot as plt
import utils

if __name__ == "__main__":
    train_dir = "slakh2100_flac_redux\\slakh2100_flac_redux\\train"
    train_data = utils.load_dataset(train_dir)
    mel_spectrogram = utils.read_flacs(train_data.iloc[0]["mix"])
    utils.read_instruments_class(
        "slakh-utils\\midi_inst_values\\general_midi_inst_0based.json"
    )

    # plt.figure(figsize=(10, 4))
    # librosa.display.specshow(
    #     mel_spectrogram, sr=22050, hop_length=512, x_axis="time", y_axis="mel"
    # )
    # plt.colorbar(format="%+2.0f dB")
    # plt.title("Mel-frequency Spectrogram")
    # plt.tight_layout()
    # plt.show()
