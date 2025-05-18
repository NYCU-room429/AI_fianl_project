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
import os
from multiprocessing import Pool, cpu_count
import scipy.stats as stats
import matplotlib.pyplot as plt
import utils


def process_track_data(row_dict):
    track_id = row_dict["track"]
    mix_path = row_dict["mix"]
    metadata_path = row_dict["metadata"]
    stem_file_paths = row_dict["stems"]
    midi_file_paths = row_dict["midis"]

    track_dict = {}
    track_dict[track_id] = {
        "mix": utils.read_flac(mix_path),
        "metadata": utils.read_metadata(metadata_path),
        "stems": {
            os.path.splitext(os.path.basename(stem_flac))[0]: utils.read_flac(stem_flac)
            for stem_flac in stem_file_paths
        },
        "midis": {
            os.path.splitext(os.path.basename(midi_mid))[0]: utils.read_midi(midi_mid)
            for midi_mid in midi_file_paths
        },
    }
    return track_dict


if __name__ == "__main__":
    instruments_class = utils.read_instruments_class(
        "slakh-utils\\midi_inst_values\\general_midi_inst_0based.json"
    )

    print("================")
    print(len(instruments_class))
    print("================")
    train_dir = "slakh2100_flac_redux\\slakh2100_flac_redux\\train"
    train_data = utils.load_dataset(train_dir)

    row_dicts = train_data.to_dict(orient="records")

    # with Pool(processes=cpu_count()) as pool:
    #     results = list(
    #         tqdm(
    #             pool.imap(process_track_data, row_dicts),
    #             total=len(row_dicts),
    #         )
    #     )

    # track_dicts = [r for r in results if r is not None]

    # final_dict = {}
    # for d in track_dicts:
    #     final_dict.update(d)

    print(process_track_data(row_dicts[0]))

    track_dict = process_track_data(row_dicts[0])
    track_id = list(track_dict.keys())[0]
    mel_spectrogram = track_dict[track_id]["mix"]

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        mel_spectrogram, sr=22050, hop_length=512, x_axis="time", y_axis="mel"
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel-frequency Spectrogram")
    plt.tight_layout()
    plt.show()
