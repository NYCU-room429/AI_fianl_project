# data_loader.py
import os
import torch
from torch.utils.data import Dataset
from joblib import Parallel, delayed
import numpy as np
import librosa
import random
from tqdm import tqdm
from loguru import logger # Import for internal logging within this module if needed

import config
from utils import get_rendered_stems, generate_mel_spectrogram, extract_midi_labels

# _process_single_track_for_paths: remains mostly the same,
# but its print statements will act as logs if logger isn't passed.
# No changes needed here for the "logger only in main" request,
# as it already uses print for errors during parallel execution.

def _process_single_track_for_paths(track_dir_path, instrument_names_vocab, is_for_test_set):
    local_file_tuples = []
    mix_flac_path = os.path.join(track_dir_path, "mix.flac")
    mix_midi_path = os.path.join(track_dir_path, "all_src.mid")
    metadata_path = os.path.join(track_dir_path, "metadata.yaml")

    if os.path.exists(mix_flac_path) and os.path.exists(mix_midi_path):
        try:
            audio_duration = librosa.get_duration(path=mix_flac_path, sr=config.SAMPLE_RATE)
            if audio_duration >= config.MIN_AUDIO_DURATION_SEC:
                local_file_tuples.append((mix_flac_path, mix_midi_path, audio_duration))
        except Exception as e:
            print(f"Warning [data_loader._process_single_track]: Skipping mix {mix_flac_path} due to librosa error: {e}")
            pass

    if is_for_test_set: return local_file_tuples

    stems_audio_dir = os.path.join(track_dir_path, "stems")
    stems_midi_dir = os.path.join(track_dir_path, "MIDI")
    if os.path.exists(stems_audio_dir) and os.path.exists(stems_midi_dir) and os.path.exists(metadata_path):
        rendered_stem_names = get_rendered_stems(metadata_path)
        for stem_base_name in rendered_stem_names:
            stem_flac_path = os.path.join(stems_audio_dir, f"{stem_base_name}.flac")
            stem_midi_path = os.path.join(stems_midi_dir, f"{stem_base_name}.mid")
            if os.path.exists(stem_flac_path) and os.path.exists(stem_midi_path):
                try:
                    audio_duration = librosa.get_duration(path=stem_flac_path, sr=config.SAMPLE_RATE)
                    if audio_duration >= config.MIN_AUDIO_DURATION_SEC:
                        local_file_tuples.append((stem_flac_path, stem_midi_path, audio_duration))
                except Exception as e:
                    print(f"Warning [data_loader._process_single_track]: Skipping stem {stem_flac_path} due to librosa error: {e}")
                    pass
    return local_file_tuples


def collect_all_file_tuples(dataset_base_dir, instrument_names_vocab, is_for_test_set=False): # Removed logger_obj
    all_track_dirs = [os.path.join(dataset_base_dir, d) for d in os.listdir(dataset_base_dir)
                      if os.path.isdir(os.path.join(dataset_base_dir, d)) and d.startswith("Track")]
    if not all_track_dirs:
        # Use global loguru logger if available, or print
        logger.error(f"No track directories found in {dataset_base_dir}")
        return []

    desc_text = f"Collecting files from {os.path.basename(dataset_base_dir)}"
    results = Parallel(n_jobs=config.NUM_WORKERS, prefer="threads")(
        delayed(_process_single_track_for_paths)(
            track_path, instrument_names_vocab, is_for_test_set
        ) for track_path in tqdm(all_track_dirs, desc=desc_text, ncols=100, leave=False)
    )
    final_file_tuples = []
    if results:
        for track_tuples in results:
            if track_tuples: final_file_tuples.extend(track_tuples)
    
    logger.info(f"Collected {len(final_file_tuples)} valid audio-MIDI pairs from {os.path.basename(dataset_base_dir)}.")
    return final_file_tuples


class InstrumentDataset(Dataset):
    def __init__(self, file_tuples, instrument_names_vocab, is_train): # Removed logger_obj
        self.file_tuples = file_tuples
        self.instrument_names_vocab = instrument_names_vocab
        self.is_train = is_train
        self.num_classes = len(instrument_names_vocab)
        # self.logger = logger # Use globally imported loguru.logger for internal messages

        if not self.file_tuples:
             logger.warning("InstrumentDataset initialized with zero file tuples!") # Use global logger

    def __len__(self):
        return len(self.file_tuples)

    def __getitem__(self, idx):
        # ... (same __getitem__ logic as "Further Enhanced" version)
        audio_path, midi_path, audio_duration_sec = self.file_tuples[idx]
        if self.is_train:
            max_start = audio_duration_sec - config.SEGMENT_SEC
            current_selected_start_sec = random.uniform(0, max_start) if max_start > 0 else 0.0
        else:
            current_selected_start_sec = max(0.0, (audio_duration_sec - config.SEGMENT_SEC) / 2.0)
        current_selected_start_sec = min(current_selected_start_sec, max(0.0, audio_duration_sec - config.SEGMENT_SEC))
        
        debug_identifier = f"{os.path.splitext(os.path.basename(audio_path))[0]}_{idx}" if self.is_train and random.random() < 0.0001 else None

        mel_spec = generate_mel_spectrogram(
            audio_path, sr=config.SAMPLE_RATE, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS, segment_sec=config.SEGMENT_SEC,
            num_target_frames=config.NUM_SPECTROGRAM_FRAMES,
            selected_start_sec=current_selected_start_sec, is_train=self.is_train,
            freq_masks=config.SPECAUG_FREQ_MASKS if self.is_train else 0,
            time_masks=config.SPECAUG_TIME_MASKS if self.is_train else 0,
            freq_max_width=config.SPECAUG_FREQ_MAX_WIDTH if self.is_train else 0,
            time_max_width=config.SPECAUG_TIME_MAX_WIDTH if self.is_train else 0,
            debug_id=debug_identifier
        )
        if mel_spec is None:
            logger.error(f"Mel spectrogram is None for {audio_path} (start_sec={current_selected_start_sec:.2f}). Using zeros.")
            mel_spec = np.full((config.N_MELS, config.NUM_SPECTROGRAM_FRAMES), -80.0, dtype=np.float32)

        labels = extract_midi_labels(
            midi_path, self.instrument_names_vocab, config.SEGMENT_SEC,
            config.NUM_SPECTROGRAM_FRAMES, selected_start_sec=current_selected_start_sec,
            debug_id=debug_identifier
        )
        mean = np.mean(mel_spec); std = np.std(mel_spec)
        if std > 1e-6: mel_spec = (mel_spec - mean) / std
        else: mel_spec = mel_spec - mean 
        
        return torch.from_numpy(mel_spec.copy()).unsqueeze(0), torch.from_numpy(labels.copy())


    def get_positive_weights(self, num_samples_for_calc=3000): # Removed logger_obj
        logger.info(f"Calculating positive weights using up to {num_samples_for_calc} samples...") # Use global logger
        all_labels_list = []
        actual_num_samples = min(len(self.file_tuples), num_samples_for_calc)
        if actual_num_samples == 0:
            logger.warning("No files to sample for pos_weight. Defaulting to ones.")
            return torch.ones(self.num_classes, dtype=torch.float32)

        sample_indices = random.sample(range(len(self.file_tuples)), actual_num_samples)
        for idx in tqdm(sample_indices, desc="Analyzing labels for pos_weight", ncols=100, leave=False):
            _, midi_path, audio_duration_sec = self.file_tuples[idx]
            selected_start_sec = max(0, (audio_duration_sec - config.SEGMENT_SEC) / 2.0)
            labels = extract_midi_labels(
                midi_path, self.instrument_names_vocab, config.SEGMENT_SEC,
                config.NUM_SPECTROGRAM_FRAMES, selected_start_sec
            )
            all_labels_list.append(labels)
        
        if not all_labels_list: logger.warning("No labels collected for pos_weight..."); return torch.ones(self.num_classes, dtype=torch.float32)
        all_labels_np = np.concatenate(all_labels_list, axis=0)
        if all_labels_np.shape[0] == 0: logger.warning("Concatenated labels for pos_weight empty..."); return torch.ones(self.num_classes, dtype=torch.float32)

        num_pos = np.sum(all_labels_np, axis=0) 
        num_total_frames = all_labels_np.shape[0]
        num_neg = num_total_frames - num_pos
        pos_weight = num_neg / (num_pos + 1e-7)
        pos_weight = np.clip(pos_weight, config.POS_WEIGHT_CLAMP_MIN, config.POS_WEIGHT_CLAMP_MAX)
        pos_weight_tensor = torch.from_numpy(pos_weight.astype(np.float32))
        logger.info(f"Calculated pos_weights: {pos_weight_tensor.tolist()}")
        return pos_weight_tensor