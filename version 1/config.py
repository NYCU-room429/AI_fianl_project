# config.py
import torch
import os

# --- Environment ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 2  # Start with fewer workers for easier debugging, increase later
SEED = 42 # For reproducibility

# --- Dataset Paths ---
# !!! IMPORTANT: Update these paths to your actual dataset locations !!!
SLAKH_ROOT_DIR = r"D:\\AI_fianl_project\\slakh2100_flac_redux\\slakh2100_flac_redux"
TRAIN_DIR_NAME = "train"
VALID_DIR_NAME = "validation"
TEST_DIR_NAME = "test"
CLASS_JSON_PATH = "D:\\AI_fianl_project\\slakh-utils\\midi_inst_values\\general_midi_inst_0based.json"

# --- Audio Processing ---
SAMPLE_RATE = 22050
SEGMENT_SEC = 10.0
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 220 # ~10ms hop; SR * SEG_SEC / HOP_LENGTH + 1 = NUM_FRAMES
NUM_SPECTROGRAM_FRAMES = int(SAMPLE_RATE * SEGMENT_SEC / HOP_LENGTH) + 1
# Ensure this matches calculation: NUM_SPECTROGRAM_FRAMES should be around 1001 for 10s, 22050 SR, 220 hop

# SpecAugment parameters
SPECAUG_FREQ_MASKS = 2
SPECAUG_TIME_MASKS = 2
SPECAUG_FREQ_MAX_WIDTH = 20
SPECAUG_TIME_MAX_WIDTH = int(NUM_SPECTROGRAM_FRAMES * 0.1)

# --- Model Architecture ---
MODEL_CNN_FILTERS = [64, 128, 256, 512]
MODEL_RNN_UNITS = 256
MODEL_ATTENTION_HEADS = 4
MODEL_DROPOUT_RATE = 0.35 # Slightly increased dropout

# --- Training ---
BATCH_SIZE = 8 # Start smaller for debugging and if memory is an issue
LEARNING_RATE = 3e-4 # Common starting point for AdamW
WEIGHT_DECAY = 1e-5
EPOCHS = 70 # Allow more epochs
GRAD_CLIP_NORM = 1.0
POS_WEIGHT_CLAMP_MIN = 1.0
POS_WEIGHT_CLAMP_MAX = 12.0 # Allow slightly higher clamping
LR_SCHEDULER_PATIENCE = 7 # More patience for LR scheduler
LR_SCHEDULER_FACTOR = 0.5
EARLY_STOPPING_PATIENCE = 15 # More patience for early stopping

# --- Output ---
# !!! IMPORTANT: Update this path !!!
MODEL_SAVE_DIR = "D:\\AI_fianl_project\\test_CRNN" # New dir for new run
MODEL_FILENAME = "instr_recognizer_best_v2.pth"
THRESHOLDS_FILENAME = "instr_recognizer_thresholds_v2.npy"
LOG_FILENAME = "training_log_v2.log"
PLOT_FILENAME = "training_plot_v2.png"

# --- GUI ---
GUI_DISPLAY_PROB_THRESHOLD = 0.1

# --- Pre-filtering data ---
MIN_AUDIO_DURATION_SEC = SEGMENT_SEC + 0.5 # Ensure audio is clearly longer than segment