import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext, font as tkfont
import torch
import numpy as np
import librosa
import os
import threading

from CRNN import CRNN
from utils import get_all_class

MODEL_PATH = "best_model_class_based.pth"
THRESHOLDS_PATH = "best_thresholds_class_based.npy"
CLASS_JSON_PATH = os.path.join(
    "slakh-utils", "midi_inst_values", "general_midi_inst_0based.json"
)

N_MELS = 96
N_FFT = 512
SR = 22050
TARGET_FRAMES = 1000
SEGMENT_SEC_FOR_MODEL = 100

loaded_model = None
instrument_classes_list = None
final_thresholds = None
device = None
status_bar = None

BG_COLOR = "#ECECEC"
FG_COLOR = "#333333"
ACCENT_COLOR = "#4A90E2"
BUTTON_FG_COLOR = "#FFFFFF"
FRAME_BG_COLOR = "#F5F5F5"
STATUS_ERROR_FG = "#D32F2F"
STATUS_SUCCESS_FG = "#388E3C"
FONT_FAMILY = "Segoe UI"
FONT_SIZE_NORMAL = 10
FONT_SIZE_LARGE = 12


def get_font(size=FONT_SIZE_NORMAL, weight="normal"):
    try:
        return tkfont.Font(family=FONT_FAMILY, size=size, weight=weight)
    except tk.TclError:  # Fallback font
        return tkfont.Font(family="Arial", size=size, weight=weight)


def setup_styles():
    style = ttk.Style()
    style.theme_use("clam")  # 'clam', 'alt', 'default', 'classic' are common options

    style.configure(".", background=BG_COLOR, foreground=FG_COLOR, font=get_font())

    style.configure("TFrame", background=FRAME_BG_COLOR)
    style.configure("Background.TFrame", background=BG_COLOR)

    style.configure(
        "TButton",
        font=get_font(weight="bold"),
        background=ACCENT_COLOR,
        foreground=BUTTON_FG_COLOR,
        padding=(10, 5),
    )
    style.map(
        "TButton",
        background=[("active", "#357ABD"), ("disabled", "#B0BEC5")],
        foreground=[("disabled", "#757575")],
    )

    style.configure("TLabel", background=FRAME_BG_COLOR, font=get_font())
    style.configure(
        "Status.TLabel", background=BG_COLOR, font=get_font(size=FONT_SIZE_NORMAL - 1)
    )
    style.configure(
        "FilePath.TLabel",
        background=FRAME_BG_COLOR,
        font=get_font(),
        foreground="#555555",
    )

    style.configure("TLabelFrame", background=FRAME_BG_COLOR, padding=10)
    style.configure(
        "TLabelFrame.Label",
        background=FRAME_BG_COLOR,
        foreground=ACCENT_COLOR,
        font=get_font(size=FONT_SIZE_LARGE, weight="bold"),
    )


def set_status(message, is_error=False, is_success=False):
    global status_text, status_bar

    if status_text:
        status_text.set(message)

    if status_bar:
        if is_error:
            status_bar.config(foreground=STATUS_ERROR_FG)
        elif is_success:
            status_bar.config(foreground=STATUS_SUCCESS_FG)
        else:
            status_bar.config(foreground=FG_COLOR)


def load_resources():
    """Loads the model, class definitions, and thresholds."""
    global loaded_model, instrument_classes_list, final_thresholds, device

    actual_class_json_path = CLASS_JSON_PATH
    if not os.path.isabs(actual_class_json_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        actual_class_json_path = os.path.join(script_dir, CLASS_JSON_PATH)

    model_path_abs = MODEL_PATH
    if not os.path.isabs(model_path_abs):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path_abs = os.path.join(script_dir, MODEL_PATH)

    thresholds_path_abs = THRESHOLDS_PATH
    if not os.path.isabs(thresholds_path_abs):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        thresholds_path_abs = os.path.join(script_dir, THRESHOLDS_PATH)

    if not os.path.exists(model_path_abs):
        set_status(f"Error: Model file not found at {model_path_abs}", is_error=True)
        return False
    if not os.path.exists(actual_class_json_path):
        set_status(
            f"Error: Class JSON file not found: {actual_class_json_path}", is_error=True
        )
        return False
    if not os.path.exists(thresholds_path_abs):
        set_status(
            f"Error: Thresholds file not found at {thresholds_path_abs}", is_error=True
        )
        return False

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_status(f"Using device: {device}")

        instrument_classes_list = get_all_class(actual_class_json_path)
        num_classes = len(instrument_classes_list)

        loaded_model = CRNN(num_classes=num_classes).to(device)
        loaded_model.load_state_dict(torch.load(model_path_abs, map_location=device))
        loaded_model.eval()

        loaded_thresholds = np.load(thresholds_path_abs)
        if len(loaded_thresholds) == num_classes:
            final_thresholds = loaded_thresholds
        else:
            set_status(f"Warning: Thresholds size mismatch. Using default 0.5.")
            final_thresholds = np.full(num_classes, 0.5)

        set_status("Model and resources loaded successfully.", is_success=True)
        return True
    except Exception as e:
        set_status(f"Error loading resources: {e}", is_error=True)
        return False


def generate_mel_for_prediction(audio_path):
    """
    Generates a mel spectrogram from an audio file for prediction.
    Handles arbitrary audio lengths by taking the first SEGMENT_SEC_FOR_MODEL seconds.
    """
    try:
        waveform, sample_rate = librosa.load(audio_path, sr=SR, mono=True)
    except Exception as e:
        set_status(f"Error loading audio: {e}", is_error=True)
        return None

    waveform_np = waveform

    min_len_for_fft = N_FFT
    if len(waveform_np) < min_len_for_fft:
        padding = min_len_for_fft - len(waveform_np)
        waveform_np = np.pad(waveform_np, (0, padding), "constant")

    target_len_samples = SR * SEGMENT_SEC_FOR_MODEL

    if len(waveform_np) > target_len_samples:
        src_segment = waveform_np[:target_len_samples]
    elif len(waveform_np) < target_len_samples:
        padding = target_len_samples - len(waveform_np)
        src_segment = np.pad(waveform_np, (0, padding), "constant")
    else:
        src_segment = waveform_np

    hop_length = (
        (len(src_segment) - N_FFT) // (TARGET_FRAMES - 1)
        if TARGET_FRAMES > 1
        else N_FFT
    )
    hop_length = max(1, hop_length)

    mel = librosa.feature.melspectrogram(
        y=src_segment, sr=SR, n_fft=N_FFT, hop_length=hop_length, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    if mel_db.shape[1] < TARGET_FRAMES:
        pad_width = TARGET_FRAMES - mel_db.shape[1]
        mel_db = np.pad(
            mel_db,
            ((0, 0), (0, pad_width)),
            mode="constant",
            constant_values=np.min(mel_db),
        )
    elif mel_db.shape[1] > TARGET_FRAMES:
        mel_db = mel_db[:, :TARGET_FRAMES]

    mel_mean = mel_db.mean()
    mel_std = mel_db.std()
    if mel_std > 0:
        mel_db = (mel_db - mel_mean) / mel_std

    return torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


def predict_audio_file(audio_path):
    global loaded_model, instrument_classes_list, final_thresholds, device, predict_button, results_text

    # Corrected check:
    if (
        loaded_model is None
        or instrument_classes_list is None
        or final_thresholds is None
    ):
        set_status(
            "Error: Resources not loaded. Click 'Load Model' first.", is_error=True
        )
        if predict_button:
            predict_button.config(state=tk.NORMAL)
        return

    set_status(f"Processing {os.path.basename(audio_path)}...")
    if results_text:
        results_text.config(state=tk.NORMAL)  # Enable for modification
        results_text.delete(1.0, tk.END)

    mel_tensor = generate_mel_for_prediction(audio_path)
    if mel_tensor is None:
        if predict_button:
            predict_button.config(state=tk.NORMAL)
        if results_text:
            results_text.config(state=tk.DISABLED)  # Disable after clearing
        return

    mel_tensor = mel_tensor.to(device)

    with torch.no_grad():
        outputs = loaded_model(mel_tensor)  # (batch=1, time_frames, num_classes)

    preds_proba = torch.sigmoid(outputs)
    current_preds_active_frames = (
        (
            preds_proba.squeeze(0)
            > torch.tensor(final_thresholds, device=device).unsqueeze(0)
        )
        .cpu()
        .numpy()
    )

    pred_inst_present_indices = np.where(current_preds_active_frames.any(axis=0))[0]
    predicted_instrument_names_present = [
        instrument_classes_list[k] for k in pred_inst_present_indices
    ]

    pred_frames_per_class = current_preds_active_frames.sum(axis=0)

    longest_duration_pred_names = ["None"]  # Default
    max_pred_frames = 0
    if pred_frames_per_class.any():
        max_pred_frames = np.max(pred_frames_per_class)
        longest_duration_pred_indices = np.where(
            pred_frames_per_class == max_pred_frames
        )[0]
        longest_duration_pred_names = [
            instrument_classes_list[k] for k in longest_duration_pred_indices
        ]

    if results_text:
        if predicted_instrument_names_present:
            results_text.insert(
                tk.END, "Predicted Instrument Classes (Present):\n", "header_bold"
            )
            for name in sorted(predicted_instrument_names_present):
                results_text.insert(tk.END, f"  • {name}\n", "bullet_point")
            results_text.insert(tk.END, "\n")
        else:
            results_text.insert(
                tk.END, "No instrument classes detected as present.\n\n", "info"
            )

        results_text.insert(
            tk.END, "Predicted Longest Duration Instrument(s):\n", "header_bold"
        )
        for name in sorted(longest_duration_pred_names):
            results_text.insert(tk.END, f"  • {name}", "bullet_point")
            if name != "None":
                estimated_duration_sec = (
                    max_pred_frames / TARGET_FRAMES
                ) * SEGMENT_SEC_FOR_MODEL
                results_text.insert(
                    tk.END,
                    f" (active for {estimated_duration_sec:.1f}s estimate)\n",
                    "duration_info",
                )
            else:
                results_text.insert(tk.END, "\n", "bullet_point")

        results_text.config(state=tk.DISABLED)

    set_status(
        f"Prediction complete for {os.path.basename(audio_path)}.", is_success=True
    )
    if predict_button:
        predict_button.config(state=tk.NORMAL)


def select_file():
    global selected_file_path
    filepath = filedialog.askopenfilename(
        title="Select Audio File",
        filetypes=(("Audio Files", "*.wav *.mp3 *.flac"), ("All files", "*.*")),
    )
    if filepath:
        if selected_file_path:
            selected_file_path.set(filepath)
        set_status(f"Selected: {os.path.basename(filepath)}")
        if results_text:
            results_text.config(state=tk.NORMAL)
            results_text.delete(1.0, tk.END)
            results_text.config(state=tk.DISABLED)


def start_prediction_thread():
    global selected_file_path, predict_button
    audio_file = selected_file_path.get() if selected_file_path else None
    if not audio_file:
        set_status("Please select an audio file first.", is_error=True)
        return
    if not os.path.exists(audio_file):
        set_status("Error: Selected audio file does not exist.", is_error=True)
        return

    if predict_button:
        predict_button.config(state=tk.DISABLED)
    set_status("Starting prediction...")
    thread = threading.Thread(target=predict_audio_file, args=(audio_file,))
    thread.daemon = True
    thread.start()


def create_gui():
    global selected_file_path, status_text, results_text, predict_button, load_model_button, status_bar

    root = tk.Tk()
    root.title("Audio Instrument Recognition")
    root.geometry("700x600")
    root.configure(bg=BG_COLOR)

    setup_styles()

    selected_file_path = tk.StringVar()
    status_text = tk.StringVar()

    main_frame = ttk.Frame(root, padding="15 15 15 15", style="Background.TFrame")
    main_frame.pack(fill=tk.BOTH, expand=True)

    file_frame = ttk.Frame(main_frame, padding="0 0 0 10", style="TFrame")
    file_frame.pack(fill=tk.X, pady=(0, 10))

    select_button = ttk.Button(
        file_frame, text="Select Audio File", command=select_file
    )
    select_button.pack(side=tk.LEFT, padx=(0, 10))

    file_label_frame = ttk.Frame(
        file_frame, style="TFrame", relief="sunken", borderwidth=1
    )
    file_label_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
    file_label = ttk.Label(
        file_label_frame,
        textvariable=selected_file_path,
        style="FilePath.TLabel",
        padding=5,
        anchor="w",
    )
    file_label.pack(fill=tk.X, expand=True)

    action_frame = ttk.Frame(main_frame, padding="0 0 0 10", style="TFrame")
    action_frame.pack(fill=tk.X, pady=(0, 15))

    load_model_button = ttk.Button(
        action_frame, text="Load Model & Resources", command=load_resources
    )
    load_model_button.pack(side=tk.LEFT, padx=(0, 10))

    predict_button = ttk.Button(
        action_frame, text="Predict Instruments", command=start_prediction_thread
    )
    predict_button.pack(side=tk.LEFT)

    results_frame = ttk.LabelFrame(main_frame, text="Prediction Results")
    results_frame.pack(fill=tk.BOTH, expand=True)

    results_text = scrolledtext.ScrolledText(
        results_frame,
        wrap=tk.WORD,
        height=12,
        font=get_font(),
        bg="#FFFFFF",
        fg=FG_COLOR,
        relief=tk.FLAT,
        borderwidth=0,
    )
    results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    results_text.tag_configure(
        "header_bold",
        font=get_font(size=FONT_SIZE_NORMAL, weight="bold"),
        foreground=ACCENT_COLOR,
    )
    results_text.tag_configure(
        "bullet_point", font=get_font(), lmargin1=20, lmargin2=20
    )
    results_text.tag_configure(
        "info", font=get_font(size=FONT_SIZE_NORMAL - 1), foreground="#555555"
    )
    results_text.tag_configure(
        "duration_info", font=get_font(size=FONT_SIZE_NORMAL - 1), foreground="#227722"
    )
    results_text.config(state=tk.DISABLED)

    status_bar_frame = ttk.Frame(
        root, style="Background.TFrame", relief="sunken", borderwidth=1
    )
    status_bar_frame.pack(side=tk.BOTTOM, fill=tk.X)
    status_bar = ttk.Label(
        status_bar_frame,
        textvariable=status_text,
        style="Status.TLabel",
        padding=(5, 3),
        anchor=tk.W,
    )
    status_bar.pack(fill=tk.X)

    set_status("Load model to begin.")
    if load_resources():
        if predict_button:
            predict_button.config(state=tk.NORMAL)
    else:
        set_status(
            "Failed to load resources on startup. Check paths and files.", is_error=True
        )
        if predict_button:
            predict_button.config(state=tk.DISABLED)  # Explicitly disable

    root.mainloop()


if __name__ == "__main__":
    script_dir_defined = True
    try:
        _ = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir_defined = False
        print(
            "Warning: __file__ not defined. Relative paths for resources might be incorrect if not run as a script."
        )
        print(
            "Ensure MODEL_PATH, THRESHOLDS_PATH, and CLASS_JSON_PATH are absolute or correctly relative to the CWD."
        )

    create_gui()
