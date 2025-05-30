# gui.py
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading
import os
import numpy as np
import torch
import librosa

import config # Your project's config.py
from utils import generate_mel_spectrogram, get_all_class_names # For prediction
from C_utils import InstrumentCRNN # Your model class

PREDICTION_MODEL = None
INSTRUMENT_NAMES_VOCAB = None
OPTIMAL_THRESHOLDS = None
MODEL_LOADED_EVENT = threading.Event()
# Global references to GUI elements that need updating from threads
gui_status_var = None
gui_results_text = None
gui_predict_button = None
gui_browse_button = None

def load_model_resources_threaded_for_gui():
    global PREDICTION_MODEL, INSTRUMENT_NAMES_VOCAB, OPTIMAL_THRESHOLDS, gui_status_var
    
    try:
        if gui_status_var: gui_status_var.set("Loading vocabulary...")
        INSTRUMENT_NAMES_VOCAB = get_all_class_names(config.CLASS_JSON_PATH)
        num_classes = len(INSTRUMENT_NAMES_VOCAB)
        
        if gui_status_var: gui_status_var.set("Loading model architecture...")
        PREDICTION_MODEL = InstrumentCRNN(
            num_classes=num_classes, n_mels=config.N_MELS,
            num_spec_frames=config.NUM_SPECTROGRAM_FRAMES,
            cnn_filters=config.MODEL_CNN_FILTERS, rnn_units=config.MODEL_RNN_UNITS,
            attention_heads=config.MODEL_ATTENTION_HEADS, dropout_rate=0.0
        ).to(config.DEVICE)
        
        model_full_path = os.path.join(config.MODEL_SAVE_DIR, config.MODEL_FILENAME)
        if not os.path.exists(model_full_path):
            raise FileNotFoundError(f"Model file not found: {model_full_path}")
            
        if gui_status_var: gui_status_var.set(f"Loading model weights...")
        PREDICTION_MODEL.load_state_dict(torch.load(model_full_path, map_location=config.DEVICE))
        PREDICTION_MODEL.eval()
        
        if gui_status_var: gui_status_var.set("Loading optimal thresholds...")
        thresholds_full_path = os.path.join(config.MODEL_SAVE_DIR, config.THRESHOLDS_FILENAME)
        if not os.path.exists(thresholds_full_path):
            OPTIMAL_THRESHOLDS = np.full(num_classes, 0.5)
            if gui_status_var: gui_status_var.set(f"Thresholds file not found. Using 0.5. Ready.")
        else:
            OPTIMAL_THRESHOLDS = np.load(thresholds_full_path)
            if gui_status_var: gui_status_var.set("Model and resources loaded. Ready.")
            
        MODEL_LOADED_EVENT.set()

    except Exception as e:
        messagebox.showerror("Loading Error", f"Failed to load resources: {e}")
        if gui_status_var: gui_status_var.set(f"Error loading: {str(e)[:50]}")
        MODEL_LOADED_EVENT.set() # Set event even on error to unblock checks


def predict_for_gui_threaded(audio_path):
    global gui_status_var, gui_results_text, gui_predict_button, gui_browse_button
    
    if not MODEL_LOADED_EVENT.is_set() or PREDICTION_MODEL is None:
        if gui_results_text:
            gui_results_text.config(state=tk.NORMAL)
            gui_results_text.insert(tk.END, "Model is not loaded yet.\n")
            gui_results_text.config(state=tk.DISABLED)
        return

    if gui_status_var: gui_status_var.set(f"Processing {os.path.basename(audio_path)}...")
    if gui_predict_button: gui_predict_button.config(state=tk.DISABLED)
    if gui_browse_button: gui_browse_button.config(state=tk.DISABLED)
    
    try:
        audio_duration = librosa.get_duration(filename=audio_path, sr=config.SAMPLE_RATE)
        selected_start_sec = max(0, (audio_duration - config.SEGMENT_SEC) / 2.0)

        mel_spec = generate_mel_spectrogram(
            audio_path, sr=config.SAMPLE_RATE, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS, segment_sec=config.SEGMENT_SEC,
            num_target_frames=config.NUM_SPECTROGRAM_FRAMES,
            selected_start_sec=selected_start_sec, is_train=False
        )
        if mel_spec is None: raise ValueError("Spectrogram generation failed.")

        mean, std = np.mean(mel_spec), np.std(mel_spec)
        mel_spec = (mel_spec - mean) / (std + 1e-5)
        mel_spec_tensor = torch.from_numpy(mel_spec).unsqueeze(0).unsqueeze(0).to(config.DEVICE)

        with torch.no_grad():
            logits = PREDICTION_MODEL(mel_spec_tensor)
            probabilities = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        
        clip_max_probs = np.max(probabilities, axis=0)

        if gui_results_text:
            gui_results_text.config(state=tk.NORMAL)
            gui_results_text.delete(1.0, tk.END)
            gui_results_text.insert(tk.END, f"--- Predictions for: {os.path.basename(audio_path)} ---\n")
            gui_results_text.insert(tk.END, f"(Analyzed segment from {selected_start_sec:.2f}s, duration {config.SEGMENT_SEC:.1f}s)\n\n")
            
            detected_count = 0
            results_to_display = []
            for i, instr_name in enumerate(INSTRUMENT_NAMES_VOCAB):
                prob = clip_max_probs[i]
                is_detected = prob >= OPTIMAL_THRESHOLDS[i]
                if is_detected or prob >= config.GUI_DISPLAY_PROB_THRESHOLD:
                    results_to_display.append((instr_name, prob, is_detected, OPTIMAL_THRESHOLDS[i]))
                if is_detected: detected_count += 1
            
            results_to_display.sort(key=lambda x: x[1], reverse=True)

            if not results_to_display:
                gui_results_text.insert(tk.END, "No instruments detected above display/learned thresholds.\n")
            else:
                gui_results_text.insert(tk.END, f"Detected Instruments ({detected_count} above learned thresholds):\n")
                for name, prob, detected, learned_thresh in results_to_display:
                    status = "DETECTED" if detected else "" # Simpler status
                    indicator_char = "*" if detected else " "
                    gui_results_text.insert(tk.END, f" {indicator_char} {name:<25}: {prob:.3f} (Th: {learned_thresh:.2f})\n")
            gui_results_text.config(state=tk.DISABLED)
        if gui_status_var: gui_status_var.set("Prediction complete.")

    except Exception as e:
        messagebox.showerror("Prediction Error", f"Error during prediction: {e}")
        if gui_status_var: gui_status_var.set(f"Prediction error: {str(e)[:50]}")
    finally:
        if gui_predict_button: gui_predict_button.config(state=tk.NORMAL)
        if gui_browse_button: gui_browse_button.config(state=tk.NORMAL)


class InstrumentRecognizerApp:
    def __init__(self, root_tk):
        self.root = root_tk
        self.root.title("Instrument Recognizer")
        self.root.geometry("750x600")
        self.root.option_add("*Font", "Helvetica 10")

        self.style = ttk.Style(self.root)
        if os.name == 'nt': # Windows
            self.style.theme_use('vista')
        else: # MacOS/Linux
            self.style.theme_use('clam') 

        # --- Main Frame ---
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(expand=True, fill=tk.BOTH)

        # --- File Selection Row ---
        file_frame = ttk.Frame(main_frame)
        file_frame.pack(fill=tk.X, pady=(0,10))
        
        ttk.Label(file_frame, text="Audio File:").pack(side=tk.LEFT, padx=(0,5))
        self.filepath_var = tk.StringVar()
        filepath_entry = ttk.Entry(file_frame, textvariable=self.filepath_var, width=60, state="readonly")
        filepath_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        
        global gui_browse_button
        gui_browse_button = ttk.Button(file_frame, text="Browse...", command=self.browse_file)
        gui_browse_button.pack(side=tk.LEFT)

        # --- Predict Button ---
        global gui_predict_button
        gui_predict_button = ttk.Button(main_frame, text="Recognize Instruments", command=self.start_prediction_thread_gui, state=tk.DISABLED)
        gui_predict_button.pack(pady=10, ipadx=10, ipady=5)

        # --- Results Area ---
        results_container = ttk.LabelFrame(main_frame, text="Prediction Results", padding="10")
        results_container.pack(expand=True, fill=tk.BOTH)

        global gui_results_text
        gui_results_text = scrolledtext.ScrolledText(results_container, wrap=tk.WORD, height=20, state=tk.DISABLED, font=("Courier New", 10))
        gui_results_text.pack(expand=True, fill=tk.BOTH)
        
        # --- Status Bar ---
        global gui_status_var
        gui_status_var = tk.StringVar()
        gui_status_var.set("Initializing...")
        status_bar = ttk.Label(self.root, textvariable=gui_status_var, relief=tk.SUNKEN, anchor=tk.W, padding="2 5")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Start loading model
        gui_status_var.set("Loading model, vocab, and thresholds... Please wait.")
        threading.Thread(target=load_model_resources_threaded_for_gui, daemon=True).start()
        self.check_model_loaded_gui()

    def check_model_loaded_gui(self):
        global gui_predict_button, gui_browse_button
        if MODEL_LOADED_EVENT.is_set():
            if PREDICTION_MODEL is not None: # Successfully loaded
                if self.filepath_var.get(): # Enable predict if file is already selected
                    if gui_predict_button: gui_predict_button.config(state=tk.NORMAL)
                # Status already set by loader thread
            else: # Error during loading
                if gui_predict_button: gui_predict_button.config(state=tk.DISABLED)
            if gui_browse_button: gui_browse_button.config(state=tk.NORMAL) # Enable browse button anyway
        else: # Not loaded yet
            if gui_browse_button: gui_browse_button.config(state=tk.DISABLED) # Keep browse disabled while loading
            self.root.after(500, self.check_model_loaded_gui)

    def browse_file(self):
        global gui_predict_button
        filepath = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=(("Audio Files", "*.wav *.mp3 *.flac"), ("All files", "*.*"))
        )
        if filepath:
            self.filepath_var.set(filepath)
            if gui_results_text:
                gui_results_text.config(state=tk.NORMAL)
                gui_results_text.delete(1.0, tk.END)
                gui_results_text.insert(tk.END, f"Selected file: {os.path.basename(filepath)}\n")
                if MODEL_LOADED_EVENT.is_set() and PREDICTION_MODEL is not None:
                    gui_results_text.insert(tk.END,"Ready to predict.")
                    if gui_predict_button: gui_predict_button.config(state=tk.NORMAL)
                else:
                    gui_results_text.insert(tk.END,"Model still loading...")
                    if gui_predict_button: gui_predict_button.config(state=tk.DISABLED)
                gui_results_text.config(state=tk.DISABLED)

    def start_prediction_thread_gui(self):
        audio_path = self.filepath_var.get()
        if not audio_path:
            messagebox.showwarning("No File", "Please select an audio file first.")
            return
        if not MODEL_LOADED_EVENT.is_set() or PREDICTION_MODEL is None:
            messagebox.showinfo("Model Not Ready", "Model is loading or failed. Please wait.")
            return
        threading.Thread(target=predict_for_gui_threaded, args=(audio_path,), daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = InstrumentRecognizerApp(root)
    root.mainloop()