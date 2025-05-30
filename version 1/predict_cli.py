# predict_cli.py
import torch
import numpy as np
import os
import argparse
import librosa

import config
from utils import generate_mel_spectrogram, get_all_class_names, setup_logger
from C_utils import InstrumentCRNN

def predict_on_audio_file(audio_file_path, model_path, thresholds_path, class_json_path, device, logger):
    if not os.path.exists(audio_file_path):
        logger.error(f"Audio file not found: {audio_file_path}")
        return

    instrument_names_vocab = get_all_class_names(class_json_path)
    num_classes = len(instrument_names_vocab)

    logger.info("Loading model for prediction...")
    model = InstrumentCRNN(
        num_classes=num_classes, n_mels=config.N_MELS, 
        num_spec_frames=config.NUM_SPECTROGRAM_FRAMES, 
        cnn_filters=config.MODEL_CNN_FILTERS, rnn_units=config.MODEL_RNN_UNITS,
        attention_heads=config.MODEL_ATTENTION_HEADS, dropout_rate=0.0 # No dropout at inference
    ).to(device)
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}"); return
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        logger.error(f"Error loading model state_dict: {e}"); return
    model.eval()

    if not os.path.exists(thresholds_path):
        logger.warning(f"Thresholds file not found: {thresholds_path}. Using 0.5 for all classes.")
        thresholds = np.full(num_classes, 0.5)
    else:
        thresholds = np.load(thresholds_path)
        thresholds = np.array([max(0.68, t) for t in  thresholds])
        logger.info(f"Loaded thresholds from {thresholds_path}")

    logger.info(f"Processing audio file: {audio_file_path}")
    audio_duration = librosa.get_duration(filename=audio_file_path, sr=config.SAMPLE_RATE)
    selected_start_sec = max(0, (audio_duration - config.SEGMENT_SEC) / 2.0) # Center crop

    mel_spec = generate_mel_spectrogram(
        audio_file_path, sr=config.SAMPLE_RATE, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS, segment_sec=config.SEGMENT_SEC,
        num_target_frames=config.NUM_SPECTROGRAM_FRAMES,
        selected_start_sec=selected_start_sec, is_train=False
    )
    if mel_spec is None: logger.error("Could not generate spectrogram."); return

    mean, std = np.mean(mel_spec), np.std(mel_spec)
    mel_spec = (mel_spec - mean) / (std + 1e-5)
    mel_spec_tensor = torch.from_numpy(mel_spec).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(mel_spec_tensor)
        probabilities = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    
    clip_max_probs = np.max(probabilities, axis=0) 

    logger.info(f"--- Predictions for: {os.path.basename(audio_file_path)} (Segment from {selected_start_sec:.2f}s) ---")
    detected_instruments_info = []
    for i, instrument_name in enumerate(instrument_names_vocab):
        prob = clip_max_probs[i]
        is_detected = prob >= thresholds[i]
        if is_detected or prob >= config.GUI_DISPLAY_PROB_THRESHOLD:
             detected_instruments_info.append({
                 "name": instrument_name, "prob": prob, 
                 "is_detected": is_detected, "threshold": thresholds[i]
             })
    
    # Sort by probability
    detected_instruments_info.sort(key=lambda x: x["prob"], reverse=True)

    answer = []
    if detected_instruments_info:
        logger.info("Instrument Predictions (Max Prob in Segment):")
        for item in detected_instruments_info:
            status = "DETECTED" if item["is_detected"] else ""
            if item["is_detected"]:
                answer.append(item['name'])
            logger.info(f"  - {item['name']:<25}: {item['prob']:.3f} {status} (Th: {item['threshold']:.2f})")
    else:
        logger.info("No instruments detected above display/learned thresholds.")
    
    print(answer)
    return detected_instruments_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict instruments from an audio file.")
    parser.add_argument("audio_file", type=str, help="Path to the input audio file.")
    parser.add_argument("--model_path", type=str, 
                        default=os.path.join(config.MODEL_SAVE_DIR, config.MODEL_FILENAME),
                        help="Path to the trained model (.pth) file.")
    parser.add_argument("--thresholds_path", type=str,
                        default=os.path.join(config.MODEL_SAVE_DIR, config.THRESHOLDS_FILENAME),
                        help="Path to the optimal thresholds (.npy) file.")
    parser.add_argument("--class_json", type=str, default=config.CLASS_JSON_PATH,
                        help="Path to the class names JSON file.")
    
    args = parser.parse_args()
    
    # Setup a simple logger for CLI
    cli_logger = setup_logger(os.getcwd(), "predict_cli_log.log") # Log in current dir for CLI

    predict_on_audio_file(args.audio_file, args.model_path, args.thresholds_path, 
                          args.class_json, config.DEVICE, cli_logger)