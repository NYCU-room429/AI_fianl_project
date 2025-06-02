import torch
import numpy as np
from tqdm import tqdm
from DataLoader import (
    collect_all_path,
    LoadDataset,
    CLASS_JSON_PATH,
    collate_fn_skip_nones,
)
from utils import get_all_class  # Now returns list of class strings
from CRNN import CRNN
from sklearn.metrics import f1_score, precision_score, recall_score
import os  # For checking file existence


def test(
    model_path,
    test_data_root,
    class_json,
    batch_size=1,
    device="cuda",
    num_tracks=None,
):
    test_midi_file_list, test_flac_file_list = collect_all_path(
        test_data_root, class_json_path=class_json, is_test=True
    )

    if num_tracks is not None and num_tracks > 0:
        print(f"Testing on the first {num_tracks} tracks.")
        test_midi_file_list = test_midi_file_list[:num_tracks]
        test_flac_file_list = test_flac_file_list[:num_tracks]

    if not test_midi_file_list:
        print("No test files found or collected. Exiting test.")
        return

    instrument_classes_list = get_all_class(
        class_json
    )  # e.g. ["Piano", "Guitar", "Drums"]
    num_classes = len(instrument_classes_list)

    model = CRNN(num_classes=num_classes).to(device)
    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist. Exiting.")
        return
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_dataset = LoadDataset(
        midi_file_list=test_midi_file_list,
        flac_file_list=test_flac_file_list,
        class_json_path=class_json,
        is_train=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_skip_nones,
    )

    thresholds_path = "best_thresholds_class_based.npy"
    final_thresholds = np.full(num_classes, 0.5)  # Default threshold

    if os.path.exists(thresholds_path):
        try:
            loaded_thresholds = np.load(thresholds_path)
            if len(loaded_thresholds) == num_classes:
                final_thresholds = loaded_thresholds
                print(f"Loaded thresholds from {thresholds_path}")
            else:
                print(
                    f"Warning: Mismatch in number of classes. {thresholds_path} has {len(loaded_thresholds)} thresholds, model has {num_classes}. Using default 0.5."
                )
        except Exception as e:
            print(
                f"Warning: Could not load or parse {thresholds_path}: {e}. Using default 0.5."
            )
    else:
        print(
            f"Warning: {thresholds_path} not found. Using default threshold of 0.5 for all classes."
        )

    # print(f"Using thresholds: {final_thresholds.tolist()[:10]}... (first 10 shown if many)")

    track_precisions = []
    track_recalls = []
    track_f1s = []
    processed_tracks = 0

    with torch.no_grad():
        for batch_idx, (mel, label) in enumerate(tqdm(test_loader, desc="Testing")):
            if mel is None or label is None:
                # print(f"Skipping empty batch {batch_idx}")
                continue

            mel = mel.to(device)
            outputs = model(mel)  # (batch, time_frames, num_classes)

            threshold_tensor = (
                torch.tensor(
                    final_thresholds, dtype=outputs.dtype, device=outputs.device
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )

            preds_proba = torch.sigmoid(outputs)
            preds_active_frames = (
                (preds_proba > threshold_tensor).cpu().numpy()
            )  # (batch, time_frames, num_classes)
            label_np = label.numpy()  # (batch, time_frames, num_classes)

            for i in range(preds_active_frames.shape[0]):
                current_preds_active_frames = preds_active_frames[
                    i
                ]  # (time_frames, num_classes)
                current_label_active_frames = label_np[i]  # (time_frames, num_classes)

                pred_inst_present_indices = np.where(
                    current_preds_active_frames.any(axis=0)
                )[0]
                label_inst_present_indices = np.where(
                    current_label_active_frames.any(axis=0)
                )[0]

                pred_inst_present_names = [
                    instrument_classes_list[k] for k in pred_inst_present_indices
                ]
                label_inst_present_names = [
                    instrument_classes_list[k] for k in label_inst_present_indices
                ]

                pred_frames_per_class = current_preds_active_frames.sum(axis=0)

                if pred_frames_per_class.any():
                    max_pred_frames = np.max(pred_frames_per_class)
                    longest_duration_pred_indices = np.where(
                        pred_frames_per_class == max_pred_frames
                    )[0]
                    longest_duration_pred_names = [
                        instrument_classes_list[k]
                        for k in longest_duration_pred_indices
                    ]
                else:
                    longest_duration_pred_names = ["None"]
                    max_pred_frames = 0

                track_identifier = "Unknown Track"
                if processed_tracks < len(test_midi_file_list):
                    try:
                        track_folder_name = test_midi_file_list[processed_tracks].split(
                            os.sep
                        )[
                            -3
                        ]  # e.g. Track00001
                        stem_name = os.path.splitext(
                            os.path.basename(test_midi_file_list[processed_tracks])
                        )[0]
                        if stem_name == "all_src":
                            track_identifier = f"Track: {track_folder_name} (Mix)"
                        else:
                            track_identifier = (
                                f"Track: {track_folder_name} (Stem: {stem_name})"
                            )
                    except IndexError:
                        track_identifier = f"Track File: {os.path.basename(test_midi_file_list[processed_tracks])}"

                print(
                    f"\n=== {track_identifier} (Overall index: {processed_tracks + 1}) ==="
                )
                print(
                    f"  Ground Truth Instrument Classes (Present): {sorted(list(set(label_inst_present_names)))}"
                )
                print(
                    f"  Predicted Instrument Classes (Present):  {sorted(list(set(pred_inst_present_names)))}"
                )
                print(
                    f"  Predicted Longest Duration Instrument(s): {sorted(longest_duration_pred_names)} (active for {max_pred_frames} frames)"
                )

                # Frame-wise metrics for this track
                track_f1_micro = f1_score(
                    current_label_active_frames,
                    current_preds_active_frames,
                    average="micro",
                    zero_division=0,
                )
                track_precision_micro = precision_score(
                    current_label_active_frames,
                    current_preds_active_frames,
                    average="micro",
                    zero_division=0,
                )
                track_recall_micro = recall_score(
                    current_label_active_frames,
                    current_preds_active_frames,
                    average="micro",
                    zero_division=0,
                )

                print(
                    f"  Track Micro F1: {track_f1_micro:.4f} | Precision: {track_precision_micro:.4f} | Recall: {track_recall_micro:.4f}"
                )

                track_f1s.append(track_f1_micro)
                track_precisions.append(track_precision_micro)
                track_recalls.append(track_recall_micro)
                processed_tracks += 1

    if not track_f1s:
        print("No tracks were processed successfully.")
        return

    print(f"\n--- Overall Average Metrics ({processed_tracks} tracks) ---")
    print(f"Average Micro F1: {np.mean(track_f1s):.4f}")
    print(f"Average Micro Precision: {np.mean(track_precisions):.4f}")
    print(f"Average Micro Recall: {np.mean(track_recalls):.4f}")


if __name__ == "__main__":
    test(
        model_path="best_model_class_based.pth",
        test_data_root=os.path.join(
            "slakh2100_flac_redux", "slakh2100_flac_redux", "test"
        ),
        class_json=CLASS_JSON_PATH,
        batch_size=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_tracks=None,
    )
