import torch
import numpy as np
from tqdm import tqdm
from DataLoader import collect_all_path, LoadDataset
from utils import get_all_class
from CRNN import CRNN
from sklearn.metrics import f1_score, precision_score, recall_score


def test(
    model_path,
    midi_file_list,
    flac_file_list,
    class_json,
    batch_size=1,
    device="cuda",
    num_tracks=3,
):
    midi_file_list = midi_file_list[:num_tracks]
    flac_file_list = flac_file_list[:num_tracks]

    instrument_class = get_all_class(class_json)
    num_classes = len(instrument_class)
    model = CRNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_dataset = LoadDataset(midi_file_list, flac_file_list, is_train=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    best_thresholds = np.load("best_thresholds.npy")
    threshold = np.array([max(0.66, t) for t in best_thresholds])
    # threshold = np.array([max(0.45, t) for t in  [0.49, 0.1, 0.5700000000000001, 0.59, 0.31, 0.1, 0.49, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.51, 0.1, 0.45, 0.1, 0.1, 0.46, 0.1, 0.37, 0.5, 0.1, 0.5700000000000001, 0.22, 0.1, 0.1, 0.38, 0.36000000000000004, 0.4, 0.46, 0.38, 0.51, 0.44, 0.1, 0.1, 0.1, 0.1, 0.1, 0.12, 0.17, 0.1, 0.1, 0.1, 0.1, 0.35000000000000003, 0.21000000000000002, 0.45, 0.1, 0.1, 0.1, 0.1, 0.27, 0.35000000000000003, 0.1, 0.24000000000000002, 0.21000000000000002, 0.11, 0.17, 0.29000000000000004, 0.35000000000000003, 0.1, 0.11, 0.1, 0.1, 0.25, 0.1, 0.1, 0.1, 0.1, 0.25, 0.16, 0.1, 0.1, 0.52, 0.34, 0.53, 0.36000000000000004, 0.22, 0.1, 0.1, 0.39, 0.2, 0.39, 0.22, 0.1, 0.23, 0.1, 0.1, 0.1, 0.5700000000000001, 0.1, 0.1, 0.1, 0.1, 0.1, 0.31, 0.1, 0.14, 0.1, 0.31, 0.55, 0.26, 0.44, 0.47000000000000003, 0.2, 0.1, 0.5, 0.34, 0.1, 0.1, 0.1, 0.1, 0.35000000000000003, 0.1, 0.1, 0.13, 0.1, 0.18000000000000002, 0.1, 0.1, 0.31, 0.1, 0.13, 0.44, 0.1, 0.1, 0.1]])

    track_precisions = []
    track_recalls = []
    track_f1s = []

    with torch.no_grad():
        for idx, (mel, label) in enumerate(test_loader):
            mel = mel.to(device)
            outputs = model(mel)
            threshold_tensor = torch.tensor(
                threshold, dtype=outputs.dtype, device=outputs.device
            )
            preds = (
                (torch.sigmoid(outputs) > threshold_tensor).cpu().numpy()
            )  # (batch, 1000, num_classes)
            label_np = label.numpy()  # (batch, 1000, num_classes)

            # 只看有出現過的樂器（frame-wise OR）
            pred_inst_idx = np.where(preds[0].any(axis=0))[0]
            label_inst_idx = np.where(label_np[0].any(axis=0))[0]
            pred_inst = [instrument_class[i] for i in pred_inst_idx]
            label_inst = [instrument_class[i] for i in label_inst_idx]

            print(f"\n=== Track {idx+1} ===")
            print("正確樂器名稱:", label_inst)
            print("預測樂器名稱:", pred_inst)

            # 計算這個 track 的 precision/recall/f1 (micro)
            track_f1 = f1_score(
                label_np.reshape(-1, len(instrument_class)),
                preds.reshape(-1, len(instrument_class)),
                average="micro",
                zero_division=0,
            )
            track_precision = precision_score(
                label_np.reshape(-1, len(instrument_class)),
                preds.reshape(-1, len(instrument_class)),
                average="micro",
                zero_division=0,
            )
            track_recall = recall_score(
                label_np.reshape(-1, len(instrument_class)),
                preds.reshape(-1, len(instrument_class)),
                average="micro",
                zero_division=0,
            )
            print(
                f"Track F1: {track_f1:.4f} | Precision: {track_precision:.4f} | Recall: {track_recall:.4f}"
            )

            track_f1s.append(track_f1)
            track_precisions.append(track_precision)
            track_recalls.append(track_recall)

    # 平均每個 track 的分數
    print(f"\n平均 F1: {np.mean(track_f1s):.4f}")
    print(f"平均 Precision: {np.mean(track_precisions):.4f}")
    print(f"平均 Recall: {np.mean(track_recalls):.4f}")


if __name__ == "__main__":
    test_midi_file_list, test_flac_file_list = collect_all_path(
        "C:\\Users\\linbr\\Desktop\\AI_fianl_project\\slakh2100_flac_redux\\slakh2100_flac_redux\\test",
        is_test=True,
    )

    test(
        model_path="C:\\Users\\linbr\\Desktop\\AI_fianl_project\\test_CRNN\\best_model.pth",
        midi_file_list=test_midi_file_list,
        flac_file_list=test_flac_file_list,
        class_json="C:\\Users\\linbr\\Desktop\\AI_fianl_project\\slakh-utils\\midi_inst_values\\general_midi_inst_0based.json",
        batch_size=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_tracks=132,
    )
