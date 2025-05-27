import torch
import numpy as np
from tqdm import tqdm
from DataLoader import collect_all_path, LoadDataset
from utils import gen_melgram, get_all_class, extract_label_from_midi
from CRNN import CRNN

def test(model_path, midi_file_list, flac_file_list, class_json, batch_size=1, device='cuda', num_tracks=3):
    # 只取前 num_tracks 個
    midi_file_list = midi_file_list[:num_tracks]
    flac_file_list = flac_file_list[:num_tracks]

    instrument_class = get_all_class(class_json)
    num_classes = len(instrument_class)
    model = CRNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_dataset = LoadDataset(midi_file_list, flac_file_list, is_train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    threshold = np.array([0.34, 0.11, 0.5700000000000001, 0.47000000000000003, 0.32, 0.1, 0.52, 0.1, 0.1, 0.1, 0.12, 0.1, 0.1, 0.1, 0.5800000000000001, 0.1, 0.39, 0.1, 0.13, 0.5700000000000001, 0.1, 0.27, 0.36000000000000004, 0.1, 0.6, 0.25, 0.5, 0.1, 0.48000000000000004, 0.45, 0.35000000000000003, 0.39, 0.5800000000000001, 0.53, 0.44, 0.46, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15000000000000002, 0.1, 0.1, 0.1, 0.1, 0.33, 0.24000000000000002, 0.4, 0.1, 0.1, 0.1, 0.1, 0.26, 0.29000000000000004, 0.1, 0.16, 0.14, 0.1, 0.16, 0.43, 0.45, 0.1, 0.1, 0.16, 0.1, 0.38, 0.1, 0.1, 0.1, 0.1, 0.29000000000000004, 0.1, 0.1, 0.1, 0.5, 0.38, 0.48000000000000004, 0.35000000000000003, 0.34, 0.1, 0.1, 0.28, 0.1, 0.34, 0.25, 0.1, 0.19, 0.1, 0.1, 0.1, 0.34, 0.1, 0.1, 0.1, 0.1, 0.1, 0.28, 0.24000000000000002, 0.11, 0.1, 0.4, 0.5800000000000001, 0.31, 0.5, 0.46, 0.13, 0.1, 0.63, 0.41000000000000003, 0.1, 0.1, 0.1, 0.1, 0.38, 0.1, 0.1, 0.1, 0.12, 0.28, 0.1, 0.1, 0.3, 0.1, 0.11, 0.4, 0.1, 0.1, 0.1])

    with torch.no_grad():
        for idx, (mel, label) in enumerate(test_loader):
            mel = mel.to(device)
            outputs = model(mel)
            threshold_tensor = torch.tensor(threshold, dtype=outputs.dtype, device=outputs.device)
            preds = (torch.sigmoid(outputs) > threshold_tensor).cpu().numpy() # (batch, 1000, num_classes)
            label_np = label.numpy()                              # (batch, 1000, num_classes)

            # 只看有出現過的樂器（frame-wise OR）
            pred_inst_idx = np.where(preds[0].any(axis=0))[0]
            label_inst_idx = np.where(label_np[0].any(axis=0))[0]
            pred_inst = [instrument_class[i] for i in pred_inst_idx]
            label_inst = [instrument_class[i] for i in label_inst_idx]

            print(f"\n=== Track {idx+1} ===")
            print("正確樂器名稱:", label_inst)
            print("預測樂器名稱:", pred_inst)

            # 若要看 frame-wise 0/1 結果也可保留
            # print("Predict (shape {}):".format(preds.shape))
            # print(preds.astype(int))
            # print("Label (shape {}):".format(label.shape))
            # print(label_np.astype(int))

if __name__ == "__main__":
    test_midi_file_list, test_flac_file_list = collect_all_path('D:\\AI_fianl_project\\slakh2100_flac_redux\\slakh2100_flac_redux\\test')

    test(
        model_path = 'D:\\AI_fianl_project\\best_model.pth',
        midi_file_list = test_midi_file_list,
        flac_file_list = test_flac_file_list,
        class_json = 'D:\\AI_fianl_project\\slakh-utils\\midi_inst_values\\general_midi_inst_0based.json',
        batch_size = 1,         # 建議設1，方便對齊
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_tracks=3
    )