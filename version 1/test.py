# import torch
# import numpy as np
# from tqdm import tqdm
# from DataLoader import collect_all_path, LoadDataset
# from utils import gen_melgram, get_all_class, extract_label_from_midi
# from CRNN import CRNN
# from sklearn.metrics import f1_score, precision_score, recall_score

# def test(model_path, midi_file_list, flac_file_list, class_json, batch_size=1, device='cuda', num_tracks=3):
#     midi_file_list = midi_file_list[:num_tracks]
#     flac_file_list = flac_file_list[:num_tracks]

#     instrument_class = get_all_class(class_json)
#     num_classes = len(instrument_class)
#     model = CRNN(num_classes=num_classes).to(device)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()

#     test_dataset = LoadDataset(midi_file_list, flac_file_list, is_train=False)
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#     best_thresholds = np.load(r"D:\AI_fianl_project\test_CRNN\instr_recognizer_thresholds.npy")
#     threshold = np.array([max(0.68, t) for t in  best_thresholds])
#     # threshold = np.array([max(0.45, t) for t in  [0.49, 0.1, 0.5700000000000001, 0.59, 0.31, 0.1, 0.49, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.51, 0.1, 0.45, 0.1, 0.1, 0.46, 0.1, 0.37, 0.5, 0.1, 0.5700000000000001, 0.22, 0.1, 0.1, 0.38, 0.36000000000000004, 0.4, 0.46, 0.38, 0.51, 0.44, 0.1, 0.1, 0.1, 0.1, 0.1, 0.12, 0.17, 0.1, 0.1, 0.1, 0.1, 0.35000000000000003, 0.21000000000000002, 0.45, 0.1, 0.1, 0.1, 0.1, 0.27, 0.35000000000000003, 0.1, 0.24000000000000002, 0.21000000000000002, 0.11, 0.17, 0.29000000000000004, 0.35000000000000003, 0.1, 0.11, 0.1, 0.1, 0.25, 0.1, 0.1, 0.1, 0.1, 0.25, 0.16, 0.1, 0.1, 0.52, 0.34, 0.53, 0.36000000000000004, 0.22, 0.1, 0.1, 0.39, 0.2, 0.39, 0.22, 0.1, 0.23, 0.1, 0.1, 0.1, 0.5700000000000001, 0.1, 0.1, 0.1, 0.1, 0.1, 0.31, 0.1, 0.14, 0.1, 0.31, 0.55, 0.26, 0.44, 0.47000000000000003, 0.2, 0.1, 0.5, 0.34, 0.1, 0.1, 0.1, 0.1, 0.35000000000000003, 0.1, 0.1, 0.13, 0.1, 0.18000000000000002, 0.1, 0.1, 0.31, 0.1, 0.13, 0.44, 0.1, 0.1, 0.1]])
#     # print(threshold)
#     track_precisions = []
#     track_recalls = []
#     track_f1s = []

#     with torch.no_grad():
#         for idx, (mel, label) in enumerate(test_loader):
#             mel = mel.to(device)
#             outputs = model(mel)
#             threshold_tensor = torch.tensor(threshold, dtype=outputs.dtype, device=outputs.device)
#             preds = (torch.sigmoid(outputs) > threshold_tensor).cpu().numpy() # (batch, 1000, num_classes)
#             label_np = label.numpy()                              # (batch, 1000, num_classes)

#             # 只看有出現過的樂器（frame-wise OR）
#             pred_inst_idx = np.where(preds[0].any(axis=0))[0]
#             label_inst_idx = np.where(label_np[0].any(axis=0))[0]
#             pred_inst = [instrument_class[i] for i in pred_inst_idx]
#             label_inst = [instrument_class[i] for i in label_inst_idx]

#             print(f"\n=== Track {idx+1} ===")
#             print("正確樂器名稱:", label_inst)
#             print("預測樂器名稱:", pred_inst)

#             # 計算這個 track 的 precision/recall/f1 (micro)
#             track_f1 = f1_score(label_np.reshape(-1, len(instrument_class)), preds.reshape(-1, len(instrument_class)), average='micro', zero_division=0)
#             track_precision = precision_score(label_np.reshape(-1, len(instrument_class)), preds.reshape(-1, len(instrument_class)), average='micro', zero_division=0)
#             track_recall = recall_score(label_np.reshape(-1, len(instrument_class)), preds.reshape(-1, len(instrument_class)), average='micro', zero_division=0)
#             print(f"Track F1: {track_f1:.4f} | Precision: {track_precision:.4f} | Recall: {track_recall:.4f}")

#             track_f1s.append(track_f1)
#             track_precisions.append(track_precision)
#             track_recalls.append(track_recall)

#     # 平均每個 track 的分數
#     print(f"\n平均 F1: {np.mean(track_f1s):.4f}")
#     print(f"平均 Precision: {np.mean(track_precisions):.4f}")
#     print(f"平均 Recall: {np.mean(track_recalls):.4f}")

# if __name__ == "__main__":
#     test_midi_file_list, test_flac_file_list = collect_all_path('D:\\AI_fianl_project\\slakh2100_flac_redux\\slakh2100_flac_redux\\test', is_test=True)

#     test(
#         model_path = 'D:\\AI_fianl_project\\test_CRNN\\instr_recognizer_best.pth',
#         midi_file_list = test_midi_file_list,
#         flac_file_list = test_flac_file_list,
#         class_json = 'D:\\AI_fianl_project\\slakh-utils\\midi_inst_values\\general_midi_inst_0based.json',
#         batch_size = 1,
#         device='cuda' if torch.cuda.is_available() else 'cpu',
#         num_tracks=132
#     )

# # import torch
# # import numpy as np
# # from tqdm import tqdm
# # from DataLoader import collect_all_path, LoadDataset
# # from utils import gen_melgram, get_all_class, extract_label_from_midi # 這裡 gen_melgram, extract_label_from_midi 不再直接用於測試，但仍需導入以供 DataLoader 使用
# # from CRNN import CRNN # 確保 CRNN 模型定義與 main.py 一致
# # from sklearn.metrics import f1_score, precision_score, recall_score
# # import os # 導入 os 模組以處理路徑

# # def test(model_path, midi_file_list, flac_file_list, class_json, batch_size=1, device='cuda', num_tracks=None):
# #     """
# #     測試模型在指定數據集上的性能。
# #     使用訓練階段學到的最佳閾值進行預測和評估。
# #     """
# #     instrument_class = get_all_class(class_json)
# #     num_classes = len(instrument_class)
    
# #     # 確保測試時的梅爾頻譜圖和標籤參數與訓練時一致
# #     NUM_FRAMES = 1000 
# #     SEGMENT_SEC = 100 

# #     # 初始化模型並載入預訓練權重
# #     # 確保 num_output_frames 參數傳遞給 CRNN
# #     model = CRNN(num_classes=num_classes, num_output_frames=NUM_FRAMES).to(device)
# #     model.load_state_dict(torch.load(model_path, map_location=device))
# #     model.eval() # 設置模型為評估模式

# #     # 如果指定了測試音軌數量，則截取列表
# #     if num_tracks is not None:
# #         midi_file_list = midi_file_list[:num_tracks]
# #         flac_file_list = flac_file_list[:num_tracks]

# #     # 創建測試資料集和資料載入器
# #     # 確保將 num_frame 和 segment_sec 傳遞給 LoadDataset
# #     test_dataset = LoadDataset(midi_file_list, flac_file_list, is_train=False, num_frame=NUM_FRAMES, segment_sec=SEGMENT_SEC)
# #     # 創建 DataLoader，加入 pin_memory=True
# #     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True) # num_workers=4 for test is usually fine

# #     # 載入訓練階段保存的最佳閾值
# #     best_thresholds_path = os.path.join(os.path.dirname(model_path), "best_thresholds.npy")
# #     if os.path.exists(best_thresholds_path):
# #         thresholds = np.load(best_thresholds_path)
# #         print(f"載入最佳閾值: {thresholds.tolist()}")
# #     else:
# #         print("警告: 未找到 best_thresholds.npy 檔案。將對所有類別使用預設閾值 0.5。")
# #         thresholds = np.full(num_classes, 0.5)
    
# #     # 將閾值轉換為 Tensor 並移到正確的設備
# #     threshold_tensor = torch.tensor(thresholds, dtype=torch.float32, device=device)

# #     track_precisions = []
# #     track_recalls = []
# #     track_f1s = []

# #     with torch.no_grad(): # 在測試時禁用梯度計算
# #         for idx, (mel, label) in enumerate(tqdm(test_loader, desc="Testing")):
# #             mel = mel.to(device)
# #             outputs = model(mel) # 模型前向傳播，輸出 logits (batch, 1000, num_classes)
            
# #             # 對 logits 應用 Sigmoid 函數，然後使用最佳閾值進行二值化預測
# #             preds = (torch.sigmoid(outputs) > threshold_tensor).cpu().numpy() # 預測結果 (batch, 1000, num_classes)
# #             label_np = label.numpy()                                      # 真實標籤 (batch, 1000, num_classes)

# #             # 提取並顯示每個音軌的實際樂器名稱和預測樂器名稱
# #             # 這裡通過檢查時間維度上是否有任何預測/真實值來判斷樂器是否存在於整個音軌
# #             pred_inst_mask = preds[0].any(axis=0) # (num_classes,) 布林遮罩
# #             label_inst_mask = label_np[0].any(axis=0) # (num_classes,) 布林遮罩

# #             pred_inst = [instrument_class[i] for i, present in enumerate(pred_inst_mask) if present]
# #             label_inst = [instrument_class[i] for i, present in enumerate(label_inst_mask) if present]

# #             print(f"\n=== 音軌 {idx+1} ({os.path.basename(flac_file_list[idx])}) ===")
# #             print("正確樂器名稱:", sorted(label_inst))
# #             print("預測樂器名稱:", sorted(pred_inst))

# #             # 計算當前音軌的 Micro F1、Precision 和 Recall
# #             # 將時間維度展平，以便對所有幀和類別進行統一評估
# #             track_f1 = f1_score(label_np.reshape(-1, num_classes), preds.reshape(-1, num_classes), average='micro', zero_division=0)
# #             track_precision = precision_score(label_np.reshape(-1, num_classes), preds.reshape(-1, num_classes), average='micro', zero_division=0)
# #             track_recall = recall_score(label_np.reshape(-1, num_classes), preds.reshape(-1, num_classes), average='micro', zero_division=0)
# #             print(f"音軌 F1: {track_f1:.4f} | 準確率: {track_precision:.4f} | 召回率: {track_recall:.4f}")

# #             track_f1s.append(track_f1)
# #             track_precisions.append(track_precision)
# #             track_recalls.append(track_recall)

# #     # 計算所有音軌的平均 F1、Precision 和 Recall
# #     print(f"\n======== 整體測試結果 ========")
# #     print(f"平均 F1: {np.mean(track_f1s):.4f}")
# #     print(f"平均 準確率: {np.mean(track_precisions):.4f}")
# #     print(f"平均 召回率: {np.mean(track_recalls):.4f}")

# # if __name__ == "__main__":
# #     # 請根據您的實際路徑修改以下變數
# #     SLAKH_TEST_PATH = 'D:\\AI_fianl_project\\slakh2100_flac_redux\\slakh2100_flac_redux\\test'
# #     MODEL_SAVE_DIR = 'D:\\AI_fianl_project\\' # 模型和閾值檔案的保存目錄

# #     # 收集測試集的檔案路徑 (這裡 collect_all_path 現在會使用 joblib)
# #     test_midi_file_list, test_flac_file_list = collect_all_path(SLAKH_TEST_PATH, is_test=True)

# #     # 運行測試函數
# #     test(
# #         model_path = os.path.join(MODEL_SAVE_DIR, 'best_model.pth'), # 確保模型檔案存在
# #         midi_file_list = test_midi_file_list,
# #         flac_file_list = test_flac_file_list,
# #         class_json = 'D:\\AI_fianl_project\\slakh-utils\\midi_inst_values\\general_midi_inst_0based.json',
# #         batch_size = 1, # 測試時通常使用較小的批次大小，甚至為 1，以便逐個音軌觀察結果
# #         device='cuda' if torch.cuda.is_available() else 'cpu',
# #         num_tracks=None # 設置為 None 則測試所有音軌；可以設置為一個數字來測試部分音軌
# #     )
# test.py
# test.py
import torch
import numpy as np
from tqdm import tqdm
import os
import argparse
from sklearn.metrics import f1_score, precision_score, recall_score # Keep these
from torch.utils.data import DataLoader

import config # Enhanced config.py
from utils import get_all_class_names, setup_logger # Enhanced utils.py
from data_loader import collect_all_file_tuples, InstrumentDataset # Enhanced data_loader.py
from C_utils import InstrumentCRNN # Enhanced model.py

def evaluate_model_per_track(model_path, thresholds_path, class_json_path, test_data_dir, device, logger, num_tracks_to_eval=None):
    logger.info(f"--- Starting Per-Track Evaluation on Test Set ---")
    logger.info(f"Loading test data from: {test_data_dir}")

    instrument_names_vocab = get_all_class_names(class_json_path)
    num_classes = len(instrument_names_vocab)

    # --- Data Loading for Test Set ---
    test_file_tuples_full = collect_all_file_tuples(
        test_data_dir,
        instrument_names_vocab,
        is_for_test_set=True
    )
    if not test_file_tuples_full:
        logger.error("No test files found. Aborting evaluation.")
        return

    if num_tracks_to_eval is not None and num_tracks_to_eval < len(test_file_tuples_full):
        logger.info(f"Evaluating on a subset of {num_tracks_to_eval} tracks.")
        test_file_tuples = test_file_tuples_full[:num_tracks_to_eval]
    else:
        test_file_tuples = test_file_tuples_full
        if num_tracks_to_eval is not None:
             logger.warning(f"num_tracks_to_eval ({num_tracks_to_eval}) >= total tracks ({len(test_file_tuples_full)}). Evaluating on all tracks.")


    test_dataset = InstrumentDataset(test_file_tuples, instrument_names_vocab, is_train=False)
    # IMPORTANT: batch_size must be 1 for per-track evaluation
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=config.NUM_WORKERS, pin_memory=True if device.type == 'cuda' else False)

    logger.info(f"Loaded {len(test_dataset)} test tracks in {len(test_loader)} batches (batch_size=1).")

    # --- Load Model ---
    logger.info(f"Loading model from: {model_path}")
    model = InstrumentCRNN(
        num_classes=num_classes, n_mels=config.N_MELS,
        num_spec_frames=config.NUM_SPECTROGRAM_FRAMES,
        cnn_filters=config.MODEL_CNN_FILTERS, rnn_units=config.MODEL_RNN_UNITS,
        attention_heads=config.MODEL_ATTENTION_HEADS, dropout_rate=0.0
    ).to(device)

    if not os.path.exists(model_path): logger.error(f"Model file not found: {model_path}"); return
    try: model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e: logger.error(f"Error loading model state_dict: {e}"); return
    model.eval()

    # --- Load Thresholds ---
    if not os.path.exists(thresholds_path):
        logger.warning(f"Thresholds file not found: {thresholds_path}. Using 0.5 for all classes.")
        thresholds = np.full(num_classes, 0.5)
    else:
        thresholds_loaded = np.load(thresholds_path)
        # Apply your custom threshold adjustment if this is a specific requirement for your test script
        # For general evaluation, it's better to use the thresholds as learned.
        # If you want to replicate your old script's behavior:
        # thresholds = np.array([max(0.68, t) for t in thresholds_loaded])
        # logger.info(f"Loaded thresholds from {thresholds_path} and applied min 0.68.")
        thresholds = thresholds_loaded # Use as is from training
        logger.info(f"Loaded thresholds from {thresholds_path}.")


    per_track_f1_scores = []
    per_track_precision_scores = []
    per_track_recall_scores = []

    logger.info("Running model on test set (track by track)...")
    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Evaluating Tracks", ncols=100, leave=True)
    
    with torch.no_grad():
        for track_idx, (mel_spec_batch, label_batch) in progress_bar:
            # batch_size is 1, so mel_spec_batch is (1, 1, F, T) and label_batch is (1, T_label, C)
            mel_spec_single = mel_spec_batch.to(device)
            # Labels for this single track: (Num_Frames, Num_Classes)
            true_labels_track_frames = label_batch.squeeze(0).cpu().numpy().astype(int)

            logits_track = model(mel_spec_single) # (1, Num_Frames, Num_Classes)
            probs_track_frames = torch.sigmoid(logits_track).squeeze(0).cpu().numpy() # (Num_Frames, Num_Classes)

            # Apply thresholds to get binary predictions for all frames in this track
            # probs_track_frames: (Num_Frames, Num_Classes)
            # thresholds: (Num_Classes) -> needs broadcasting to (1, Num_Classes)
            binary_preds_track_frames = (probs_track_frames >= thresholds[np.newaxis, :]).astype(int)

            # --- Display Per-Track Predicted vs. True Instruments (Clip-Level Summary) ---
            # True instruments in this track (if active in any frame)
            true_instruments_in_track_indices = np.where(true_labels_track_frames.any(axis=0))[0]
            true_instrument_names = sorted([instrument_names_vocab[i] for i in true_instruments_in_track_indices])

            # Predicted instruments in this track (if predicted in any frame above threshold)
            pred_instruments_in_track_indices = np.where(binary_preds_track_frames.any(axis=0))[0]
            pred_instrument_names = sorted([instrument_names_vocab[i] for i in pred_instruments_in_track_indices])
            
            original_audio_path = test_dataset.file_tuples[track_idx][0] # Get the audio path for logging
            logger.info(f"\n--- Track {track_idx+1}/{len(test_loader)}: {os.path.basename(original_audio_path)} ---")
            logger.info(f"  Ground Truth Instruments: {', '.join(true_instrument_names) if true_instrument_names else 'None'}")
            logger.info(f"  Predicted Instruments   : {', '.join(pred_instrument_names) if pred_instrument_names else 'None'}")

            # --- Calculate Frame-Level Micro F1, Precision, Recall FOR THIS TRACK ---
            # Reshape for sklearn: (Num_Frames_in_Track * Num_Classes_flattened)
            # Or more simply: (Num_Frames_in_Track, Num_Classes) which sklearn handles.
            
            if true_labels_track_frames.size > 0 : # Ensure there are frames to evaluate
                track_f1 = f1_score(true_labels_track_frames, binary_preds_track_frames, average="micro", zero_division=0)
                track_precision = precision_score(true_labels_track_frames, binary_preds_track_frames, average="micro", zero_division=0)
                track_recall = recall_score(true_labels_track_frames, binary_preds_track_frames, average="micro", zero_division=0)
                
                logger.info(f"  Track Micro F1: {track_f1:.4f} | Precision: {track_precision:.4f} | Recall: {track_recall:.4f}")

                per_track_f1_scores.append(track_f1)
                per_track_precision_scores.append(track_precision)
                per_track_recall_scores.append(track_recall)
            else:
                logger.warning(f"  Track {track_idx+1} had no frames in its label data. Skipping metrics for this track.")


    logger.info("\n--- Overall Average Per-Track Metrics ---")
    if per_track_f1_scores: # Check if list is not empty
        avg_f1 = np.mean(per_track_f1_scores)
        avg_precision = np.mean(per_track_precision_scores)
        avg_recall = np.mean(per_track_recall_scores)
        std_f1 = np.std(per_track_f1_scores)

        logger.info(f"Average Per-Track Micro F1-score  : {avg_f1:.4f} (Std: {std_f1:.4f})")
        logger.info(f"Average Per-Track Micro Precision: {avg_precision:.4f}")
        logger.info(f"Average Per-Track Micro Recall   : {avg_recall:.4f}")
    else:
        logger.warning("No per-track metrics were calculated (e.g., all tracks had issues).")
    
    logger.info(f"--- Evaluation Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model per track on the test set.")
    parser.add_argument("--model_path", type=str,
                        default=os.path.join(config.MODEL_SAVE_DIR, config.MODEL_FILENAME),
                        help="Path to the trained model (.pth) file.")
    parser.add_argument("--thresholds_path", type=str,
                        default=os.path.join(config.MODEL_SAVE_DIR, config.THRESHOLDS_FILENAME),
                        help="Path to the optimal thresholds (.npy) file.")
    parser.add_argument("--test_dir", type=str,
                        default=os.path.join(config.SLAKH_ROOT_DIR, config.TEST_DIR_NAME),
                        help="Path to the Slakh test set directory.")
    parser.add_argument("--num_tracks", type=int, default=None,
                        help="Number of tracks from the test set to evaluate (default: all).")
    # class_json_path is taken from config.py

    args = parser.parse_args()

    eval_logger = setup_logger(config.MODEL_SAVE_DIR, "evaluation_per_track_log.log")

    evaluate_model_per_track(args.model_path, args.thresholds_path, config.CLASS_JSON_PATH,
                             args.test_dir, config.DEVICE, eval_logger, args.num_tracks)