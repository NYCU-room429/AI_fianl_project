import pretty_midi
import numpy as np
from loguru import logger
import torch
import os, tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score
import torch.optim as optim
import torch.nn as nn
from DataLoader import collect_all_path, LoadDataset
from CRNN import CRNN, train, validate
from utils import plot

# 讀取 MIDI 檔案
# def demo_frame_lable():
#     midi = pretty_midi.PrettyMIDI('D:\\AI_fianl_project\\slakh2100_flac_redux\\slakh2100_flac_redux\\train\Track00001\\all_src.mid')

#     # 取得整體長度
#     total_time = midi.get_end_time()

#     # 定義 frame 大小（秒），例如 0.01 秒 = 10 毫秒
#     frame_size = total_time / 1000

#     # 計算總 frame 數
#     num_frames = 1000

#     # 建立每一格對應的樂器集合（用 set 來避免重複）
#     frame_instruments = [set() for _ in range(num_frames)]

#     # 處理每個 instrument
#     for instr in midi.instruments:
#         if instr.is_drum:
#             instrument_name = 'Drums'
#         else:
#             instrument_name = pretty_midi.program_to_instrument_name(instr.program)
        
#         # 處理每個音符
#         for note in instr.notes:
#             start_idx = int(note.start / frame_size)
#             end_idx = int(note.end / frame_size)
#             for i in range(start_idx, end_idx + 1):
#                 if 0 <= i < num_frames:
#                     frame_instruments[i].add(instrument_name)

#     # 印出部分結果
#     for i, instruments in enumerate(frame_instruments):  # 只印前 20 個 frames
#         print(f"Frame {i} ({i * frame_size:.2f}s): {sorted(instruments)}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """
    load data
    """
    logger.info("Start loading data")
    train_midi_file_list, train_flac_file_list = collect_all_path('D:\\AI_fianl_project\\slakh2100_flac_redux\\slakh2100_flac_redux\\train')
    val_midi_file_list, val_flac_file_list = collect_all_path('D:\\AI_fianl_project\\slakh2100_flac_redux\\slakh2100_flac_redux\\validation')
    train_dataset = LoadDataset(midi_file_list=train_midi_file_list, flac_file_list=train_flac_file_list, is_train=True)
    val_dataset = LoadDataset(midi_file_list=val_midi_file_list, flac_file_list=val_flac_file_list, is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=8)

    """
    Calculating pos weight
    """
    logger.info("Start calculating pos weight")
    all_labels = []
    for _, label in tqdm.tqdm(train_loader, desc="Collecting labels for pos_weight"):
        all_labels.append(label.numpy())
    all_labels = np.concatenate(all_labels, axis=0)  # (total_samples, time, num_classes)
    num_pos = all_labels.sum(axis=(0, 1))  # (num_classes,)
    num_neg = all_labels.shape[0] * all_labels.shape[1] - num_pos
    pos_weight = num_neg / (num_pos + 1e-8)
    pos_weight = np.clip(pos_weight, 1.0, 8.0)  # 可選：限制最大值
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32).to(device)   

    logger.info(f"Calculated pos_weight_tensor: {pos_weight_tensor.tolist()}")

    """
    CRNN training
    """
    logger.info("Start training CRNN")
    

    model = CRNN().to(device)

    base_params = [param for name, param in model.named_parameters() if param.requires_grad]
    optimizer = optim.Adam(base_params, lr=2e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    train_losses = []
    val_losses = []
    best_f1 = 0
    best_model_path = "best_model.pth"

    EPOCHS = 10
    for epoch in range(EPOCHS): #epoch
        train_loss = train(model, train_loader, criterion, optimizer, device)
        # val_loss, f1, f1_macro = validate(model, val_loader, criterion, device, threshold=0.625)
        val_loss, val_outputs, val_labels = validate(model, val_loader, criterion, device)
        # =========================================================
        logger.info("Finding optimal thresholds per class...")
        num_classes = model.fc.out_features

        best_thresholds = np.zeros(num_classes)
        best_f1_per_class = np.zeros(num_classes)

        for class_idx in tqdm.tqdm(range(num_classes), desc="Optimizing thresholds"):
            class_logits = val_outputs[:, class_idx]
            class_true_labels = val_labels[:, class_idx]
            
            current_best_f1 = -1
            current_best_thresh = 0.5 # 預設起始閾值

            # 嘗試一系列閾值，例如從 0.01 到 0.99，步長 0.01
            for thresh in np.arange(0.01, 1.0, 0.01):
                class_preds = (torch.sigmoid(torch.tensor(class_logits)) > thresh).numpy()
                
                # 對於每個類別，計算其 F1 Score
                # 注意：這裡應該使用 'binary' 模式的 F1 score for single class
                # 但由於 sklearn.metrics.f1_score 在處理單個類別時，如果所有真實值都是 0 或 1，
                # 且預測值也是，會導致問題 (zero_division)。
                # 更穩健的做法是使用 `average='binary'` 針對單個類別。
                # 或者更簡單地，檢查分母是否為零。
                
                # 確保有真實的正樣本或負樣本，否則 f1_score 可能會報錯或為 nan
                if np.sum(class_true_labels) == 0 and np.sum(class_preds) == 0:
                    f1_score_val = 1.0 # 完美預測，因為沒有任何正樣本
                elif np.sum(class_true_labels) == 0: # 沒有正樣本但預測了正樣本
                    f1_score_val = 0.0
                else:
                    f1_score_val = f1_score(class_true_labels, class_preds, average='binary', zero_division=0)
                
                if f1_score_val > current_best_f1:
                    current_best_f1 = f1_score_val
                    current_best_thresh = thresh
            
            best_thresholds[class_idx] = max(current_best_thresh, 0.4)
            best_f1_per_class[class_idx] = current_best_f1

        logger.info(f"Optimal thresholds per class: {best_thresholds.tolist()}")

        final_preds = (torch.sigmoid(torch.tensor(val_outputs)) > torch.tensor(best_thresholds)).numpy()

        f1_micro_optimal_thresh = f1_score(val_labels, final_preds, average='micro', zero_division=0)
        f1_macro_optimal_thresh = f1_score(val_labels, final_preds, average='macro', zero_division=0)

        logger.info(f"Validation Loss: {val_loss:.4f} | Micro F1 (Optimal Thresh): {f1_micro_optimal_thresh:.4f} | Macro F1 (Optimal Thresh): {f1_macro_optimal_thresh:.4f}")

        # 現在比較 f1_macro_optimal_thresh 來保存最佳模型
        if f1_macro_optimal_thresh > best_f1:
            best_f1 = f1_macro_optimal_thresh
            torch.save(model.state_dict(), best_model_path)
            np.save("best_thresholds.npy", best_thresholds)
            logger.info(f"Best model saved at epoch {epoch+1} with F1 Macro (Optimal Thresh): {f1_macro_optimal_thresh:.4f}")
        # scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # logger.info(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, F1: {f1:.4f}, F1 Macro: {f1_macro:.4f}")

        # if f1_macro > best_f1:
        #     best_f1 = f1_macro
        #     torch.save(model.state_dict(), best_model_path)
        #     logger.info(f"Best model saved at epoch {epoch+1} with F1 Macro: {f1_macro:.4f}")

    """
    CRNN plot loss
    """
    plot(train_losses, val_losses)

if __name__ == '__main__':
    main()