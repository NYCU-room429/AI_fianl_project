import torch
from torch.utils.data import DataLoader
from CRNN_ai import CachedMelLabelDataset, CRNN
import torch.nn as nn
import torch.optim as optim
import utils_ai
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import random
from tqdm import tqdm
import os

# 參數
DATASET_ROOT = "/Users/eddy/Desktop/code/AI_fianl_project/slakh2100_flac_redux/slakh2100_flac_redux/train"
INSTRUMENT_MAPPING_PATH = "/Users/eddy/Desktop/code/AI_fianl_project/slakh-utils/midi_inst_values/general_midi_inst_0based.json"
MEL_SAVE_DIR = "mel_cache"
N_MELS = 128
MAX_FRAMES = 1000
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.001
TEST_SPLIT_RATIO = 0.2
RANDOM_STATE = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 樂器類別
instruments_mapping = utils_ai.read_instruments_class(INSTRUMENT_MAPPING_PATH)
instrument_classes = utils_ai.get_target_instrument_classes(INSTRUMENT_MAPPING_PATH)
n_classes = len(instrument_classes)

# 掃描資料集
tracks = utils_ai.scan_tracks(DATASET_ROOT)
train_tracks, val_tracks = train_test_split(tracks, test_size=TEST_SPLIT_RATIO, random_state=RANDOM_STATE)

# === pos_weight 計算與載入 ===
POS_WEIGHT_PATH = "pos_weight.npy"
if os.path.exists(POS_WEIGHT_PATH):
    pos_weight = torch.tensor(np.load(POS_WEIGHT_PATH), dtype=torch.float32)
    print(f"[INFO] Loaded pos_weight from {POS_WEIGHT_PATH}")
    print("pos_weight:", pos_weight)
    pos_weight = np.clip(pos_weight, 0.5, 10)
    np.save(POS_WEIGHT_PATH, pos_weight)
    print(f"[INFO] Clipped and saved pos_weight to {POS_WEIGHT_PATH}")
    print("clip 後 pos_weight:", pos_weight)
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
else:
    def compute_pos_weight(dataset):
        total = np.zeros(n_classes)
        count = 0
        loader = DataLoader(dataset, batch_size=32)
        for _, label in tqdm(loader, desc="Computing pos_weight"):
            total += label.sum(dim=(0,1)).numpy()
            count += label.shape[1]
        pos_weight = (count - total) / (total + 1e-6)
        return pos_weight
    # 只用訓練集計算
    temp_train_set = CachedMelLabelDataset(
        train_tracks, MEL_SAVE_DIR, instrument_classes, instruments_mapping,
        n_mels=N_MELS, max_frames=MAX_FRAMES, train=True
    )
    pos_weight = compute_pos_weight(temp_train_set)
    pos_weight = np.clip(pos_weight, 0.5, 10)  # 建議 clip 範圍
    np.save(POS_WEIGHT_PATH, pos_weight)
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
    print(f"[INFO] Computed and saved pos_weight to {POS_WEIGHT_PATH}")
    print("pos_weight:", pos_weight)

# === 標準化一致性：只計算一次並存檔 ===
MEAN_STD_PATH = "mel_global_mean_std_Focal_D.npy"
if os.path.exists(MEAN_STD_PATH):
    mean_std = np.load(MEAN_STD_PATH)
    global_mean, global_std = mean_std[0], mean_std[1]
    print(f"[INFO] Loaded global_mean={global_mean}, global_std={global_std} from {MEAN_STD_PATH}")
else:
    def compute_global_mean_std_from_cache(track_list, mel_dir, sample_size=20):
        sample = random.sample(track_list, min(sample_size, len(track_list)))
        all_mels = []
        for item in sample:
            mel_path = os.path.join(mel_dir, f"{item['track']}.npy")
            mel = np.load(mel_path)
            all_mels.append(mel)
        all_mels = np.concatenate([m.flatten() for m in all_mels])
        return all_mels.mean(), all_mels.std()
    global_mean, global_std = compute_global_mean_std_from_cache(train_tracks, MEL_SAVE_DIR)
    np.save(MEAN_STD_PATH, np.array([global_mean, global_std]))
    print(f"[INFO] Computed and saved global_mean={global_mean}, global_std={global_std} to {MEAN_STD_PATH}")

# === 資料增強（SpecAugment + noise）===
def augment_mel(mel):
    if random.random() < 0.5:
        freq_mask = random.randint(0, mel.shape[0]//8)
        freq_start = random.randint(0, mel.shape[0]-freq_mask)
        mel[freq_start:freq_start+freq_mask, :] = 0
    if random.random() < 0.5:
        time_mask = random.randint(0, mel.shape[1]//8)
        time_start = random.randint(0, mel.shape[1]-time_mask)
        mel[:, time_start:time_start+time_mask] = 0
    if random.random() < 0.3:
        mel = mel + np.random.randn(*mel.shape) * 0.01
    return mel

# === Dataset with 標準化與增強 ===
class AugCachedMelLabelDataset(CachedMelLabelDataset):
    def __getitem__(self, idx):
        mel_tensor, label_tensor = super().__getitem__(idx)
        mel = mel_tensor.squeeze(0).numpy()
        # 標準化
        mel = (mel - global_mean) / (global_std + 1e-6)
        # 資料增強（僅訓練集）
        if self.train:
            mel = augment_mel(mel)
        mel_tensor = torch.from_numpy(mel).unsqueeze(0).float()
        return mel_tensor, label_tensor

# 資料集
train_set = AugCachedMelLabelDataset(train_tracks, MEL_SAVE_DIR, instrument_classes, instruments_mapping, n_mels=N_MELS, max_frames=MAX_FRAMES, train=True)
val_set = AugCachedMelLabelDataset(val_tracks, MEL_SAVE_DIR, instrument_classes, instruments_mapping, n_mels=N_MELS, max_frames=MAX_FRAMES, train=False)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

# === 檢查 label 是否為 0/1 ===
for mel, label in train_loader:
    print("label min:", label.min().item(), "label max:", label.max().item())
    print("label shape:", label.shape)
    print("label unique values:", torch.unique(label))
    break

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, reduction='mean', pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight
    def forward(self, input, target):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            input, target, reduction='none', pos_weight=self.pos_weight
        )
        probas = torch.sigmoid(input)
        pt = torch.where(target == 1, probas, 1 - probas)
        focal_weight = (1 - pt) ** self.gamma
        loss = focal_weight * bce_loss
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()

# 初始化 BCEWithLogitsLoss + pos_weight
# ...existing code...

# 你可以直接註解掉 BCEWithLogitsLoss，改用 FocalLoss（不加 pos_weight）
# criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
criterion = FocalLoss(gamma=2, pos_weight=None)  # 不加 pos_weight

model = CRNN(N_MELS, n_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

global_best_f1 = 0
for epoch in range(EPOCHS):
    model.train()
    train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
    for mel, label in train_iter:
        mel, label = mel.to(device), label.to(device)
        out = model(mel)
        min_len = min(out.shape[1], label.shape[1])
        loss = criterion(out[:,:min_len,:], label[:,:min_len,:])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_iter.set_postfix(loss=loss.item())
    # 驗證
    model.eval()
    val_loss = 0
    all_labels = []
    all_preds = []
    val_iter = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
    with torch.no_grad():
        for mel, label in val_iter:
            mel, label = mel.to(device), label.to(device)
            out = model(mel)  # logits
            min_len = min(out.shape[1], label.shape[1])
            # F1/threshold 用 sigmoid
            all_preds.append(torch.sigmoid(out[:, :min_len, :]).cpu().numpy().reshape(-1, n_classes))
            all_labels.append(label[:, :min_len, :].cpu().numpy().reshape(-1, n_classes))
            # Loss 用 logits
            val_loss += criterion(out[:, :min_len, :], label[:, :min_len, :]).item()
        all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    print(f"Epoch {epoch+1}/{EPOCHS}, Val Loss: {val_loss/len(val_loader):.4f}")

    # per-class threshold grid search
    best_f1 = 0
    best_ths = np.full(n_classes, 0.5)
    for i in range(n_classes):
        for th in np.arange(0.1, 0.9, 0.05):
            pred_bin = (all_preds[:, i] > th).astype(int)
            f1 = f1_score(all_labels[:, i], pred_bin, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_ths[i] = th
    np.save("best_ths_Focal_D.npy", best_ths)

    final_pred = (all_preds > best_ths).astype(int)

    # === 新增各種指標 ===
    val_epoch_loss = val_loss / len(val_loader)
    val_binary_accuracy = (final_pred == all_labels).mean()
    val_precision_micro = precision_score(all_labels, final_pred, average="micro", zero_division=0)
    val_recall_micro = recall_score(all_labels, final_pred, average="micro", zero_division=0)
    val_f1_micro = f1_score(all_labels, final_pred, average="micro", zero_division=0)
    val_f1_macro = f1_score(all_labels, final_pred, average="macro", zero_division=0)
    per_class_f1 = f1_score(all_labels, final_pred, average=None, zero_division=0)

    print(
        f"Epoch {epoch+1}/{EPOCHS}, Validation Loss: {val_epoch_loss:.4f}, "
        f"Val Binary Acc: {val_binary_accuracy:.4f}, "
        f"Val Micro Prec: {val_precision_micro:.4f}, "
        f"Val Micro Rec: {val_recall_micro:.4f}, "
        f"Val Micro F1: {val_f1_micro:.4f}, "
        f"Val Macro F1: {val_f1_macro:.4f}"
    )
    print(f"Per-class F1: {per_class_f1}")
    print(f"==> Best per-class threshold: {best_ths}, Best F1: {val_f1_micro:.4f}")

    if val_f1_micro > global_best_f1:
        global_best_f1 = val_f1_micro
        torch.save(model.state_dict(), "best_model_Focal_D.pth")
        np.save("best_ths_Focal_best_D.npy", best_ths)
        print(f"[INFO] Saved best model at epoch {epoch+1} with F1={global_best_f1:.4f}")

torch.save(model.state_dict(), "crnn_model_Focal_D.pth")
