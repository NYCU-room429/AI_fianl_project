import numpy as np
import torch
from tqdm import tqdm  # For progress bars
import torch.nn as nn
import torch.optim as optim

# Use specific imports from utils and CRNN
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)  # Multi-label metrics

# Remove unused imports: StandardScaler, mean_squared_error, mean_absolute_error, r2_score, multiprocessing, Pool, cpu_count, scipy.stats, matplotlib.pyplot, librosa

import os  # Keep os for path handling and cpu_count

import utils  # Import your utils.py script
from CRNN import (
    MusicInstrumentDataset,
    CRNN,
    INSTRUMENT_MAPPING_PATH,
    MAX_MEL_FRAMES,
)  # Import classes and necessary constants from CRNN.py


# --- Configuration Parameters ---
# Paths
# <--- *** Change this path to your dataset root directory ***
DATASET_ROOT_PATH = "slakh2100_flac_redux/slakh2100_flac_redux/train"
# INSTRUMENT_MAPPING_PATH is imported from CRNN.py

MODEL_SAVE_PATH = "instrument_crnn_model.pth"

# Data parameters (Constants like MAX_MEL_FRAMES are imported from CRNN.py)

# Training parameters
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.001
TEST_SPLIT_RATIO = 0.2  # Ratio for validation/test split
RANDOM_STATE = 42  # For reproducible splits


# --- Main Execution ---

if __name__ == "__main__":
    # The initial check for instruments_class length can remain for info
    instruments_mapping_check = utils.read_instruments_class(INSTRUMENT_MAPPING_PATH)
    print("================")
    print(f"Number of instruments in mapping file: {len(instruments_mapping_check)}")
    print("================")

    # --- 1. Load Dataset Information ---
    # utils.load_dataset only scans the directories and gets file paths
    print(f"Loading dataset information from {DATASET_ROOT_PATH}...")
    dataset_df = utils.load_dataset(DATASET_ROOT_PATH)
    print(f"Found {len(dataset_df)} tracks.")

    if len(dataset_df) == 0:
        print(
            "No tracks found in the dataset directory. Please check DATASET_ROOT_PATH."
        )
        exit()

    # --- 2. Create Dataset Instance and Load Data ---
    # The MusicInstrumentDataset class handles reading audio/midi and preprocessing into memory
    print(
        f"\nCreating and loading data into MusicInstrumentDataset (MAX_MEL_FRAMES={MAX_MEL_FRAMES})..."
    )
    try:
        # Pass the dataframe and mapping path to the Dataset
        full_dataset = MusicInstrumentDataset(
            dataframe=dataset_df,
            instrument_mapping_path=INSTRUMENT_MAPPING_PATH,
            max_mel_frames=MAX_MEL_FRAMES,  # Pass the constant
        )

    except ValueError as e:
        print(f"Error creating dataset: {e}")
        exit()
    except FileNotFoundError as e:
        print(f"Error: Required file not found during dataset creation - {e}")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred during dataset creation: {e}")
        exit()

    if len(full_dataset) == 0:
        print(
            "No valid data samples prepared by the dataset. Check file availability, paths, and processing errors in utils and dataset loading."
        )
        exit()

    # --- 3. Determine Model Input/Output Shapes from Dataset ---
    # Get shapes from the first sample after data loading
    # Dataset item shape is [1, N_MELS, MAX_MEL_FRAMES] for input, [num_classes] for label
    sample_input, sample_label = full_dataset[0]
    input_channel = sample_input.shape[0]  # Should be 1
    n_mels = sample_input.shape[1]
    max_mel_frames_actual = sample_input.shape[2]  # This should match MAX_MEL_FRAMES
    pooled_time_steps = sample_label.shape[0]  # 125
    num_classes = sample_label.shape[
        1
    ]  # Number of output classes = number of instrument classes

    print(
        f"\nModel Input Shape (C, H, W): ({input_channel}, {n_mels}, {max_mel_frames_actual})"
    )
    print(f"Number of output classes: {num_classes}")
    print(f"Pooled time steps: {pooled_time_steps}")
    # Optionally print instrument classes defined by the dataset
    # print(f"Target instrument classes: {full_dataset.instrument_classes}")

    # --- 4. Split Dataset ---
    # Get indices for splitting
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=TEST_SPLIT_RATIO,
        random_state=RANDOM_STATE,
        # Note: Stratifying multi-label data is complex. Skipping for simplicity.
        # If needed, look into techniques like Iterative Stratification.
        # stratify=full_dataset.labels
    )

    # Create subset datasets using indices
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # --- 5. Create DataLoaders ---
    # Use os.cpu_count() for potentially faster loading
    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count() or 1, pin_memory=True)
    # val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count() or 1, pin_memory=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    print(
        f"Using {os.cpu_count() or 1} worker processes for DataLoaders. Pin_memory enabled."
    )

    # --- 6. Build the CRNN Model ---
    print("\nBuilding CRNN model...")
    # Pass the determined shapes to the model constructor
    model = CRNN(
        input_channel=input_channel,
        n_mels=n_mels,
        max_mel_frames=max_mel_frames_actual,
        num_classes=num_classes,
    )
    print("Model built.")

    # --- 7. Set up Device (CPU/GPU) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device for training: {device}")

    # --- 計算 POS_WEIGHT ---
    print("\nCalculating pos_weight for BCELoss...")
    num_positives_per_class = np.zeros(
        num_classes, dtype=np.float64
    )  # 使用 float64 避免溢出
    num_negatives_per_class = np.zeros(num_classes, dtype=np.float64)

    # 為了計算 pos_weight，我們需要迭代一次 train_loader
    # 確保 train_loader 是基於未打亂的 train_dataset (如果 shuffle=True，每次迭代順序不同，但統計結果應一致)
    # 或者，更穩妥的做法是直接從 train_dataset 的標籤計算（如果Dataset將所有標籤保存在內存中）
    # 這裡我們假設 MusicInstrumentDataset 的 self.labels 存儲了下採樣後的標籤
    # 並且 train_dataset 是 Subset，我們可以訪問其底層數據集 full_dataset

    # 直接從 full_dataset 的訓練子集標籤計算
    # 獲取訓練集的所有標籤:
    # full_dataset.labels 的形狀是 (total_samples, pooled_time_steps, num_classes)
    # train_indices 是訓練樣本在 full_dataset 中的索引
    if hasattr(full_dataset, "labels") and isinstance(full_dataset.labels, np.ndarray):
        train_labels_all_samples = full_dataset.labels[train_indices]
        # train_labels_all_samples shape: (num_train_samples, pooled_time_steps, num_classes)

        for c in range(num_classes):
            num_positives_per_class[c] = np.sum(train_labels_all_samples[:, :, c] == 1)
            num_negatives_per_class[c] = np.sum(train_labels_all_samples[:, :, c] == 0)

        total_label_entries = train_labels_all_samples.size
        print(f"Total label entries processed for pos_weight: {total_label_entries}")

    else:  # 如果 full_dataset.labels 不可用或格式不對，則退回到遍歷 DataLoader (較慢)
        print(
            "Warning: full_dataset.labels not directly accessible or not a NumPy array. Falling back to iterating train_loader for pos_weight calculation (slower)."
        )
        for _, batch_labels in tqdm(
            train_loader, desc="Counting labels for pos_weight"
        ):
            batch_labels_np = batch_labels.cpu().numpy()
            for c in range(num_classes):
                num_positives_per_class[c] += np.sum(batch_labels_np[:, :, c] == 1)
                num_negatives_per_class[c] += np.sum(batch_labels_np[:, :, c] == 0)

    print(f"Number of positive samples per class: {num_positives_per_class}")
    print(f"Number of negative samples per class: {num_negatives_per_class}")

    pos_weight_values = np.ones(num_classes, dtype=np.float32)  # 確保是 float32
    for c in range(num_classes):
        if num_positives_per_class[c] > 0:  # 確保分母不為0
            # 如果 num_negatives_per_class[c] 為0，結果會是0，這不合理。
            # 應該是 num_negatives / num_positives。如果 num_negatives 是0，說明這個類別總是正的，pos_weight 應該小。
            # 如果 num_positives 是0，pos_weight 應該大或未定義。
            if num_negatives_per_class[c] > 0:
                pos_weight_values[c] = (
                    num_negatives_per_class[c] / num_positives_per_class[c]
                )
            else:  # 類別 c 總是正的 (沒有負樣本)
                pos_weight_values[c] = 1.0 / (
                    num_positives_per_class[c] + 1e-6
                )  # 給一個很小的權重，避免為0
                print(
                    f"Warning: Class {c} has no negative samples. pos_weight set to a small value."
                )
        else:  # 類別 c 沒有正樣本
            pos_weight_values[c] = (
                np.sum(num_negatives_per_class)
                / (np.sum(num_positives_per_class) + 1e-6)
                if np.sum(num_positives_per_class) > 0
                else 1.0
            )  # 使用全局比例或設為1
            print(
                f"Warning: Class {c} has no positive samples. pos_weight set based on global ratio or to 1."
            )

    print(f"Calculated pos_weight values: {pos_weight_values}")
    pos_weight_tensor = torch.tensor(pos_weight_values, dtype=torch.float).to(device)
    # --- POS_WEIGHT 計算結束 ---

    # --- 8. Define Loss Function and Optimizer ---
    # Binary Cross-Entropy Loss for multi-label classification
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)  # <--- 改為這個
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 9. Training Loop ---
    print("\nStarting model training...")
    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train()  # Set model to training mode
        running_loss = 0.0
        train_loader_tqdm = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False
        )

        for inputs, labels in train_loader_tqdm:
            # Move data to the appropriate device
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            train_loader_tqdm.set_postfix(
                loss=loss.item()
            )  # Display current batch loss

        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{EPOCHS}, Training Loss: {epoch_loss:.4f}")

        # --- Validation Phase ---
        model.eval()  # 設定模型為驗證模式
        val_running_loss = 0.0
        all_labels = []  # 收集所有真實標籤
        all_predictions = []  # 收集所有預測機率

        val_loader_tqdm = tqdm(
            val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val  ]", leave=False
        )

        with torch.no_grad():  # 驗證階段不計算梯度
            for inputs, labels in val_loader_tqdm:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)  # 輸出為 sigmoid 機率

                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)

                all_labels.append(labels.cpu().numpy())
                all_predictions.append(outputs.cpu().numpy())  # 收集機率

        val_epoch_loss = val_running_loss / len(val_dataset)

        # 將 batch 合併
        all_labels = np.concatenate(
            all_labels, axis=0
        )  # (N, pooled_time_steps, num_classes)
        all_predictions = np.concatenate(all_predictions, axis=0)

        # 展平成 (N * pooled_time_steps, num_classes) 以符合 sklearn 格式
        all_labels_flat = all_labels.reshape(-1, all_labels.shape[-1])
        all_predictions_flat = all_predictions.reshape(-1, all_predictions.shape[-1])

        # 機率轉為二值
        THRESHOLD = 0.1  # 可以根據需要調整閾值
        binary_predictions_flat = (all_predictions_flat > THRESHOLD).astype(float)

        # 計算多標籤指標
        val_precision_micro = precision_score(
            all_labels_flat, binary_predictions_flat, average="micro", zero_division=0
        )
        val_recall_micro = recall_score(
            all_labels_flat, binary_predictions_flat, average="micro", zero_division=0
        )
        val_f1_micro = f1_score(
            all_labels_flat, binary_predictions_flat, average="micro", zero_division=0
        )
        # element-wise 二元準確率
        val_binary_accuracy = np.mean(all_labels_flat == binary_predictions_flat)

        print(
            f"Epoch {epoch+1}/{EPOCHS}, Validation Loss: {val_epoch_loss:.4f}, "
            f"Val Binary Acc: {val_binary_accuracy:.4f}, "
            f"Val Micro Prec: {val_precision_micro:.4f}, "
            f"Val Micro Rec: {val_recall_micro:.4f}, "
            f"Val Micro F1: {val_f1_micro:.4f}"
        )

    print("\nTraining finished.")

    # --- 11. Save the Model ---
    # Save the model's state dictionary
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model state dictionary saved to {MODEL_SAVE_PATH}")
