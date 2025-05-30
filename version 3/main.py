import pretty_midi
import numpy as np
from loguru import logger
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score
import torch.optim as optim
import torch.nn as nn
from DataLoader import collect_all_path, LoadDataset
from CRNN import CRNN, train, validate
from utils import plot
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib


def collect_labels_parallel(train_loader):
    def get_label(batch):
        _, label = batch
        return label.numpy()

    batches = list(train_loader)
    with tqdm_joblib(tqdm(desc="Collecting labels", total=len(batches))):
        results = Parallel(n_jobs=-1, backend="threading")(
            delayed(get_label)(batch) for batch in batches
        )
    return np.concatenate(results, axis=0)


def find_best_threshold(class_logits, class_true_labels):
    current_best_f1 = -1
    current_best_thresh = 0.5
    for thresh in np.arange(0.01, 1.0, 0.01):
        class_preds = (torch.sigmoid(torch.tensor(class_logits)) > thresh).numpy()
        if np.sum(class_true_labels) == 0 and np.sum(class_preds) == 0:
            f1_score_val = 1.0
        elif np.sum(class_true_labels) == 0:
            f1_score_val = 0.0
        else:
            f1_score_val = f1_score(
                class_true_labels,
                class_preds,
                average="binary",
                zero_division=0,
            )
        if f1_score_val > current_best_f1:
            current_best_f1 = f1_score_val
            current_best_thresh = thresh
    return max(current_best_thresh, 0.4), current_best_f1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Start loading data")
    train_midi_file_list, train_flac_file_list = collect_all_path(
        "C:\\Users\\linbr\\Desktop\\AI_fianl_project\\slakh2100_flac_redux\\slakh2100_flac_redux\\train"
    )
    val_midi_file_list, val_flac_file_list = collect_all_path(
        "C:\\Users\\linbr\\Desktop\\AI_fianl_project\\slakh2100_flac_redux\\slakh2100_flac_redux\\validation"
    )
    train_dataset = LoadDataset(
        midi_file_list=train_midi_file_list,
        flac_file_list=train_flac_file_list,
        is_train=True,
    )
    val_dataset = LoadDataset(
        midi_file_list=val_midi_file_list,
        flac_file_list=val_flac_file_list,
        is_train=False,
    )
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=8)

    logger.info("Start calculating pos weight")
    all_labels = collect_labels_parallel(train_loader)
    num_pos = all_labels.sum(axis=(0, 1))
    num_neg = all_labels.shape[0] * all_labels.shape[1] - num_pos
    pos_weight = num_neg / (num_pos + 1e-8)
    pos_weight = np.clip(pos_weight, 1.0, 8.0)
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32).to(device)
    logger.info(f"Calculated pos_weight_tensor: {pos_weight_tensor.tolist()}")

    logger.info("Start training CRNN")
    model = CRNN().to(device)
    base_params = [
        param for name, param in model.named_parameters() if param.requires_grad
    ]
    optimizer = optim.Adam(base_params, lr=2e-3, weight_decay=1e-3)  # L2正則化加強
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    train_losses = []
    val_losses = []
    best_f1 = 0
    best_model_path = "best_model.pth"
    EPOCHS = 100
    patience = 10
    no_improve = 0

    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_outputs, val_labels = validate(
            model, val_loader, criterion, device
        )
        logger.info("Finding optimal thresholds per class...")

        num_classes = model.fc.out_features

        # === Parallel threshold search with tqdm_joblib ===
        with tqdm_joblib(tqdm(desc="Threshold search", total=num_classes)):
            results = Parallel(n_jobs=-1)(
                delayed(find_best_threshold)(
                    val_outputs[:, class_idx], val_labels[:, class_idx]
                )
                for class_idx in range(num_classes)
            )
        best_thresholds, best_f1_per_class = zip(*results)
        best_thresholds = np.array(best_thresholds)
        best_f1_per_class = np.array(best_f1_per_class)

        logger.info(f"Optimal thresholds per class: {best_thresholds.tolist()}")
        final_preds = (
            torch.sigmoid(torch.tensor(val_outputs)) > torch.tensor(best_thresholds)
        ).numpy()
        f1_micro_optimal_thresh = f1_score(
            val_labels, final_preds, average="micro", zero_division=0
        )
        f1_macro_optimal_thresh = f1_score(
            val_labels, final_preds, average="macro", zero_division=0
        )
        logger.info(
            f"Validation Loss: {val_loss:.4f} | Micro F1 (Optimal Thresh): {f1_micro_optimal_thresh:.4f} | Macro F1 (Optimal Thresh): {f1_macro_optimal_thresh:.4f}"
        )
        if f1_macro_optimal_thresh > best_f1:
            best_f1 = f1_macro_optimal_thresh
            no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            np.save("best_thresholds.npy", best_thresholds)
            logger.info(
                f"Best model saved at epoch {epoch+1} with F1 Macro (Optimal Thresh): {f1_macro_optimal_thresh:.4f}"
            )
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    plot(train_losses, val_losses)


if __name__ == "__main__":
    main()
