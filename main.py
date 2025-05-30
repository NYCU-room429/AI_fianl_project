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
from DataLoader import collect_all_path, LoadDataset, CLASS_JSON_PATH, collate_fn_skip_nones
from CRNN import CRNN, train, validate
from utils import plot, get_all_class # get_all_class now returns list of class strings
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib # Ensure this is installed: pip install tqdm-joblib


# IMPORTANT: Ensure general_midi_inst_0based.json is a single valid JSON object.
# CLASS_JSON_PATH is defined in DataLoader.py and imported here.

def collect_labels_parallel(data_loader_for_labels):
    # Note: This function collects labels from a loader.
    # Ensure the loader's dataset is initialized correctly (e.g., not in 'is_train=True' mode if it affects label generation)
    # The primary purpose here is to get a representative set of labels for pos_weight calculation.
    def get_label(batch_data):
        _, label_tensor = batch_data
        if label_tensor is None:
            return np.array([]) # Return empty if batch was problematic
        return label_tensor.numpy()

    # Batches from the loader
    # Need to handle if collate_fn_skip_nones returns (None, None)
    batches = []
    for b in data_loader_for_labels:
        if b[0] is not None and b[1] is not None: # Check if batch is valid
            batches.append(b)
    
    if not batches:
        logger.error("No valid batches found to collect labels for pos_weight calculation.")
        return np.array([])


    with tqdm_joblib(tqdm(desc="Collecting labels for pos_weight", total=len(batches))):
        results = Parallel(n_jobs=-1, backend="threading")( # threading is fine for numpy operations
            delayed(get_label)(batch) for batch in batches
        )
    
    valid_results = [r for r in results if r.size > 0]
    if not valid_results:
        logger.error("All label collection results were empty.")
        return np.array([])
        
    return np.concatenate(valid_results, axis=0)


def find_best_threshold(class_logits, class_true_labels):
    current_best_f1 = -1
    current_best_thresh = 0.01 # Start low
    # Iterate from 0.01 up to 0.99
    for thresh_candidate in np.arange(0.01, 1.0, 0.01):
        # Apply sigmoid before thresholding if logits are passed
        # Assuming class_logits are raw model outputs (before sigmoid)
        class_preds = (torch.sigmoid(torch.tensor(class_logits)) > thresh_candidate).numpy()
        
        # Handle cases for F1 score calculation
        if np.sum(class_true_labels) == 0 and np.sum(class_preds) == 0: # Both no positives
            f1_score_val = 1.0
        elif np.sum(class_true_labels) == 0 and np.sum(class_preds) > 0: # False positives only
            f1_score_val = 0.0
        elif np.sum(class_true_labels) > 0 and np.sum(class_preds) == 0: # False negatives only
            f1_score_val = 0.0
        else: # Normal case
            f1_score_val = f1_score(
                class_true_labels,
                class_preds,
                average="binary", # Per-class F1
                zero_division=0,
            )
            
        if f1_score_val > current_best_f1:
            current_best_f1 = f1_score_val
            current_best_thresh = thresh_candidate
            
    # Removed: max(current_best_thresh, 0.4). Let the data decide the threshold.
    return current_best_thresh, current_best_f1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Using class definition JSON: {CLASS_JSON_PATH}")

    logger.info("Start collecting training file paths")
    train_midi_file_list, train_flac_file_list = collect_all_path(
        "slakh2100_flac_redux\\slakh2100_flac_redux\\train",
        class_json_path=CLASS_JSON_PATH,
        is_test=False # Important for training data to include stems
    )
    logger.info("Start collecting validation file paths")
    val_midi_file_list, val_flac_file_list = collect_all_path(
        "slakh2100_flac_redux\\slakh2100_flac_redux\\validation",
        class_json_path=CLASS_JSON_PATH,
        is_test=False # Validation can also use stems if desired, or set to True if only mix
    )

    if not train_midi_file_list:
        logger.error("No training files collected. Exiting.")
        return
    if not val_midi_file_list:
        logger.error("No validation files collected. Exiting.")
        return

    train_dataset = LoadDataset(
        midi_file_list=train_midi_file_list,
        flac_file_list=train_flac_file_list,
        class_json_path=CLASS_JSON_PATH,
        is_train=True,
    )
    val_dataset = LoadDataset(
        midi_file_list=val_midi_file_list,
        flac_file_list=val_flac_file_list,
        class_json_path=CLASS_JSON_PATH,
        is_train=False,
    )
    
    # Using collate_fn_skip_nones to handle potential None items from dataset
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8, collate_fn=collate_fn_skip_nones, pin_memory=True if device.type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=8, collate_fn=collate_fn_skip_nones, pin_memory=True if device.type == 'cuda' else False)

    # For pos_weight, create a temporary loader if train_loader uses is_train=True for augmentations
    # that might affect label distribution (though unlikely for these labels).
    # It's generally fine to use the main train_loader if SpecAugment doesn't alter label presence.
    logger.info("Start calculating pos weight from training data")
    all_labels_for_pos_weight = collect_labels_parallel(train_loader) # Use train_loader directly

    if all_labels_for_pos_weight.size == 0:
        logger.error("Could not collect labels for pos_weight calculation. Using default pos_weight=1.")
        pos_weight_tensor = torch.ones(len(get_all_class(CLASS_JSON_PATH)), dtype=torch.float32).to(device)
    else:
        num_frames_total = all_labels_for_pos_weight.shape[0] * all_labels_for_pos_weight.shape[1]
        if num_frames_total == 0: # Should be caught by .size == 0
             logger.error("Zero frames found for pos_weight. Defaulting pos_weight.")
             pos_weight_tensor = torch.ones(len(get_all_class(CLASS_JSON_PATH)), dtype=torch.float32).to(device)
        else:
            num_pos = all_labels_for_pos_weight.sum(axis=(0, 1)) # Sum over batch and time_frame dimensions
            num_neg = num_frames_total - num_pos 
            
            # Handle cases where a class never appears (num_pos[class_idx] == 0)
            # Avoid division by zero by adding epsilon.
            # If a class has no positive examples, its weight doesn't matter much or could be set to 1.
            pos_weight = num_neg / (num_pos + 1e-8) 
            pos_weight[num_pos == 0] = 1.0 # If no positive examples, set weight to 1
            
            pos_weight = np.clip(pos_weight, 1.0, 10.0) # Adjusted clip max slightly
            pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32).to(device)

    logger.info(f"Calculated pos_weight_tensor: {pos_weight_tensor.tolist()}")

    instrument_class_names = get_all_class(CLASS_JSON_PATH)
    num_classes = len(instrument_class_names)
    logger.info(f"Number of instrument classes: {num_classes}")
    logger.info(f"Instrument classes: {instrument_class_names}")


    logger.info("Start training CRNN")
    model = CRNN(num_classes=num_classes).to(device)
    
    #optimizer = optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-3)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2) # AdamW often preferred, lr slightly lower
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True) # For F1 score

    train_losses = []
    val_losses = []
    best_f1_macro = 0 # Changed to track macro F1
    best_model_path = "best_model_class_based.pth" # New model name
    best_thresholds_path = "best_thresholds_class_based.npy"

    EPOCHS = 100
    patience = 10 # Early stopping patience
    no_improve_epochs = 0

    for epoch in range(EPOCHS):
        logger.info(f"Epoch {epoch+1}/{EPOCHS}")
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_outputs_logits, val_labels_true = validate(
            model, val_loader, criterion, device
        )

        if val_outputs_logits.size == 0 or val_labels_true.size == 0:
            logger.warning(f"Validation output or labels are empty for epoch {epoch+1}. Skipping F1 calculation and scheduler step.")
            if val_loss is not None : val_losses.append(val_loss) # Still record val_loss if available
            if train_loss is not None: train_losses.append(train_loss)
            continue # Skip to next epoch if no validation data to score

        logger.info("Finding optimal thresholds per class on validation set...")
        
        # Parallel threshold search
        # Ensure val_outputs_logits and val_labels_true are (num_samples, num_classes)
        if val_outputs_logits.ndim == 3: # (batch_combined, time_frames, num_classes)
            # This should already be (num_total_frames, num_classes) from validate function
            pass 
        
        with tqdm_joblib(tqdm(desc="Threshold search", total=num_classes)) as progress_bar:
            results = Parallel(n_jobs=-1)( # Using default backend (loky or multiprocessing)
                delayed(find_best_threshold)(
                    val_outputs_logits[:, class_idx], val_labels_true[:, class_idx]
                )
                for class_idx in range(num_classes)
            )
        
        current_epoch_best_thresholds, current_epoch_f1_per_class = zip(*results)
        current_epoch_best_thresholds = np.array(current_epoch_best_thresholds)
        # current_epoch_f1_per_class = np.array(current_epoch_f1_per_class) # F1 per class with optimal threshold

        logger.info(f"Optimal thresholds (first 10): {current_epoch_best_thresholds.tolist()[:10]}")
        
        # Apply sigmoid and optimal thresholds to get predictions
        val_preds_optimal_thresh = (
            torch.sigmoid(torch.tensor(val_outputs_logits)) > torch.tensor(current_epoch_best_thresholds).unsqueeze(0)
        ).numpy() # unsqueeze for broadcasting if val_outputs_logits is (N, C)
        
        f1_micro_optimal_thresh = f1_score(
            val_labels_true, val_preds_optimal_thresh, average="micro", zero_division=0
        )
        f1_macro_optimal_thresh = f1_score(
            val_labels_true, val_preds_optimal_thresh, average="macro", zero_division=0
        )
        logger.info(
            f"Epoch {epoch+1} Val Loss: {val_loss:.4f} | Micro F1 (Optimal Thresh): {f1_micro_optimal_thresh:.4f} | Macro F1 (Optimal Thresh): {f1_macro_optimal_thresh:.4f}"
        )

        scheduler.step(f1_macro_optimal_thresh) # Step LR scheduler based on Macro F1

        if f1_macro_optimal_thresh > best_f1_macro:
            best_f1_macro = f1_macro_optimal_thresh
            no_improve_epochs = 0
            torch.save(model.state_dict(), best_model_path)
            np.save(best_thresholds_path, current_epoch_best_thresholds)
            logger.info(
                f"Best model saved at epoch {epoch+1} with F1 Macro (Optimal Thresh): {f1_macro_optimal_thresh:.4f}"
            )
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                logger.info(f"Early stopping at epoch {epoch+1} due to no improvement in Macro F1 for {patience} epochs.")
                break
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    
    logger.info(f"Training finished. Best Macro F1 on validation: {best_f1_macro:.4f}")
    if train_losses and val_losses:
        plot(train_losses, val_losses) # Plot losses

if __name__ == "__main__":
    main()