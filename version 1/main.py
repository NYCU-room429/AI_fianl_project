# main.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import random
import json # For logging config
import sys # For logger
from loguru import logger # Import globally here

import config
from utils import get_all_class_names, plot_training_history # Removed setup_logger from here
from data_loader import collect_all_file_tuples, InstrumentDataset
from C_utils import InstrumentCRNN
from train_utils import train_one_epoch, validate_and_find_thresholds

def setup_main_logger(save_dir, filename="training_log.log"): # Specific setup for main
    log_path = os.path.join(save_dir, filename)
    logger.remove() # Remove default handler to avoid duplicate console logs
    logger.add(sys.stderr, level="DEBUG", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    logger.add(log_path, rotation="10 MB", level="INFO", enqueue=True, format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}")
    return logger # Return the configured logger instance

def set_seed(seed_value):
    random.seed(seed_value); np.random.seed(seed_value); torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value); torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    set_seed(config.SEED)
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    # Logger is now setup and used directly from main
    setup_main_logger(config.MODEL_SAVE_DIR, config.LOG_FILENAME) # Call the setup

    logger.info(f"--- Initializing Run ---")
    logger.info(f"Using device: {config.DEVICE}")
    config_vars = {k: v for k, v in vars(config).items() if not k.startswith('__')}
    logger.info(f"Full Configuration:\n{json.dumps(config_vars, indent=2, default=str)}")

    instrument_names_vocab = get_all_class_names(config.CLASS_JSON_PATH)
    num_classes = len(instrument_names_vocab)

    logger.info("Collecting training file paths...")
    train_file_tuples = collect_all_file_tuples(
        os.path.join(config.SLAKH_ROOT_DIR, config.TRAIN_DIR_NAME),
        instrument_names_vocab, is_for_test_set=False
    )
    logger.info("Collecting validation file paths...")
    val_file_tuples = collect_all_file_tuples(
        os.path.join(config.SLAKH_ROOT_DIR, config.VALID_DIR_NAME),
        instrument_names_vocab, is_for_test_set=False
    )

    if not train_file_tuples: logger.critical("No training files found. Exiting."); return
    if not val_file_tuples: logger.critical("No validation files found. Exiting."); return

    train_dataset = InstrumentDataset(train_file_tuples, instrument_names_vocab, is_train=True)
    val_dataset = InstrumentDataset(val_file_tuples, instrument_names_vocab, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=config.DEVICE.type == 'cuda', drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=config.NUM_WORKERS, pin_memory=config.DEVICE.type == 'cuda')

    pos_weights = train_dataset.get_positive_weights().to(config.DEVICE)
    
    model = InstrumentCRNN(num_classes=num_classes).to(config.DEVICE)
    
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")
    logger.debug(f"Model Architecture:\n{model}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=config.LR_SCHEDULER_FACTOR,
                                  patience=config.LR_SCHEDULER_PATIENCE, verbose=False)

    best_macro_f1 = -1.0
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_micro_f1': [], 'val_macro_f1': []}

    logger.info(f"--- Starting Training for {config.EPOCHS} Epochs ---")
    for epoch in range(config.EPOCHS):
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch+1}/{config.EPOCHS} | LR: {current_lr:.2e}")
        
        # Removed logger from train_one_epoch and validate_and_find_thresholds calls
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE, config.GRAD_CLIP_NORM, epoch+1, config.EPOCHS)
        history['train_loss'].append(train_loss)
        logger.info(f"Epoch {epoch+1} Avg Train Loss: {train_loss:.4f}")
        
        val_loss, micro_f1, macro_f1, optimal_thresholds = validate_and_find_thresholds(
            model, val_loader, criterion, config.DEVICE, num_classes, epoch+1, config.EPOCHS
        )
        history['val_loss'].append(val_loss); history['val_micro_f1'].append(micro_f1); history['val_macro_f1'].append(macro_f1)
        
        logger.info(f"Epoch {epoch+1} Val Loss: {val_loss:.4f} | Val Micro-F1: {micro_f1:.4f} | Val Macro-F1: {macro_f1:.4f}")
        
        if macro_f1 > (best_macro_f1 + 1e-4) : 
             logger.debug(f"Optimal thresholds for new best epoch {epoch+1}: {optimal_thresholds.round(3).tolist()}")

        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(macro_f1)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < prev_lr:
            logger.info(f"Learning rate reduced from {prev_lr:.2e} to {new_lr:.2e}")


        if macro_f1 > best_macro_f1:
            old_best_f1 = best_macro_f1; best_macro_f1 = macro_f1; epochs_no_improve = 0
            try:
                torch.save(model.state_dict(), os.path.join(config.MODEL_SAVE_DIR, config.MODEL_FILENAME))
                np.save(os.path.join(config.MODEL_SAVE_DIR, config.THRESHOLDS_FILENAME), optimal_thresholds)
                logger.info(f"*** Best Model Saved! Val Macro F1 improved from {old_best_f1:.4f} to {best_macro_f1:.4f} ***")
            except Exception as e: logger.error(f"Error saving model/thresholds: {e}")
        else:
            epochs_no_improve += 1
            logger.info(f"Val Macro F1 ({macro_f1:.4f}) did not improve for {epochs_no_improve} epoch(s). Best: {best_macro_f1:.4f}")

        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            logger.info(f"Early stopping triggered after {config.EARLY_STOPPING_PATIENCE} epochs with no improvement.")
            break
            
    logger.info(f"--- Training Finished ---")
    logger.info(f"Best Validation Macro F1: {best_macro_f1:.4f}")
    plot_training_history(history, config.MODEL_SAVE_DIR, config.PLOT_FILENAME) # Removed logger

if __name__ == "__main__":
    main()