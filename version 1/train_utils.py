# train_utils.py
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_recall_curve
# No logger import here, assuming main.py handles logging summaries

def train_one_epoch(model, data_loader, criterion, optimizer, device, grad_clip_norm, epoch_num, total_epochs): # removed logger
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch_num}/{total_epochs} Training", ncols=110, leave=False)
    for i, (mel_specs, labels) in enumerate(progress_bar):
        mel_specs = mel_specs.to(device); labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(mel_specs)
        loss = criterion(logits.view(-1, logits.shape[-1]), labels.view(-1, labels.shape[-1]))
        loss.backward()
        if grad_clip_norm > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % (len(data_loader) // 5 + 1) == 0 or i == len(data_loader) -1 :
             progress_bar.set_postfix(avg_loss_batch=f"{loss.item():.4f}")
    return total_loss / len(data_loader)

def validate_and_find_thresholds(model, data_loader, criterion, device, num_classes, epoch_num, total_epochs): # removed logger
    model.eval()
    total_loss = 0.0
    all_logits_flat_list, all_labels_flat_list = [], []
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch_num}/{total_epochs} Validating", ncols=110, leave=False)
    with torch.no_grad():
        for mel_specs, labels in progress_bar:
            mel_specs = mel_specs.to(device); labels = labels.to(device)
            logits = model(mel_specs)
            loss = criterion(logits.view(-1, logits.shape[-1]), labels.view(-1, labels.shape[-1]))
            total_loss += loss.item()
            all_logits_flat_list.append(logits.view(-1, num_classes).cpu().numpy())
            all_labels_flat_list.append(labels.view(-1, num_classes).cpu().numpy())
            
    avg_val_loss = total_loss / len(data_loader)
    all_logits_np = np.concatenate(all_logits_flat_list, axis=0)
    all_labels_np = np.concatenate(all_labels_flat_list, axis=0).astype(int)
    all_probs_np = 1 / (1 + np.exp(-all_logits_np))

    optimal_thresholds = np.full(num_classes, 0.5)
    # print("DEBUG: Optimizing thresholds per class using PR curve...") # Use print if logger not available here

    class_iter = tqdm(range(num_classes), desc="Optimizing Thresh", ncols=100, leave=False, position=1) # position=1 for nested tqdm
    for class_idx in class_iter:
        class_probs = all_probs_np[:, class_idx]; class_true = all_labels_np[:, class_idx]
        if np.sum(class_true) == 0: optimal_thresholds[class_idx] = 0.95; continue
        if np.sum(class_true) == len(class_true): optimal_thresholds[class_idx] = 0.05; continue
        
        precision, recall, thresholds_pr = precision_recall_curve(class_true, class_probs)
        if len(thresholds_pr) == 0: optimal_thresholds[class_idx] = 0.5; continue

        f1_scores_pr = np.zeros_like(precision)
        valid_pr_idx = (precision + recall) > 1e-8
        f1_scores_pr[valid_pr_idx] = 2 * (precision[valid_pr_idx] * recall[valid_pr_idx]) / (precision[valid_pr_idx] + recall[valid_pr_idx])
        
        relevant_f1_scores = f1_scores_pr[:-1] if len(f1_scores_pr) > len(thresholds_pr) else f1_scores_pr
        if len(relevant_f1_scores) == 0: optimal_thresholds[class_idx] = 0.5; continue
            
        best_idx = np.nanargmax(relevant_f1_scores)
        optimal_thresholds[class_idx] = thresholds_pr[best_idx]
        optimal_thresholds[class_idx] = np.clip(optimal_thresholds[class_idx], 0.01, 0.99)

    final_preds_optimal = (all_probs_np >= optimal_thresholds[np.newaxis, :]).astype(int)
    micro_f1 = f1_score(all_labels_np, final_preds_optimal, average="micro", zero_division=0)
    macro_f1 = f1_score(all_labels_np, final_preds_optimal, average="macro", zero_division=0)
    
    return avg_val_loss, micro_f1, macro_f1, optimal_thresholds