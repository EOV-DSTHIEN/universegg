#!/usr/bin/env python3
# ================== UniverSeg Training on Ovatus Dataset ==================
# Đào tạo model few-shot segmentation với chiến lược:
# 1. Tạo support set động từ support pool
# 2. Query image = test image, Label = GT mask
# 3. Dùng weighted loss cho imbalanced classes
# 4. Validation trên dev set
# 5. Test trên test set

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import json
from datetime import datetime

from make_OVatusData import OvatusDataset, LABEL2ID, NUM_CLASSES
from split_dataset import patient_level_split_60_20_20
from universeg import universeg

# ================== CONFIGURATION ==================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Model and training
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
WARMUP_EPOCHS = 5
RESUME_FROM_CHECKPOINT = None  # Set to checkpoint path to resume: "checkpoints/latest_checkpoint.pt"

# Data configuration
SUPPORT_CONFIG = {
    # Format: (support_total, support_target, weight)
    # weight: loss weight for this class (higher = more important)
    "nang_da_thuy":      (64, 24, 1.0),   
    "nang_don_thuy":     (64, 32, 1.0),   
    "nang_da_thuy_dac":  (48, 40, 2.0),   # Rare: higher weight
    "nang_don_thuy_dac": (32, 28, 3.0),   # Very rare: highest weight
    "u_bi":              (64, 60, 1.0),   
    "u_dac":             (64, 60, 1.5),   
}

LABEL_NAMES = list(LABEL2ID.keys())

# Thresholds for evaluation
THRESHOLDS = {
    "nang_da_thuy": 0.45,
    "nang_don_thuy": 0.45,
    "nang_da_thuy_dac": 0.40,
    "nang_don_thuy_dac": 0.40,
    "u_bi": 0.50,
    "u_dac": 0.35,
}

# Output directories
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"
VIS_DIR = "visualizations_training"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

VISUALIZE_N = 10


# ================== LOSS FUNCTIONS ==================
class WeightedDiceLoss(nn.Module):
    """Binary Dice loss with per-class weights for handling imbalance"""
    
    def __init__(self, class_weights=None, smooth=1e-6):
        super().__init__()
        self.class_weights = class_weights or {}
        self.smooth = smooth
    
    def forward(self, pred, target, class_idx=None):
        """
        Args:
            pred: [B, H, W] logits (before sigmoid)
            target: [B, H, W] binary labels
            class_idx: which class (for weight lookup)
        """
        sigmoid_pred = torch.sigmoid(pred)
        
        # Dice coefficient
        intersection = (sigmoid_pred * target).sum()
        union = sigmoid_pred.sum() + target.sum()
        
        if union == 0:
            dice = 1.0 if target.sum() == 0 else 0.0
        else:
            dice = (2 * intersection + self.smooth) / (union + self.smooth)
        
        # Apply class weight
        if class_idx is not None and class_idx in self.class_weights:
            weight = self.class_weights[class_idx]
        else:
            weight = 1.0
        
        loss = (1 - dice) * weight
        return loss, dice.detach()


class FocalLoss(nn.Module):
    """Focal loss for hard examples"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, H, W] logits
            target: [B, H, W] binary labels
        """
        sigmoid_pred = torch.sigmoid(pred)
        ce = nn.BCEWithLogitsLoss(reduction='none')(pred, target)
        
        pt = sigmoid_pred * target + (1 - sigmoid_pred) * (1 - target)
        focal_weight = (1 - pt) ** self.gamma
        
        focal_loss = self.alpha * focal_weight * ce
        return focal_loss.mean()


# ================== CHECKPOINT FUNCTIONS ==================
def save_checkpoint(model, optimizer, scheduler, epoch, history, checkpoint_path):
    """Save training checkpoint with full state for resuming"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'history': history,
    }, checkpoint_path)
    print(f"✓ Checkpoint saved: {checkpoint_path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    """Load training checkpoint and resume from saved epoch"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    history = checkpoint['history']
    print(f"✓ Checkpoint loaded: {checkpoint_path}")
    print(f"  Resuming from epoch {start_epoch}/{EPOCHS}")
    return start_epoch, history


# ================== UTILITY FUNCTIONS ==================
def build_label_indices(dataset):
    """Build mapping: label_idx -> list of sample indices"""
    label_indices = defaultdict(list)
    
    print("\n[INFO] Building label indices from support pool...")
    for idx in range(len(dataset)):
        _, masks, _ = dataset[idx]
        for label_idx in range(NUM_CLASSES):
            if masks[label_idx].sum() > 10:
                label_indices[label_idx].append(idx)
    
    print("[INFO] Label distribution in support pool:")
    for label_idx in range(NUM_CLASSES):
        indices = label_indices[label_idx]
        label_name = LABEL_NAMES[label_idx]
        pct = 100 * len(indices) / len(dataset)
        print(f"  [{label_idx}] {label_name:20s}: {len(indices):3d} ({pct:5.1f}%)")
    
    return label_indices


def sample_support_adaptive(pool, label_indices, target_label_idx, 
                            n_total, n_target, seed=None):
    """Adaptive stratified sampling with guarantee of target examples"""
    rng = np.random.default_rng(seed) if seed else np.random
    
    target_pool = label_indices[target_label_idx]
    
    # Adjust if not enough samples
    n_target = min(n_target, len(target_pool))
    
    if len(pool) < n_total:
        n_total = len(pool)
        n_target = min(n_target, n_total)
    
    # Sample target examples
    if n_target > 0:
        target_idxs = rng.choice(target_pool, size=n_target, replace=False).tolist()
    else:
        target_idxs = []
    
    # Sample random examples
    n_random = n_total - n_target
    available = [i for i in range(len(pool)) if i not in target_idxs]
    
    if len(available) < n_random:
        random_idxs = available
    else:
        random_idxs = rng.choice(available, size=n_random, replace=False).tolist()
    
    # Combine and shuffle
    final_idxs = target_idxs + random_idxs
    rng.shuffle(final_idxs)
    
    # Load data
    support_images = []
    support_labels_list = []
    
    for i in final_idxs:
        img, labels, _ = pool[i]
        support_images.append(img)
        support_labels_list.append(labels)
    
    support_images = torch.stack(support_images).to(DEVICE)
    support_labels = torch.stack(support_labels_list).to(DEVICE)
    
    return support_images, support_labels


def dice_score(y_pred, y_true, threshold=0.5):
    """Calculate Dice coefficient"""
    y_pred_binary = (y_pred > threshold).float()
    intersection = (y_pred_binary * y_true).sum()
    union = y_pred_binary.sum() + y_true.sum()
    
    if union == 0:
        return 1.0 if y_true.sum() == 0 else 0.0
    
    return (2 * intersection / union).item()


def get_threshold_for_label(label_idx):
    """Get threshold for a specific label"""
    label_name = LABEL_NAMES[label_idx]
    return THRESHOLDS.get(label_name, 0.5)


def visualize_batch(images, gt_masks, pred_masks, epoch, batch_idx, valid_labels):
    """Visualize training results"""
    if len(valid_labels) == 0:
        return
    
    batch_size = images.shape[0]
    n_labels = len(valid_labels)
    
    fig, axes = plt.subplots(batch_size, n_labels, figsize=(4*n_labels, 3*batch_size))
    
    if batch_size == 1:
        axes = axes[np.newaxis, :]
    
    for b in range(min(batch_size, 3)):  # Show max 3 samples per visualization
        for col, label_idx in enumerate(valid_labels):
            label_name = LABEL_NAMES[label_idx]
            
            # Image
            rgb = images[b].permute(1, 2, 0).cpu().numpy()
            
            # GT and Pred
            gt = gt_masks[b, label_idx].cpu().numpy()
            pred = pred_masks[b, label_idx].cpu().numpy()
            threshold = get_threshold_for_label(label_idx)
            pred_binary = (pred > threshold).astype(np.float32)
            
            # Overlay
            canvas = np.zeros((*rgb.shape[:2], 3))
            canvas[:, :] = rgb  # Base image
            canvas[gt > 0.5, 1] = 0.5  # Green for GT
            canvas[pred_binary > 0.5, 0] = 0.5  # Red for Pred
            
            ax = axes[b, col]
            ax.imshow(canvas)
            
            dice = dice_score(torch.tensor(pred), torch.tensor(gt), threshold=threshold)
            ax.set_title(f"{label_name}\nDice: {dice:.3f}", fontsize=8)
            ax.axis('off')
    
    plt.tight_layout()
    filepath = os.path.join(VIS_DIR, f"epoch_{epoch:03d}_batch_{batch_idx:03d}.png")
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()


# ================== TRAINING FUNCTION ==================
def train_epoch(model, support_pool, label_indices, dev_set, epoch, optimizer, 
                dice_loss_fn, focal_loss_fn, class_weights):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0.0
    total_dice = defaultdict(list)
    num_batches = 0
    vis_count = 0
    
    # Sample a batch of query images from support pool
    # In few-shot setting: query = test image, support = training examples
    batch_size = BATCH_SIZE
    n_queries = min(batch_size, len(support_pool))
    query_indices = np.random.choice(len(support_pool), n_queries, replace=False)
    
    pbar = tqdm(query_indices, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for query_idx in pbar:
        query_img, query_gt, _ = support_pool[query_idx]
        query_img = query_img.to(DEVICE)
        query_gt = query_gt.to(DEVICE)
        query_img_gray = query_img.mean(dim=0, keepdim=True)
        
        # Find valid labels
        valid_labels = [i for i in range(NUM_CLASSES) if query_gt[i].sum() > 10]
        
        if len(valid_labels) == 0:
            continue
        
        batch_loss = 0.0
        batch_dices = []
        pred_masks = []
        
        # Predict each label independently
        for label_idx in valid_labels:
            label_name = LABEL_NAMES[label_idx]
            n_total, n_target, _ = SUPPORT_CONFIG[label_name]
            
            # Sample support set
            seed = epoch * len(support_pool) + query_idx * NUM_CLASSES + label_idx
            support_images, support_labels_all = sample_support_adaptive(
                support_pool, label_indices, label_idx,
                n_total=n_total, n_target=n_target, seed=seed
            )
            
            # Extract target label
            support_labels_single = support_labels_all[:, label_idx:label_idx+1]
            support_images_gray = support_images.mean(dim=1, keepdim=True)
            
            # Forward pass
            with torch.set_grad_enabled(True):
                # Model expects: (B, H, W), (B, N, H, W), (B, N, H, W)
                logits = model(
                    query_img_gray.unsqueeze(0),  # [1, H, W]
                    support_images_gray.unsqueeze(0),  # [1, N, H, W]
                    support_labels_single.unsqueeze(0)  # [1, N, H, W]
                )[0, 0]  # [H, W]
                
                # Loss
                dice_loss, dice = dice_loss_fn(
                    logits, query_gt[label_idx], class_idx=label_idx
                )
                focal_loss = focal_loss_fn(logits, query_gt[label_idx])
                
                loss = dice_loss + 0.5 * focal_loss
                batch_loss += loss
                batch_dices.append(dice.item())
                
                pred_masks.append(logits.detach())
                total_dice[label_name].append(dice.item())
        
        # Backward pass
        if batch_loss > 0:
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += batch_loss.item()
            num_batches += 1
            
            mean_dice = np.mean(batch_dices) if batch_dices else 0.0
            pbar.set_postfix({
                'loss': batch_loss.item(),
                'dice': mean_dice
            })
            
            # Visualization
            if vis_count < VISUALIZE_N:
                try:
                    pred_tensor = torch.stack(pred_masks) if pred_masks else torch.zeros(1, 128, 128)
                    visualize_batch(
                        query_img.unsqueeze(0), 
                        query_gt.unsqueeze(0),
                        pred_tensor.unsqueeze(0),
                        epoch, query_idx, valid_labels
                    )
                    vis_count += 1
                except Exception as e:
                    print(f"[VIS ERROR] {e}")
        
        gc.collect()
        torch.cuda.empty_cache()
    
    avg_loss = total_loss / max(num_batches, 1)
    
    # Per-label statistics
    per_label_stats = {}
    for label_name, scores in total_dice.items():
        if scores:
            per_label_stats[label_name] = {
                'mean': np.mean(scores),
                'std': np.std(scores)
            }
    
    return avg_loss, per_label_stats


# ================== VALIDATION FUNCTION ==================
def validate(model, dev_set, support_pool, label_indices, epoch):
    """Validate on dev set"""
    model.eval()
    
    all_dices = defaultdict(list)
    total_dices = []
    
    pbar = tqdm(dev_set, desc=f"Validating epoch {epoch+1}")
    
    for query_img, query_gt, _ in pbar:
        query_img = query_img.to(DEVICE)
        query_gt = query_gt.to(DEVICE)
        query_img_gray = query_img.mean(dim=0, keepdim=True)
        
        valid_labels = [i for i in range(NUM_CLASSES) if query_gt[i].sum() > 10]
        
        if len(valid_labels) == 0:
            continue
        
        with torch.no_grad():
            for label_idx in valid_labels:
                label_name = LABEL_NAMES[label_idx]
                n_total, n_target, _ = SUPPORT_CONFIG[label_name]
                
                # Use fixed seed for reproducibility
                seed = 999 + label_idx
                support_images, support_labels_all = sample_support_adaptive(
                    support_pool, label_indices, label_idx,
                    n_total=n_total, n_target=n_target, seed=seed
                )
                
                support_labels_single = support_labels_all[:, label_idx:label_idx+1]
                support_images_gray = support_images.mean(dim=1, keepdim=True)
                
                logits = model(
                    query_img_gray.unsqueeze(0),
                    support_images_gray.unsqueeze(0),
                    support_labels_single.unsqueeze(0)
                )[0, 0]
                
                threshold = get_threshold_for_label(label_idx)
                dice = dice_score(logits, query_gt[label_idx], threshold=threshold)
                
                all_dices[label_name].append(dice)
                total_dices.append(dice)
    
    # Statistics
    val_stats = {}
    for label_name, scores in all_dices.items():
        if scores:
            val_stats[label_name] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'n': len(scores)
            }
    
    overall_dice = np.mean(total_dices) if total_dices else 0.0
    
    return overall_dice, val_stats


# ================== MAIN TRAINING LOOP ==================
def main():
    print("\n" + "="*70)
    print("UniverSeg Training on Ovatus Dataset")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print("="*70)
    
    # Load data
    print("\n[1] Loading dataset...")
    dataset = OvatusDataset(check_overlap=False)
    
    print("\n[2] Patient-level split...")
    support_pool, dev_set, test_set = patient_level_split_60_20_20(dataset, seed=42)
    print(f"    Support: {len(support_pool)} | Dev: {len(dev_set)} | Test: {len(test_set)}")
    
    # Build label indices
    label_indices = build_label_indices(support_pool)
    
    # Load model
    print("\n[3] Loading UniverSeg (pretrained)...")
    model = universeg(pretrained=True).to(DEVICE)
    
    # Prepare class weights
    class_weights = {}
    for idx, label_name in enumerate(LABEL_NAMES):
        _, _, weight = SUPPORT_CONFIG[label_name]
        class_weights[idx] = weight
    
    print("\n[4] Loss functions and optimizer...")
    dice_loss_fn = WeightedDiceLoss(class_weights=class_weights).to(DEVICE)
    focal_loss_fn = FocalLoss().to(DEVICE)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=EPOCHS // 5, T_mult=1, eta_min=1e-6
    )
    
    # Training loop
    print("\n[5] Starting training...\n")
    
    history = {
        'train_loss': [],
        'val_dice': [],
        'best_val_dice': 0.0,
        'best_epoch': 0
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"training_{timestamp}.json")
    
    start_epoch = 0
    # Load checkpoint if specified
    if RESUME_FROM_CHECKPOINT and os.path.exists(RESUME_FROM_CHECKPOINT):
        print(f"\n[5a] Loading checkpoint: {RESUME_FROM_CHECKPOINT}")
        start_epoch, history = load_checkpoint(
            model, optimizer, scheduler, RESUME_FROM_CHECKPOINT, DEVICE
        )
        print()
    
    for epoch in range(start_epoch, EPOCHS):
        # Training
        train_loss, train_stats = train_epoch(
            model, support_pool, label_indices, dev_set, epoch,
            optimizer, dice_loss_fn, focal_loss_fn, class_weights
        )
        
        # Validation every N epochs
        if (epoch + 1) % max(1, EPOCHS // 10) == 0:
            val_dice, val_stats = validate(model, dev_set, support_pool, label_indices, epoch)
            
            scheduler.step()
            
            history['train_loss'].append(train_loss)
            history['val_dice'].append(val_dice)
            
            # Print summary
            print(f"\n[Epoch {epoch+1:3d}] Loss: {train_loss:.4f} | Val Dice: {val_dice:.4f}")
            print("  Train stats:")
            for label_name, stats in train_stats.items():
                print(f"    {label_name:20s}: {stats['mean']:.3f}±{stats['std']:.3f}")
            
            print("  Val stats:")
            for label_name, stats in val_stats.items():
                n = stats.get('n', 0)
                print(f"    {label_name:20s}: {stats['mean']:.3f}±{stats['std']:.3f} (n={n})")
            
            # Save best checkpoint
            if val_dice > history['best_val_dice']:
                history['best_val_dice'] = val_dice
                history['best_epoch'] = epoch
                
                checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"\n✓ Saved best checkpoint: {checkpoint_path}")
        
        # Save periodic checkpoint
        if (epoch + 1) % (EPOCHS // 5) == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1:03d}_checkpoint.pt")
            save_checkpoint(model, optimizer, scheduler, epoch, history, checkpoint_path)
        
        # Save latest checkpoint (always updated for easy resuming)
        latest_checkpoint = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pt")
        save_checkpoint(model, optimizer, scheduler, epoch, history, latest_checkpoint)
    
    # Save training history
    with open(log_file, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to: {log_file}")
    
    # Final test evaluation
    print("\n" + "="*70)
    print("FINAL TEST EVALUATION")
    print("="*70)
    
    # Load best model
    best_ckpt = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
        print(f"Loaded best model from epoch {history['best_epoch']+1}")
    
    model.eval()
    test_dices = defaultdict(list)
    
    pbar = tqdm(test_set, desc="Testing")
    for query_img, query_gt, _ in pbar:
        query_img = query_img.to(DEVICE)
        query_gt = query_gt.to(DEVICE)
        query_img_gray = query_img.mean(dim=0, keepdim=True)
        
        valid_labels = [i for i in range(NUM_CLASSES) if query_gt[i].sum() > 10]
        
        if len(valid_labels) == 0:
            continue
        
        with torch.no_grad():
            for label_idx in valid_labels:
                label_name = LABEL_NAMES[label_idx]
                n_total, n_target, _ = SUPPORT_CONFIG[label_name]
                
                seed = 777 + label_idx
                support_images, support_labels_all = sample_support_adaptive(
                    support_pool, label_indices, label_idx,
                    n_total=n_total, n_target=n_target, seed=seed
                )
                
                support_labels_single = support_labels_all[:, label_idx:label_idx+1]
                support_images_gray = support_images.mean(dim=1, keepdim=True)
                
                logits = model(
                    query_img_gray.unsqueeze(0),
                    support_images_gray.unsqueeze(0),
                    support_labels_single.unsqueeze(0)
                )[0, 0]
                
                threshold = get_threshold_for_label(label_idx)
                dice = dice_score(logits, query_gt[label_idx], threshold=threshold)
                test_dices[label_name].append(dice)
    
    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    
    all_dices = []
    for label_name in LABEL_NAMES:
        scores = test_dices[label_name]
        if scores:
            mean_dice = np.mean(scores)
            std_dice = np.std(scores)
            median_dice = np.median(scores)
            n_good = sum(1 for d in scores if d >= 0.7)
            n_bad = sum(1 for d in scores if d < 0.3)
            
            print(f"{label_name:20s}: {mean_dice:.3f}±{std_dice:.3f} (median: {median_dice:.3f}, "
                  f"n={len(scores)}, good={n_good}, bad={n_bad})")
            all_dices.extend(scores)
    
    if all_dices:
        print(f"\nOVERALL DICE: {np.mean(all_dices):.4f}±{np.std(all_dices):.4f}")
        print(f"Median: {np.median(all_dices):.4f}")
        print(f"Range: [{np.min(all_dices):.4f}, {np.max(all_dices):.4f}]")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
