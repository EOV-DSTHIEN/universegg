# ================== TRAINING SCRIPT: UniverSeg with Adaptive Sampling ==================
# Train UniverSeg model cho multi-label medical image segmentation
# Dựa trên infer.py và make_OVatusData.py
#
# Key features:
# 1. Adaptive stratified sampling cho imbalanced labels
# 2. Per-label loss weighting
# 3. Learning rate scheduling
# 4. Validation tracking
# 5. Model checkpointing

import os, sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import json
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from make_OVatusData import OvatusDataset
from split_dataset import patient_level_split_60_20_20
from universeg import universeg

# ================== TRAINING CONFIG ==================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create timestamp for this run
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = f"runs/run_{TIMESTAMP}"
LOGS_DIR = os.path.join(RUN_DIR, "logs")
CHECKPOINT_DIR = os.path.join(RUN_DIR, "checkpoints")
VIS_DIR = os.path.join(RUN_DIR, "visualizations")
TENSORBOARD_DIR = os.path.join(RUN_DIR, "tensorboard")

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)

print(f"\n📁 Run directory: {RUN_DIR}")
print(f"📊 TensorBoard logs: {TENSORBOARD_DIR}")

LABEL_NAMES = ["nang_da_thuy", "nang_don_thuy", "nang_da_thuy_dac", 
               "nang_don_thuy_dac", "u_bi", "u_dac"]

# Thresholds for validation
THRESHOLDS = {
    "nang_da_thuy": 0.45,
    "nang_don_thuy": 0.45,
    "nang_da_thuy_dac": 0.40,
    "nang_don_thuy_dac": 0.40,
    "u_bi": 0.50,
    "u_dac": 0.35,
}

# Adaptive sampling config (same as infer.py)
LABEL_CONFIG = {
    "nang_da_thuy":      (64, 24, 3),
    "nang_don_thuy":     (64, 32, 5),
    "nang_da_thuy_dac":  (48, 40, 8),
    "nang_don_thuy_dac": (32, 28, 10),
    "u_bi":              (64, 60, 5),
    "u_dac":             (64, 60, 10),
}

# ================== TRAINING HYPERPARAMETERS ==================
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 50
WARMUP_EPOCHS = 5
BATCH_SIZE = 8  # Number of query images per batch
VALIDATE_EVERY = 5
SAVE_EVERY = 10
EARLY_STOPPING_PATIENCE = 15

# Label weights (inverse frequency weighting)
LABEL_WEIGHTS = {
    "nang_da_thuy": 1.0,       # Common
    "nang_don_thuy": 1.0,
    "nang_da_thuy_dac": 2.0,   # Rare
    "nang_don_thuy_dac": 3.0,  # Very rare
    "u_bi": 1.2,
    "u_dac": 1.5,
}

VISUALIZE_N = 10


class TrainingLogger:
    """Track metrics during training with TensorBoard"""
    def __init__(self, log_dir, tensorboard_dir):
        self.log_dir = log_dir
        self.history = defaultdict(list)
        self.log_file = os.path.join(log_dir, "training_log.json")
        
        # TensorBoard writer
        self.writer = SummaryWriter(tensorboard_dir)
        print(f"✅ TensorBoard initialized: tensorboard --logdir={tensorboard_dir}")
    
    def log(self, epoch, **metrics):
        for key, value in metrics.items():
            self.history[key].append(value)
            
            # Log to TensorBoard
            if isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    self.writer.add_scalar(f"{key}/{sub_key}", sub_val, epoch)
            else:
                self.writer.add_scalar(key, value, epoch)
        
        # Save to JSON
        with open(self.log_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def log_per_label_metrics(self, epoch, per_label_dice, metric_name="per_label_dice"):
        """Log per-label metrics to TensorBoard"""
        for label_name, dice in per_label_dice.items():
            self.writer.add_scalar(f"{metric_name}/{label_name}", dice, epoch)
    
    def plot(self):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        if 'train_loss' in self.history:
            axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
            axes[0, 0].plot(self.history.get('val_loss', []), label='Val Loss')
            axes[0, 0].set_title('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid()
        
        if 'train_dice' in self.history:
            axes[0, 1].plot(self.history['train_dice'], label='Train Dice')
            axes[0, 1].plot(self.history.get('val_dice', []), label='Val Dice')
            axes[0, 1].set_title('Mean Dice')
            axes[0, 1].legend()
            axes[0, 1].grid()
        
        if 'learning_rate' in self.history:
            axes[1, 0].plot(self.history['learning_rate'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].grid()
        
        if 'per_label_dice' in self.history:
            # Plot per-label performance
            per_label = self.history['per_label_dice']
            if len(per_label) > 0:
                last_dice = per_label[-1]
                axes[1, 1].bar(range(len(LABEL_NAMES)), 
                             [last_dice.get(name, 0) for name in LABEL_NAMES],
                             color=['green' if last_dice.get(name, 0) > 0.7 else 'orange' 
                                   for name in LABEL_NAMES])
                axes[1, 1].set_xticks(range(len(LABEL_NAMES)))
                axes[1, 1].set_xticklabels([n[:8] for n in LABEL_NAMES], rotation=45)
                axes[1, 1].set_title('Per-Label Dice (Latest)')
                axes[1, 1].set_ylim([0, 1])
                axes[1, 1].grid(axis='y')
        
        plt.tight_layout()
        plot_path = os.path.join(self.log_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n📊 Training curves saved: {plot_path}")
    
    def close(self):
        """Close TensorBoard writer"""
        self.writer.close()


def build_label_indices(dataset):
    """Build index mapping: label_idx -> list of sample indices"""
    label_indices = defaultdict(list)
    
    print("\n[INFO] Building label indices...")
    for idx in range(len(dataset)):
        _, masks, _ = dataset[idx]
        for label_idx in range(masks.shape[0]):
            if masks[label_idx].sum() > 10:
                label_indices[label_idx].append(idx)
    
    print("[INFO] Label distribution in support pool:")
    for label_idx, indices in label_indices.items():
        label_name = LABEL_NAMES[label_idx]
        pct = 100 * len(indices) / len(dataset)
        
        if len(indices) < 20:
            rarity = "❌ VERY RARE"
        elif len(indices) < 50:
            rarity = "⚠️  RARE"
        elif len(indices) < 100:
            rarity = "✓  MEDIUM"
        else:
            rarity = "✅ COMMON"
        
        print(f"  [{label_idx}] {label_name:20s}: {len(indices):3d} ({pct:5.1f}%) {rarity}")
    
    return label_indices


def sample_support_adaptive(pool, label_indices, target_label_idx, 
                            n_total, n_target, seed=None):
    """Adaptive stratified sampling"""
    rng = np.random.default_rng(seed) if seed else np.random
    
    target_pool = label_indices[target_label_idx]
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


def dice_loss(pred, target, threshold=0.5):
    """Soft Dice Loss"""
    pred_binary = (pred > threshold).float()
    
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum()
    
    if union == 0:
        return 0.0 if target.sum() == 0 else 1.0
    
    dice = 2 * intersection / (union + 1e-7)
    return 1.0 - dice


def dice_score(y_pred, y_true, threshold=0.5):
    """Binary Dice metric"""
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


class BCEDiceLoss(nn.Module):
    """Combined BCE + Dice Loss with per-label weighting"""
    def __init__(self, label_weights):
        super().__init__()
        self.label_weights = torch.tensor(
            [label_weights.get(name, 1.0) for name in LABEL_NAMES],
            dtype=torch.float32
        )
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, logits, targets):
        """
        logits: [B, 6, 128, 128]
        targets: [B, 6, 128, 128]
        """
        self.label_weights = self.label_weights.to(logits.device)
        
        # BCE loss
        bce_loss = self.bce(logits, targets)  # [B, 6, 128, 128]
        
        # Average over spatial dimensions, then weight by label
        bce_loss = bce_loss.mean(dim=(0, 2, 3))  # [6]
        bce_loss = (bce_loss * self.label_weights).mean()
        
        # Sigmoid for Dice
        pred = torch.sigmoid(logits)
        
        # Dice loss per label
        dice_losses = []
        for label_idx in range(6):
            d_loss = dice_loss(pred[:, label_idx], targets[:, label_idx], 
                             threshold=get_threshold_for_label(label_idx))
            dice_losses.append(d_loss)
        
        dice_losses = torch.tensor(dice_losses, device=logits.device)
        dice_loss_weighted = (dice_losses * self.label_weights).mean()
        
        # Combine
        total_loss = 0.7 * bce_loss + 0.3 * dice_loss_weighted
        
        return total_loss, bce_loss.item(), dice_loss_weighted.item()


def train_epoch(model, support_pool, label_indices, epoch, logger, loss_fn, optimizer):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0.0
    epoch_dice = 0.0
    num_batches = 0
    
    # Sample query images for this epoch
    num_samples = min(BATCH_SIZE * 50, len(support_pool))  # Up to 50 batches
    query_indices = np.random.choice(len(support_pool), size=num_samples, replace=True)
    
    pbar = tqdm(query_indices, desc=f"[Train {epoch}]")
    
    for batch_idx, query_idx in enumerate(pbar):
        query_img, query_gt, _ = support_pool[query_idx]
        query_img_rgb = query_img.unsqueeze(0).to(DEVICE)
        query_img_gray = query_img_rgb.mean(dim=1, keepdim=True)
        query_gt = query_gt.unsqueeze(0).to(DEVICE)
        
        # Find valid labels
        valid_labels = [i for i in range(6) if query_gt[0, i].sum() > 10]
        
        if len(valid_labels) == 0:
            continue
        
        # Predict each label
        final_predictions = torch.zeros(1, 6, 128, 128, device=DEVICE)
        
        for label_idx in valid_labels:
            label_name = LABEL_NAMES[label_idx]
            n_total, n_target, _ = LABEL_CONFIG[label_name]
            
            # Single forward pass (no ensemble during training)
            seed = epoch * 1000 + query_idx
            support_images, support_labels_all = sample_support_adaptive(
                support_pool, label_indices, label_idx,
                n_total=n_total, n_target=n_target, seed=seed
            )
            
            support_labels_single = support_labels_all[:, label_idx:label_idx+1]
            support_images_gray = support_images.mean(dim=1, keepdim=True)
            
            logits = model(
                query_img_gray,
                support_images_gray.unsqueeze(0),
                support_labels_single.unsqueeze(0)
            )
            
            final_predictions[0, label_idx] = torch.sigmoid(logits)[0, 0]
        
        # Compute loss only on valid labels
        loss_input = torch.zeros(1, 6, 128, 128, device=DEVICE)
        loss_target = torch.zeros(1, 6, 128, 128, device=DEVICE)
        
        for label_idx in valid_labels:
            loss_input[0, label_idx] = final_predictions[0, label_idx]
            loss_target[0, label_idx] = query_gt[0, label_idx]
        
        total_loss, bce_val, dice_val = loss_fn(loss_input, loss_target)
        
        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Metrics
        epoch_loss += total_loss.item()
        
        # Compute dice score
        dices = []
        for label_idx in valid_labels:
            d = dice_score(final_predictions[0, label_idx], query_gt[0, label_idx],
                         threshold=get_threshold_for_label(label_idx))
            dices.append(d)
        
        if dices:
            epoch_dice += np.mean(dices)
        
        num_batches += 1
        
        # Log to TensorBoard every 10 batches
        if (batch_idx + 1) % 10 == 0:
            step = epoch * (num_samples // 10) + batch_idx // 10
            logger.writer.add_scalar('train/batch_loss', total_loss.item(), step)
            logger.writer.add_scalar('train/batch_dice', np.mean(dices) if dices else 0.0, step)
            logger.writer.add_scalar('train/bce_loss', bce_val, step)
            logger.writer.add_scalar('train/dice_loss', dice_val, step)
        
        pbar.set_postfix({
            'loss': total_loss.item(),
            'dice': np.mean(dices) if dices else 0.0
        })
        
        gc.collect()
        torch.cuda.empty_cache()
    
    avg_loss = epoch_loss / max(num_batches, 1)
    avg_dice = epoch_dice / max(num_batches, 1)
    
    return avg_loss, avg_dice


def validate(model, support_pool, label_indices, dev_set, loss_fn):
    """Validate on dev set"""
    model.eval()
    val_loss = 0.0
    val_dices = []
    per_label_dice = defaultdict(list)
    num_samples = 0
    
    with torch.no_grad():
        for query_idx, (query_img, query_gt, _) in enumerate(tqdm(dev_set, desc="[Validating]")):
            query_img_rgb = query_img.unsqueeze(0).to(DEVICE)
            query_img_gray = query_img_rgb.mean(dim=1, keepdim=True)
            query_gt = query_gt.unsqueeze(0).to(DEVICE)
            
            valid_labels = [i for i in range(6) if query_gt[0, i].sum() > 10]
            
            if len(valid_labels) == 0:
                continue
            
            final_predictions = torch.zeros(1, 6, 128, 128, device=DEVICE)
            
            for label_idx in valid_labels:
                label_name = LABEL_NAMES[label_idx]
                n_total, n_target, ensemble_k = LABEL_CONFIG[label_name]
                
                # Ensemble on validation
                preds = []
                for k in range(ensemble_k):
                    seed = 42 + query_idx * 100 + label_idx * 10 + k
                    support_images, support_labels_all = sample_support_adaptive(
                        support_pool, label_indices, label_idx,
                        n_total=n_total, n_target=n_target, seed=seed
                    )
                    
                    support_labels_single = support_labels_all[:, label_idx:label_idx+1]
                    support_images_gray = support_images.mean(dim=1, keepdim=True)
                    
                    logits = model(
                        query_img_gray,
                        support_images_gray.unsqueeze(0),
                        support_labels_single.unsqueeze(0)
                    )
                    
                    preds.append(torch.sigmoid(logits)[0, 0])
                
                final_predictions[0, label_idx] = torch.mean(torch.stack(preds), dim=0)
            
            # Loss
            loss_input = torch.zeros(1, 6, 128, 128, device=DEVICE)
            loss_target = torch.zeros(1, 6, 128, 128, device=DEVICE)
            
            for label_idx in valid_labels:
                loss_input[0, label_idx] = final_predictions[0, label_idx]
                loss_target[0, label_idx] = query_gt[0, label_idx]
            
            total_loss, _, _ = loss_fn(loss_input, loss_target)
            val_loss += total_loss.item()
            
            # Dice scores
            dices = []
            for label_idx in valid_labels:
                d = dice_score(final_predictions[0, label_idx], query_gt[0, label_idx],
                             threshold=get_threshold_for_label(label_idx))
                dices.append(d)
                per_label_dice[label_idx].append(d)
            
            if dices:
                val_dices.append(np.mean(dices))
            
            num_samples += 1
            gc.collect()
            torch.cuda.empty_cache()
    
    avg_val_loss = val_loss / max(num_samples, 1)
    avg_val_dice = np.mean(val_dices) if val_dices else 0.0
    
    per_label_avg = {}
    for label_idx, scores in per_label_dice.items():
        per_label_avg[LABEL_NAMES[label_idx]] = np.mean(scores)
    
    return avg_val_loss, avg_val_dice, per_label_avg


def main():
    print("\n" + "="*70)
    print("UniverSeg - TRAINING with Adaptive Stratified Sampling")
    print("="*70)
    print("Configuration:")
    print(f"  Device:           {DEVICE}")
    print(f"  Learning Rate:    {LEARNING_RATE}")
    print(f"  Batch Size:       {BATCH_SIZE}")
    print(f"  Epochs:           {EPOCHS}")
    print(f"  Warmup Epochs:    {WARMUP_EPOCHS}")
    print("="*70)
    
    # Load dataset
    print("\n[1] Loading dataset...")
    dataset = OvatusDataset(check_overlap=False)
    
    # Split
    print("\n[2] Patient-level split...")
    support_pool, dev_set, test_set = patient_level_split_60_20_20(dataset, seed=42)
    print(f"    Support: {len(support_pool)} | Dev: {len(dev_set)} | Test: {len(test_set)}")
    
    # Build label indices
    label_indices = build_label_indices(support_pool)
    
    # Load model
    print("\n[3] Loading UniverSeg...")
    model = universeg(pretrained=True).to(DEVICE)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        else:
            return 0.5 ** ((epoch - WARMUP_EPOCHS) // 10)
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Loss function
    loss_fn = BCEDiceLoss(LABEL_WEIGHTS).to(DEVICE)
    
    # Logger
    logger = TrainingLogger(LOGS_DIR, TENSORBOARD_DIR)
    
    # Training loop
    print(f"\n[4] Starting training...\n")
    
    best_val_dice = 0.0
    best_epoch = 0
    no_improve_count = 0
    
    for epoch in range(EPOCHS):
        # Train
        train_loss, train_dice = train_epoch(
            model, support_pool, label_indices, epoch, logger, loss_fn, optimizer
        )
        
        # LR step
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # Validate
        if (epoch + 1) % VALIDATE_EVERY == 0:
            print(f"\n[Epoch {epoch+1}] Validating...")
            val_loss, val_dice, per_label_dice = validate(
                model, support_pool, label_indices, dev_set, loss_fn
            )
            
            print(f"\n[Epoch {epoch+1}] Results:")
            print(f"  Train Loss:  {train_loss:.4f} | Train Dice: {train_dice:.4f}")
            print(f"  Val Loss:    {val_loss:.4f} | Val Dice:   {val_dice:.4f}")
            print(f"  LR:          {current_lr:.6f}")
            
            print(f"\n  Per-label validation Dice:")
            for label_name, dice in per_label_dice.items():
                print(f"    {label_name:20s}: {dice:.4f}")
            
            # Logging
            logger.log(
                epoch,
                train_loss=train_loss,
                train_dice=train_dice,
                val_loss=val_loss,
                val_dice=val_dice,
                learning_rate=current_lr,
                per_label_dice=per_label_dice
            )
            
            # Log per-label metrics to TensorBoard
            logger.log_per_label_metrics(epoch, per_label_dice, "val/per_label_dice")
            
            # Save best model
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                best_epoch = epoch
                no_improve_count = 0
                
                checkpoint_name = f"best_model_epoch_{epoch+1:03d}_dice_{val_dice:.4f}.pt"
                checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_dice': val_dice,
                    'val_loss': val_loss,
                    'timestamp': TIMESTAMP,
                }, checkpoint_path)
                
                print(f"  ✅ Best model saved: {checkpoint_path}")
            else:
                no_improve_count += 1
        
        # Save periodic checkpoint
        if (epoch + 1) % SAVE_EVERY == 0:
            checkpoint_name = f"checkpoint_epoch_{epoch+1:03d}_dice_{train_dice:.4f}.pt"
            checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'timestamp': TIMESTAMP,
            }, checkpoint_path)
            print(f"  💾 Checkpoint saved: {checkpoint_path}")
        
        # Early stopping
        if no_improve_count >= EARLY_STOPPING_PATIENCE:
            print(f"\n⏸️  Early stopping at epoch {epoch+1}")
            break
    
    # Plot training curves
    logger.plot()
    
    # Close TensorBoard writer
    logger.close()
    
    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    print(f"Best validation Dice: {best_val_dice:.4f} (epoch {best_epoch+1})")
    print(f"Run directory:        {RUN_DIR}/")
    print(f"Checkpoints saved to: {CHECKPOINT_DIR}/")
    print(f"Logs saved to:        {LOGS_DIR}/")
    print(f"\n📊 TensorBoard command:")
    print(f"   tensorboard --logdir={TENSORBOARD_DIR}")
    print("="*70 + "\n")
    
    # Save training config
    config_path = os.path.join(RUN_DIR, "training_config.json")
    config = {
        "timestamp": TIMESTAMP,
        "best_val_dice": best_val_dice,
        "best_epoch": best_epoch + 1,
        "total_epochs": epoch + 1,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_epochs": WARMUP_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "label_weights": LABEL_WEIGHTS,
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"📋 Config saved: {config_path}\n")


if __name__ == "__main__":
    main()
