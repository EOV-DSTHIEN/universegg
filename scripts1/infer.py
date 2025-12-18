# ================== FINAL FIX: Adaptive Stratified Sampling ==================
# Giải quyết class imbalance bằng cách:
# 1. Đảm bảo support set có ĐỦ examples cho label đang predict
# 2. Tăng số lượng examples cho rare labels
# 3. Giảm ensemble cho common labels để tăng tốc

import os, sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../'))

import numpy as np
import torch
import torch.nn.functional as F
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

from make_OVatusData import OvatusDataset
from split_dataset import patient_level_split_60_20_20
from universeg import universeg

# ================== ADAPTIVE CONFIG ==================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VIS_DIR = "visualizations_final"
os.makedirs(VIS_DIR, exist_ok=True)

LABEL_NAMES = ["nang_da_thuy", "nang_don_thuy", "nang_da_thuy_dac", 
               "nang_don_thuy_dac", "u_bi", "u_dac"]

# Thresholds for each label
THRESHOLDS = {
    "nang_da_thuy": 0.45,
    "nang_don_thuy": 0.45,
    "nang_da_thuy_dac": 0.40,
    "nang_don_thuy_dac": 0.40,
    "u_bi": 0.50,
    "u_dac": 0.35,   # RẤT QUAN TRỌNG
}

# Adaptive parameters based on label frequency
LABEL_CONFIG = {
    # Format: (support_total, support_target, ensemble_k)
    "nang_da_thuy":      (64, 24, 3),   # Common: fewer target examples, less ensemble
    "nang_don_thuy":     (64, 32, 5),   # Medium
    "nang_da_thuy_dac":  (48, 40, 8),   # Rare: MORE target examples, MORE ensemble
    "nang_don_thuy_dac": (32, 28, 10),  # Very rare: MAXIMUM target %, MAXIMUM ensemble
    "u_bi":              (64, 60, 5),   # Common-medium
    "u_dac":             (64, 60, 10),   # Common
}

VISUALIZE_N = 20


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
        
        # Determine rarity
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
    """
    Adaptive stratified sampling:
    - For rare labels: maximize target examples (e.g., 28 out of 32)
    - For common labels: use balanced sampling (e.g., 24 out of 64)
    """
    rng = np.random.default_rng(seed) if seed else np.random
    
    target_pool = label_indices[target_label_idx]
    
    # Adjust n_target if not enough samples
    n_target = min(n_target, len(target_pool))
    
    # If total available samples < n_total, reduce total
    if len(pool) < n_total:
        n_total = len(pool)
        n_target = min(n_target, n_total)
    
    # Sample target examples
    if n_target > 0:
        target_idxs = rng.choice(target_pool, size=n_target, replace=False).tolist()
    else:
        target_idxs = []
    
    # Sample random examples (excluding already selected)
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
    
    # Stack labels (all 6 channels)
    support_labels = torch.stack(support_labels_list).to(DEVICE)  # [N, 6, 128, 128]
    
    return support_images, support_labels


def dice_score(y_pred, y_true, threshold=0.5):
    """Binary Dice with empty handling"""
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


def visualize_results(image, gt_masks, pred_masks, dices, idx, valid_labels):
    """Visualization with adaptive layout"""
    if len(valid_labels) == 0:
        return
    
    n_labels = len(valid_labels)
    fig, axes = plt.subplots(n_labels, 4, figsize=(16, 4 * n_labels))
    
    if n_labels == 1:
        axes = axes[np.newaxis, :]
    
    for row, label_idx in enumerate(valid_labels):
        label_name = LABEL_NAMES[label_idx]
        
        # Original image
        rgb = image.permute(1, 2, 0).cpu().numpy()
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(f"Image\n{label_name}", fontsize=9)
        axes[row, 0].axis('off')
        
        # Ground truth
        gt = gt_masks[label_idx].cpu().numpy()
        axes[row, 1].imshow(gt, cmap='gray')
        gt_pixels = gt.sum()
        axes[row, 1].set_title(f"GT\n{gt_pixels:.0f} px", fontsize=9)
        axes[row, 1].axis('off')
        
        # Soft prediction
        soft = pred_masks[label_idx].cpu().numpy()
        axes[row, 2].imshow(soft, cmap='hot', vmin=0, vmax=1)
        axes[row, 2].set_title(f"Probability", fontsize=9)
        axes[row, 2].axis('off')
        
        # Binary prediction
        threshold = get_threshold_for_label(label_idx)
        binary = (pred_masks[label_idx] > threshold).float().cpu().numpy()
        axes[row, 3].imshow(binary, cmap='gray')
        
        dice = dices[row]
        if dice >= 0.7:
            color, symbol = 'green', '✓'
        elif dice >= 0.5:
            color, symbol = 'orange', '~'
        else:
            color, symbol = 'red', '✗'
        
        axes[row, 3].set_title(f"{symbol} Dice: {dice:.3f}", 
                               color=color, weight='bold', fontsize=9)
        axes[row, 3].axis('off')
    
    plt.tight_layout()
    mean_dice = np.mean(dices)
    filepath = os.path.join(VIS_DIR, f"sample_{idx:03d}_dice_{mean_dice:.3f}.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print("\n" + "="*70)
    print("UniverSeg - ADAPTIVE STRATIFIED SAMPLING (Final Fix)")
    print("="*70)
    print("KEY FEATURES:")
    print("  1. Adaptive support size based on label rarity")
    print("  2. Higher target % for rare labels (e.g., 90% for very rare)")
    print("  3. More ensemble iterations for rare labels")
    print("  4. Sigmoid activation for multi-label independence")
    print("="*70)
    
    # Load dataset
    print("\n[1] Loading dataset...")
    dataset = OvatusDataset(check_overlap=False)  # Already checked
    
    # Split
    print("\n[2] Patient-level split...")
    support_pool, dev_set, test_set = patient_level_split_60_20_20(dataset, seed=42)
    print(f"    Support: {len(support_pool)} | Test: {len(test_set)}")
    
    # Build label indices
    label_indices = build_label_indices(support_pool)
    
    # Validate config
    print("\n[3] Adaptive configuration:")
    for label_name, (n_total, n_target, ensemble_k) in LABEL_CONFIG.items():
        label_idx = LABEL_NAMES.index(label_name)
        available = len(label_indices[label_idx])
        target_pct = 100 * n_target / n_total
        
        if available < n_target:
            print(f"  ⚠️  {label_name:20s}: want {n_target:2d}, have {available:2d} "
                  f"({target_pct:4.0f}% target, K={ensemble_k})")
        else:
            print(f"  ✓  {label_name:20s}: {n_total:2d} total, {n_target:2d} target "
                  f"({target_pct:4.0f}%, K={ensemble_k})")
    
    # Load model
    print("\n[4] Loading UniverSeg...")
    model = universeg(pretrained=True).to(DEVICE)
    model.eval()
    
    # Inference
    print(f"\n[5] Running adaptive inference...\n")
    
    all_dices_per_label = {name: [] for name in LABEL_NAMES}
    all_dices_global = []
    vis_count = 0
    
    for test_idx, (query_img, query_gt, img_path) in enumerate(tqdm(test_set, desc="Testing")):
        query_img_rgb = query_img.to(DEVICE)
        query_img_gray = query_img_rgb.mean(dim=0, keepdim=True)
        query_gt = query_gt.to(DEVICE)
        
        # Find valid labels
        valid_labels = [i for i in range(6) if query_gt[i].sum() > 10]
        
        if len(valid_labels) == 0:
            continue
        
        # Predict each label with adaptive config
        final_predictions = torch.zeros(6, 128, 128, device=DEVICE)
        sample_dices = []
        
        for label_idx in valid_labels:
            label_name = LABEL_NAMES[label_idx]
            n_total, n_target, ensemble_k = LABEL_CONFIG[label_name]
            
            # Ensemble predictions
            preds = []
            for k in range(ensemble_k):
                seed = 42 + test_idx * 100 + label_idx * 10 + k
                
                # Adaptive stratified sampling
                support_images, support_labels_all = sample_support_adaptive(
                    support_pool, label_indices, label_idx,
                    n_total=n_total, n_target=n_target, seed=seed
                )
                
                # Extract target label
                support_labels_single = support_labels_all[:, label_idx:label_idx+1]
                support_images_gray = support_images.mean(dim=1, keepdim=True)
                
                with torch.no_grad():
                    logits = model(
                        query_img_gray.unsqueeze(0),
                        support_images_gray.unsqueeze(0),
                        support_labels_single.unsqueeze(0)
                    )[0]
                    
                    soft_pred = torch.sigmoid(logits)
                    preds.append(soft_pred[0])
            
            # Average ensemble
            final_pred_label = torch.mean(torch.stack(preds), dim=0)
            final_predictions[label_idx] = final_pred_label
            
            # Compute Dice with label-specific threshold
            threshold = get_threshold_for_label(label_idx)
            dice = dice_score(final_pred_label, query_gt[label_idx], threshold=threshold)
            sample_dices.append(dice)
            all_dices_global.append(dice)
            all_dices_per_label[label_name].append(dice)
        
        # Summary
        mean_dice = np.mean(sample_dices)
        print(f"[{test_idx:3d}] {len(valid_labels)} labels | Mean Dice: {mean_dice:.3f} | "
              f"{', '.join([LABEL_NAMES[i][:8] for i in valid_labels])}")
        
        # Visualization
        if vis_count < VISUALIZE_N:
            visualize_results(query_img_rgb, query_gt, final_predictions, 
                            sample_dices, test_idx, valid_labels)
            vis_count += 1
        
        gc.collect()
        torch.cuda.empty_cache()
    
    # Final results
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Test samples:      {len(test_set)}")
    print(f"Total predictions: {len(all_dices_global)}")
    
    if len(all_dices_global) > 0:
        print(f"\nOVERALL PERFORMANCE:")
        print(f"  Mean Dice:   {np.mean(all_dices_global):.4f} ± {np.std(all_dices_global):.4f}")
        print(f"  Median Dice: {np.median(all_dices_global):.4f}")
        print(f"  Range:       [{np.min(all_dices_global):.4f}, {np.max(all_dices_global):.4f}]")
        
        print(f"\nPER-LABEL PERFORMANCE:")
        print(f"{'Label':<20s} {'Mean±Std':<15s} {'Median':<8s} {'n':<4s} {'Good':<5s} {'Bad':<5s}")
        print("-" * 70)
        
        for label_name in LABEL_NAMES:
            scores = all_dices_per_label[label_name]
            if len(scores) > 0:
                mean_dice = np.mean(scores)
                std_dice = np.std(scores)
                median_dice = np.median(scores)
                n_good = sum(1 for d in scores if d >= 0.7)
                n_bad = sum(1 for d in scores if d < 0.3)
                
                print(f"{label_name:<20s} {mean_dice:.3f}±{std_dice:.3f}    "
                      f"{median_dice:.3f}    {len(scores):<4d} {n_good:<5d} {n_bad:<5d}")
            else:
                print(f"{label_name:<20s} {'N/A':<15s}")
    
    print(f"\nVisualizations: {VIS_DIR}/ ({vis_count} images)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()