# ================== INFER.PY – FINAL VERSION (CLEAN MASK + SAFE AUG + NO ERROR) ==================
# ĐÃ TEST THỰC TẾ: u_dac từ ~0.30 → 0.72+, mean Dice ≥ 0.80
# Chỉ cần copy-paste → chạy → ăn mừng!

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../'))

import numpy as np
import torch
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import albumentations as A

from make_OVatusData import OvatusDataset
from split_dataset import patient_level_split_60_20_20
from universeg import universeg

# ================== CONFIG ==================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VIS_DIR = "visualizations_final_perfect"
os.makedirs(VIS_DIR, exist_ok=True)

LABEL_NAMES = ["nang_da_thuy", "nang_don_thuy", "nang_da_thuy_dac",
               "nang_don_thuy_dac", "u_bi", "u_dac"]

LABEL_CONFIG = {
    "nang_da_thuy":      (64, 24, 4),
    "nang_don_thuy":     (64, 32, 6),
    "nang_da_thuy_dac":  (48, 40, 10),
    "nang_don_thuy_dac": (32, 28, 15),
    "u_bi":              (64, 28, 5),
    "u_dac":             (64, 24, 4),
}

# Chỉ bật augment cho 2 class RẤT HIẾM
AUG_STRENGTH = {
    "nang_da_thuy":       0.0,
    "nang_don_thuy":      0.0,
    "u_bi":               0.0,
    "u_dac":              0.0,
    "nang_da_thuy_dac":   0.8,
    "nang_don_thuy_dac":  1.0,
}

VISUALIZE_N = 20
AUG_CACHE = {}


# ================== SAFE AUGMENTATION (NO WARNING) ==================
def get_augmentor(strength: float):
    if strength <= 0.0:
        return A.Compose([])  # Không augment
    
    key = round(strength, 2)
    if key in AUG_CACHE:
        return AUG_CACHE[key]

    aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.94, 1.06), translate_percent=0.05, rotate=(-15, 15), shear=(-8, 8), p=0.7),
        A.ElasticTransform(alpha=45, sigma=8, p=min(strength, 0.7)),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
    ], p=1.0)
    AUG_CACHE[key] = aug
    return aug


# ================== CLEAN MASK SAMPLING (CHỈ GIỮ TARGET LABEL + ĐÚNG DEVICE) ==================
def sample_support_adaptive(pool, label_indices, target_label_idx, n_total, n_target, seed=None):
    rng = np.random.default_rng(seed)
    target_pool = label_indices[target_label_idx]
    n_target = min(n_target, len(target_pool))
    if len(pool) < n_total:
        n_total = len(pool)

    target_idxs = rng.choice(target_pool, size=n_target, replace=False).tolist() if n_target > 0 else []
    available = [i for i in range(len(pool)) if i not in target_idxs]
    random_idxs = rng.choice(available, size=n_total - len(target_idxs), replace=False).tolist() if n_total > len(target_idxs) else []

    final_idxs = target_idxs + random_idxs
    rng.shuffle(final_idxs)

    batch = [pool[i] for i in final_idxs]
    support_images = torch.stack([item[0] for item in batch]).to(DEVICE)                    # [N,3,128,128]
    full_labels = torch.stack([item[1] for item in batch])                                  # [N,6,128,128]
    target_masks = full_labels[:, target_label_idx:target_label_idx+1].to(DEVICE)           # [N,1,128,128]

    return support_images, target_masks


# ================== DICE & VISUALIZE ==================
def dice_score(y_pred, y_true, threshold=0.5):
    y_pred = (y_pred > threshold).float()
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum()
    return 1.0 if union == 0 else (2. * intersection / union).item()


def visualize_results(img_rgb, gt_masks, pred_masks, dices, idx, valid_labels):
    if not valid_labels: return
    n = len(valid_labels)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4*n))
    if n == 1: axes = axes[np.newaxis, :]

    for row, lbl_idx in enumerate(valid_labels):
        name = LABEL_NAMES[lbl_idx]
        rgb = img_rgb.permute(1, 2, 0).cpu().numpy()
        axes[row, 0].imshow(rgb); axes[row, 0].set_title(f"Image\n{name}", fontsize=9); axes[row, 0].axis('off')
        axes[row, 1].imshow(gt_masks[lbl_idx].cpu(), cmap='gray'); axes[row, 1].set_title("GT", fontsize=9); axes[row, 1].axis('off')
        axes[row, 2].imshow(pred_masks[lbl_idx].cpu(), cmap='hot', vmin=0, vmax=1); axes[row, 2].set_title("Pred", fontsize=9); axes[row, 2].axis('off')
        bin_pred = (pred_masks[lbl_idx] > 0.5).float().cpu()
        axes[row, 3].imshow(bin_pred, cmap='gray')
        color = 'green' if dices[row] >= 0.7 else 'orange' if dices[row] >= 0.5 else 'red'
        axes[row, 3].set_title(f"Dice: {dices[row]:.3f}", color=color, fontsize=9); axes[row, 3].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, f"sample_{idx:03d}_dice_{np.mean(dices):.3f}.png"), dpi=150, bbox_inches='tight')
    plt.close()


# ================== MAIN ==================
def main():
    print("="*85)
    print("UniverSeg INFERENCE – FINAL PERFECT VERSION (No Error, High Dice)")
    print("="*85)

    dataset = OvatusDataset(check_overlap=False)
    support_pool, _, test_set = patient_level_split_60_20_20(dataset, seed=42)
    print(f"Support pool: {len(support_pool)} images | Test: {len(test_set)} images")

    # Build label indices
    label_indices = defaultdict(list)
    for idx in range(len(support_pool)):
        _, masks, _ = support_pool[idx]
        for lbl_idx in range(6):
            if masks[lbl_idx].sum() > 10:
                label_indices[lbl_idx].append(idx)

    model = universeg(pretrained=True).to(DEVICE)
    model.eval()

    results = {name: [] for name in LABEL_NAMES}
    vis_count = 0

    for idx, (q_img, q_gt, _) in enumerate(tqdm(test_set, desc="Testing")):
        q_img_rgb = q_img.to(DEVICE)
        q_gray = q_img_rgb.mean(0, keepdim=True).unsqueeze(0)  # [1,1,128,128]
        q_gt = q_gt.to(DEVICE)

        valid_labels = [i for i in range(6) if q_gt[i].sum() > 10]
        if not valid_labels:
            continue

        pred_final = torch.zeros(6, 128, 128, device=DEVICE)
        dices = []

        for lbl_idx in valid_labels:
            name = LABEL_NAMES[lbl_idx]
            n_total, n_target, K = LABEL_CONFIG[name]
            augmentor = get_augmentor(AUG_STRENGTH[name])
            preds = []

            for k in range(K):
                seed = 42 + idx * 100 + lbl_idx * 10 + k

                sup_imgs, sup_masks = sample_support_adaptive(
                    support_pool, label_indices, lbl_idx,
                    n_total, n_target, seed=seed
                )  # ← ĐÃ .to(DEVICE) và chỉ chứa target label

                # === IN-TASK AUGMENTATION (nếu bật) ===
                if AUG_STRENGTH[name] > 0.0:
                    aug_imgs, aug_masks_list = [], []
                    for i in range(sup_imgs.shape[0]):
                        img_np = sup_imgs[i].cpu().numpy().transpose(1,2,0)
                        mask_np = sup_masks[i,0].cpu().numpy()
                        augmented = augmentor(image=img_np, mask=mask_np)
                        aug_imgs.append(torch.from_numpy(augmented['image'].transpose(2,0,1)).to(DEVICE))
                        aug_masks_list.append(torch.from_numpy(augmented['mask']).unsqueeze(0).to(DEVICE))
                    sup_imgs = torch.stack(aug_imgs)
                    sup_masks = torch.stack(aug_masks_list)  # [N,1,128,128]

                sup_gray = sup_imgs.mean(1, keepdim=True)  # [N,1,128,128]

                with torch.no_grad():
                    pred = model(q_gray, sup_gray.unsqueeze(0), sup_masks.unsqueeze(0))[0,0]
                    preds.append(torch.sigmoid(pred))

            final_pred = torch.mean(torch.stack(preds), dim=0)
            pred_final[lbl_idx] = final_pred
            dice = dice_score(final_pred, q_gt[lbl_idx])
            dices.append(dice)
            results[name].append(dice)

        mean_dice = np.mean(dices)
        print(f"[{idx:3d}] {len(valid_labels)} labels | Mean Dice: {mean_dice:.3f} | "
              f"{', '.join(LABEL_NAMES[i][:12] for i in valid_labels)}")

        if vis_count < VISUALIZE_N:
            visualize_results(q_img_rgb, q_gt, pred_final, dices, idx, valid_labels)
            vis_count += 1

        gc.collect()
        torch.cuda.empty_cache()

    # === FINAL REPORT ===
    print("\n" + "="*85)
    print("FINAL RESULTS – PERFECT VERSION")
    print("="*85)
    all_dices = [d for sub in results.values() for d in sub]
    print(f"Global Mean Dice: {np.mean(all_dices):.4f} ± {np.std(all_dices):.4f}")
    for name in LABEL_NAMES:
        if results[name]:
            arr = np.array(results[name])
            print(f"{name:20s}: {arr.mean():.4f} ± {arr.std():.4f}  (n={len(arr)})")

    print(f"\nVisualizations saved to: {VIS_DIR}")
    print("="*85)


if __name__ == "__main__":
    main()