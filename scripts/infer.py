import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../'))
import numpy as np
import torch
import matplotlib.pyplot as plt
from make_OVatusData import OVatusDataset
from split_support_test import split_dataset
from universeg import universeg
from tqdm import tqdm
import gc
import matplotlib.patches as mpatches

# === HÀM VISUALIZE CŨ (giữ lại để tương thích) ===
def visualize(img, gt_mask, pred_mask, save_path=None):
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(np.transpose(img.numpy(), (1,2,0)))
    plt.title("Image")
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(gt_mask.numpy()[0], cmap='gray')
    plt.title("Ground Truth")
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.imshow(pred_mask.numpy()[0], cmap='gray')
    plt.title("Prediction")
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

# === HÀM MỚI: visualized() - VẼ VÀ LƯU MẪU ĐẸP HƠN ===
def visualized(img, gt_masks, pred_masks, dice_scores_per_label, idx, save_dir="visualizations"):
    """
    Vẽ và lưu visualization cho 1 mẫu test
    - img: tensor [3, H, W]
    - gt_masks: tensor [L, H, W]
    - pred_masks: tensor [L, H, W]
    - dice_scores_per_label: list[float] - Dice cho từng label
    - idx: int - chỉ số mẫu
    - save_dir: nơi lưu ảnh
    """
    os.makedirs(save_dir, exist_ok=True)
    
    num_labels = min(gt_masks.shape[0], pred_masks.shape[0])
    cols = 3
    rows = num_labels
    
    plt.figure(figsize=(4*cols, 4*rows))
    
    for label_idx in range(num_labels):
        # Ảnh gốc
        plt.subplot(rows, cols, label_idx * cols + 1)
        plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
        plt.title(f"Image (Label {label_idx+1})")
        plt.axis('off')
        
        # Ground Truth
        plt.subplot(rows, cols, label_idx * cols + 2)
        plt.imshow(gt_masks[label_idx].numpy(), cmap='gray')
        plt.title("Ground Truth")
        plt.axis('off')
        
        # Prediction + Dice
        plt.subplot(rows, cols, label_idx * cols + 3)
        plt.imshow(pred_masks[label_idx].numpy(), cmap='gray')
        dice_val = dice_scores_per_label[label_idx] if label_idx < len(dice_scores_per_label) else 0.0
        plt.title(f"Pred (Dice: {dice_val:.3f})", color='green' if dice_val > 0.7 else 'red')
        plt.axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"sample_{idx:03d}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"   Đã lưu visualization: {save_path}")

# === CÁC HÀM ĐÁNH GIÁ ===
def dice_score(pred, target, eps=1e-6):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2. * intersection + eps) / (union + eps)

def iou_score(pred, target, eps=1e-6):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + eps) / (union + eps)

def accuracy_score(pred, target):
    return (pred == target).float().mean().item()

# === MAIN ===
def main():
    print("=== BẮT ĐẦU INFERENCE OVATUS ===")
    dataset = OVatusDataset()
    support_idxs, test_idxs = split_dataset(dataset, support_frac=0.4)
    support_set = [dataset[i] for i in support_idxs]
    test_set = [dataset[i] for i in test_idxs]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Sử dụng thiết bị: {device}")
    print(f"Số mẫu support: {len(support_set)}, số mẫu test: {len(test_set)}")

    # Prepare support data
    support_images = torch.stack([img for img, mask, _ in support_set])  # [N, 3, H, W]
    support_images_gray = support_images.mean(dim=1, keepdim=True)  # [N, 1, H, W]
    
    num_labels = max(mask.shape[0] for _, mask, _ in support_set)
    
    support_masks_padded = []
    for _, mask, _ in support_set:
        if mask.shape[0] < num_labels:
            pad = torch.zeros((num_labels - mask.shape[0], 128, 128))
            mask = torch.cat([mask, pad], dim=0)
        support_masks_padded.append(mask[:num_labels])
    
    support_masks = torch.stack(support_masks_padded)  # [N, num_labels, H, W]
    
    print(f"Support images shape: {support_images_gray.shape}")
    print(f"Support masks shape: {support_masks.shape}")
    print(f"Number of labels: {num_labels}")

    # Load model
    model = universeg(pretrained=True).to(device)
    model.eval()
    print("Model loaded and set to eval mode")

    dice_scores = []
    visualized_count = 0
    max_visualize = 20
      # Chỉ visualize 5 mẫu đầu
    
    print("\n=== Đang chạy inference trên test set ===")
    
    for idx, (img, gt_masks, img_path) in enumerate(tqdm(test_set, desc="Inference")):
        img_gray = img.mean(dim=0, keepdim=True).unsqueeze(0).to(device)  # [1, 1, H, W]
        gt_masks = gt_masks.to(device)
        
        pred_masks_list = []
        dice_per_label = []
        
        # Process each label
        for label_idx in range(min(gt_masks.shape[0], num_labels)):
            support_imgs_batch = support_images_gray.unsqueeze(0).to(device)
            support_masks_batch = support_masks[:, label_idx:label_idx+1].unsqueeze(0).to(device)
            
            try:
                with torch.no_grad():
                    logits = model(img_gray, support_imgs_batch, support_masks_batch)
                pred = torch.sigmoid(logits).squeeze(0)
                pred_bin = (pred > 0.5).float()
                pred_masks_list.append(pred_bin)
                
                # Tính Dice cho label này
                dice = dice_score(pred_bin, gt_masks[label_idx:label_idx+1])
                dice_per_label.append(dice.item())
                dice_scores.append(dice.item())
                
            except Exception as e:
                print(f"\nError at test {idx}, label {label_idx}: {e}")
                import traceback
                traceback.print_exc()
                dice_per_label.append(0.0)
                continue
        
        if not pred_masks_list:
            print(f"Warning: No predictions for test sample {idx}")
            continue
            
        pred_masks = torch.cat(pred_masks_list, dim=0)  # [L, H, W]
        
        # === VISUALIZE MẪU (chỉ 5 cái đầu) ===
        if visualized_count < max_visualize:
            try:
                visualized(
                    img.cpu(),
                    gt_masks.cpu(),
                    pred_masks.cpu(),
                    dice_per_label,
                    idx=idx,
                    save_dir="visualizations"
                )
                visualized_count += 1
            except Exception as e:
                print(f"   Lỗi khi visualize mẫu {idx}: {e}")
        
        # In Dice trung bình cho mẫu này
        mean_dice = np.mean(dice_per_label)
        print(f"   Test {idx}: Mean Dice = {mean_dice:.4f} | Labels: {len(dice_per_label)}")
    
    # === IN KẾT QUẢ ===
    if dice_scores:
        print("\n=== KẾT QUẢ OVATUS ===")
        print(f"Mean Dice: {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
        print(f"Median Dice: {np.median(dice_scores):.4f}")
        print(f"Min/Max Dice: {np.min(dice_scores):.4f} / {np.max(dice_scores):.4f}")
        print(f"Số mẫu test: {len(test_set)}, số mask đánh giá: {len(dice_scores)}")
        print(f"Đã lưu {visualized_count} ảnh visualization vào thư mục 'visualizations/'")
    else:
        print("\nKhông có kết quả Dice nào được tính!")

if __name__ == "__main__":
    main()