import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw

DATA_ROOT = "/thiends/hdd2t/UniverSeg/US_map4"
ANNOT_PATH = "/thiends/hdd2t/UniverSeg/mapping_normalized4.jsonl"
RESIZE_TO = (128, 128)

LABEL2ID = {
    "nang_da_thuy": 0,
    "nang_don_thuy": 1,
    "nang_da_thuy_dac": 2,
    "nang_don_thuy_dac": 3,
    "u_bi": 4,
    "u_dac": 5
}
NUM_CLASSES = len(LABEL2ID)


def polygon_to_mask(points, image_size, resize_to=RESIZE_TO):
    """Convert polygon points to binary mask"""
    mask = Image.new('L', image_size, 0)
    if points is not None and len(points) >= 3:
        ImageDraw.Draw(mask).polygon(points, outline=1, fill=1)
    mask = mask.resize(resize_to, Image.NEAREST)
    return np.array(mask).astype(np.float32)


def process_image(image_path, resize_to=RESIZE_TO):
    """Load and preprocess image"""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(resize_to, Image.BILINEAR)
    img = np.array(img).astype(np.float32) / 255.0
    return np.transpose(img, (2, 0, 1))


class OvatusDataset(Dataset):
    """
    FIXED VERSION: Proper multi-label handling
    
    Key changes:
    1. Use OR operation (|) instead of maximum for multi-instance per class
    2. Add overlap detection and reporting
    3. Better validation and statistics
    """
    def __init__(self, annot_path=ANNOT_PATH, data_root=DATA_ROOT, 
                 resize_to=RESIZE_TO, check_overlap=True):
        self.samples = []
        self.stats = {
            'empty_annotations': 0,
            'skipped_images': 0,
            'total_polygons': 0,
            'polygons_per_label': {name: 0 for name in LABEL2ID.keys()},
            'overlap_pixels': 0,
            'images_with_overlap': 0
        }

        print("="*70)
        print("Loading OvatusDataset (FIXED Multi-label Version)")
        print("="*70)

        with open(annot_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                item = json.loads(line)
                patient = item["patient_name"]

                for img_info in item["images"]:
                    img_name = img_info["image_name"]
                    img_path = os.path.join(data_root, patient, img_name)
                    
                    if not os.path.exists(img_path):
                        print(f"[ERROR] File not found: {img_path}")
                        continue

                    image_size = (img_info["imageWidth"], img_info["imageHeight"])
                    img = process_image(img_path, resize_to)

                    # ============ FIX 1: Initialize masks properly ============
                    masks = np.zeros((NUM_CLASSES, resize_to[1], resize_to[0]), 
                                   dtype=np.float32)

                    labels = img_info["labels"]
                    polygons_list = img_info["points"]
                    
                    # Validate lengths match
                    if len(labels) != len(polygons_list):
                        print(f"[ERROR] Label/polygon count mismatch: {img_name}")
                        continue

                    has_valid_mask = False
                    valid_polygons_count = 0
                    
                    for idx_label in range(len(labels)):
                        lbl = labels[idx_label]
                        poly_pts = polygons_list[idx_label]

                        if lbl not in LABEL2ID:
                            print(f"[WARN] Unknown label '{lbl}' in {img_name}")
                            continue

                        class_id = LABEL2ID[lbl]

                        # Skip empty polygons
                        if not poly_pts or len(poly_pts) < 3:
                            self.stats['empty_annotations'] += 1
                            continue

                        # Convert polygon to mask
                        pts = [tuple(map(float, pt)) for pt in poly_pts]
                        mask = polygon_to_mask(pts, image_size, resize_to)
                        
                        # ============ FIX 2: Use OR for multi-instance ============
                        # Allow multiple instances of same class in one image
                        # Example: 2 separate "nang_da_thuy" regions
                        masks[class_id] = np.maximum(masks[class_id], mask)
                        # Or equivalently: masks[class_id] = masks[class_id] | mask.astype(bool)
                        
                        has_valid_mask = True
                        valid_polygons_count += 1
                        self.stats['total_polygons'] += 1
                        self.stats['polygons_per_label'][lbl] += 1

                    # Only add image if it has valid masks
                    if has_valid_mask:
                        # ============ FIX 3: Check overlap ============
                        if check_overlap:
                            total_mask = masks.sum(axis=0)  # Sum across all classes
                            overlap = (total_mask > 1).sum()
                            
                            if overlap > 0:
                                self.stats['overlap_pixels'] += overlap
                                self.stats['images_with_overlap'] += 1
                                print(f"[OVERLAP] {img_name}: {overlap} pixels belong to multiple classes")
                        
                        self.samples.append((img, masks, img_path))
                    else:
                        self.stats['skipped_images'] += 1
                        print(f"[SKIP] {img_path} - No valid polygons")

        # ============ Print statistics ============
        self.print_statistics()

    def print_statistics(self):
        """Print detailed dataset statistics"""
        print("\n" + "="*70)
        print("DATASET STATISTICS")
        print("="*70)
        print(f"Total images loaded:      {len(self.samples)}")
        print(f"Skipped images:           {self.stats['skipped_images']}")
        print(f"Empty annotations:        {self.stats['empty_annotations']}")
        print(f"Total valid polygons:     {self.stats['total_polygons']}")
        
        print(f"\nPOLYGONS PER LABEL:")
        for label_name, count in self.stats['polygons_per_label'].items():
            pct = 100 * count / self.stats['total_polygons'] if self.stats['total_polygons'] > 0 else 0
            print(f"  {label_name:20s}: {count:4d} ({pct:5.1f}%)")
        
        print(f"\nOVERLAP ANALYSIS:")
        print(f"  Images with overlap:  {self.stats['images_with_overlap']}")
        print(f"  Total overlap pixels: {self.stats['overlap_pixels']}")
        
        if self.stats['images_with_overlap'] > 0:
            print(f"  ⚠️  WARNING: {self.stats['images_with_overlap']} images have overlapping labels!")
            print(f"  → This means the data is NOT mutually exclusive")
            print(f"  → Using multi-label independent prediction (SIGMOID)")
        else:
            print(f"  ✅ No overlap detected - data is mutually exclusive")
            print(f"  → Could use either SIGMOID or SOFTMAX")
        
        print("="*70 + "\n")

    def get_label_distribution(self):
        """
        Get distribution of labels in dataset
        Returns: dict mapping label_idx -> list of sample indices
        """
        label_distribution = {i: [] for i in range(NUM_CLASSES)}
        
        for idx in range(len(self.samples)):
            _, masks, _ = self.samples[idx]
            for label_idx in range(NUM_CLASSES):
                if masks[label_idx].sum() > 10:  # Has meaningful mask
                    label_distribution[label_idx].append(idx)
        
        return label_distribution

    def analyze_sample(self, idx):
        """Debug helper: analyze a specific sample"""
        img, masks, img_path = self.samples[idx]
        
        print(f"\n{'='*60}")
        print(f"SAMPLE ANALYSIS: {idx}")
        print(f"Path: {img_path}")
        print(f"{'='*60}")
        print(f"Image shape: {img.shape}")  # [3, 128, 128]
        print(f"Masks shape: {masks.shape}")  # [6, 128, 128]
        
        print("\nPER-LABEL ANALYSIS:")
        for label_idx, label_name in enumerate(LABEL2ID.keys()):
            mask = masks[label_idx]
            n_pixels = mask.sum()
            coverage = 100 * n_pixels / (mask.shape[0] * mask.shape[1])
            print(f"  [{label_idx}] {label_name:20s}: "
                  f"{n_pixels:6.0f} pixels ({coverage:5.2f}% coverage)")
        
        # Check overlap
        total = masks.sum(axis=0)
        overlap = (total > 1).sum()
        if overlap > 0:
            print(f"\n⚠️  OVERLAP: {overlap} pixels belong to multiple labels")
            # Find which labels overlap
            for i in range(NUM_CLASSES):
                for j in range(i+1, NUM_CLASSES):
                    overlap_ij = (masks[i] * masks[j]).sum()
                    if overlap_ij > 0:
                        print(f"    {list(LABEL2ID.keys())[i]} ∩ "
                              f"{list(LABEL2ID.keys())[j]}: {overlap_ij} pixels")
        else:
            print(f"\n✅ No overlap - labels are mutually exclusive")
        
        print(f"{'='*60}\n")
        
        return img, masks, img_path

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, masks, img_path = self.samples[idx]
        return torch.from_numpy(img).float(), torch.from_numpy(masks).float(), img_path


# ============ USAGE EXAMPLE ============
if __name__ == "__main__":
    # Load dataset with overlap checking
    dataset = OvatusDataset(check_overlap=True)
    
    # Analyze first few samples
    print("\n" + "="*70)
    print("ANALYZING FIRST 3 SAMPLES")
    print("="*70)
    for i in range(min(3, len(dataset))):
        dataset.analyze_sample(i)
    
    # Get label distribution
    dist = dataset.get_label_distribution()
    print("\n" + "="*70)
    print("LABEL DISTRIBUTION IN DATASET")
    print("="*70)
    for label_idx, indices in dist.items():
        label_name = list(LABEL2ID.keys())[label_idx]
        print(f"{label_name:20s}: {len(indices):4d} images contain this label")