import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw

DATA_ROOT = "/thiends/hdd2t/UniverSeg/US_map4"
ANNOT_PATH = "/thiends/hdd2t/UniverSeg/mapping_normalized4.jsonl"
RESIZE_TO = (128, 128)

def polygon_to_mask(points, image_size, resize_to=RESIZE_TO):
    mask = Image.new('L', image_size, 0)
    # Chỉ vẽ nếu có ít nhất 3 điểm
    if points is not None and len(points) >= 3:
        ImageDraw.Draw(mask).polygon(points, outline=1, fill=1)
    mask = mask.resize(resize_to, Image.NEAREST)
    mask = np.array(mask).astype(np.float32)
    return mask

def process_image(image_path, resize_to=RESIZE_TO):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(resize_to, Image.BILINEAR)
    img = np.array(img).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return img

class OVatusDataset(Dataset):
    def __init__(self, annot_path=ANNOT_PATH, data_root=DATA_ROOT, resize_to=RESIZE_TO):
        self.samples = []
        with open(annot_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                patient = item["patient_name"]
                for img_info in item["images"]:
                    img_name = img_info["image_name"]
                    img_path = os.path.join(data_root, patient, img_name)
                    image_size = (img_info["imageWidth"], img_info["imageHeight"])
                    img = process_image(img_path, resize_to)
                    masks = []
                    for points_list in img_info["points"]:
                        points = [tuple(map(float, pt)) for pt in points_list]
                        mask = polygon_to_mask(points, image_size, resize_to)
                        masks.append(mask)
                    masks = np.stack(masks)  # [num_labels, H, W]
                    self.samples.append((img, masks, img_path))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img, masks, img_path = self.samples[idx]
        img = torch.from_numpy(img).float()
        masks = torch.from_numpy(masks).float()
        return img, masks, img_path