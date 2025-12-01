import os
import json
import numpy as np
from PIL import Image, ImageDraw

# Đường dẫn
DATA_ROOT = "/thiends/hdd2t/UniverSeg/US_map4"
ANNOT_PATH = "/thiends/hdd2t/UniverSeg/mapping_normalized4.jsonl"
RESIZE_TO = (128, 128)

def polygon_to_mask(points, image_size, resize_to=RESIZE_TO):
    mask = Image.new('L', image_size, 0)
    ImageDraw.Draw(mask).polygon(points, outline=1, fill=1)
    mask = mask.resize(resize_to, Image.NEAREST)
    mask = np.array(mask).astype(np.float32)
    return mask

def process_image(image_path, resize_to=RESIZE_TO):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(resize_to, Image.BILINEAR)
    img = np.array(img).astype(np.float32) / 255.0
    # Nếu muốn shape [C, H, W]:
    img = np.transpose(img, (2, 0, 1))
    return img

def main():
    with open(ANNOT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            patient = item["patient_name"]
            for img_info in item["images"]:
                img_name = img_info["image_name"]
                img_path = os.path.join(DATA_ROOT, patient, img_name)
                image_size = (img_info["imageWidth"], img_info["imageHeight"])
                # Chuẩn hóa ảnh
                img = process_image(img_path)
                # Chuẩn hóa mask cho từng label
                masks = []
                for label, points_list in zip(img_info["labels"], img_info["points"]):
                    # Chuyển điểm về tuple int
                    points = [tuple(map(float, pt)) for pt in points_list]
                    mask = polygon_to_mask(points, image_size)
                    masks.append(mask)
                masks = np.stack(masks)  # [num_labels, H, W]
                # Lưu hoặc trả về img, masks để dùng cho UniverSeg
                # Ví dụ: print shapes
                print(f"{img_path}: img shape {img.shape}, masks shape {masks.shape}")

if __name__ == "__main__":
    main()