"""
Script chia dữ liệu OTU_2D thành Train và Validation
Theo file train.txt và val.txt có sẵn
"""

import os
import shutil
from pathlib import Path

# ============== CẤU HÌNH ==============
BASE_DIR = "/thiends/hdd2t/UniverSeg/OTU_2D"
IMAGES_DIR = os.path.join(BASE_DIR, "images")
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "annotations")

# File chứa danh sách ID
TRAIN_TXT = os.path.join(BASE_DIR, "train.txt")
VAL_TXT = os.path.join(BASE_DIR, "val.txt")

# Thư mục đầu ra
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train1", "Image")
TRAIN_LABEL_DIR = os.path.join(BASE_DIR, "train1", "Label")
VAL_IMG_DIR = os.path.join(BASE_DIR, "validation1", "Image")
VAL_LABEL_DIR = os.path.join(BASE_DIR, "validation1", "Label")

# Chế độ: 'copy' hoặc 'move'
MODE = 'copy'  # Dùng 'copy' để giữ nguyên dữ liệu gốc
# ======================================


def create_directories():
    """Tạo các thư mục cần thiết"""
    dirs = [TRAIN_IMG_DIR, TRAIN_LABEL_DIR, VAL_IMG_DIR, VAL_LABEL_DIR]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"✓ Tạo thư mục: {d}")


def clear_directories():
    """Xóa sạch các thư mục đích trước khi chia"""
    dirs = [TRAIN_IMG_DIR, TRAIN_LABEL_DIR, VAL_IMG_DIR, VAL_LABEL_DIR]
    for d in dirs:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"✗ Xóa thư mục cũ: {d}")
    create_directories()


def get_file_ids():
    """Đọc danh sách ID từ file train.txt và val.txt"""
    
    # Đọc train.txt
    with open(TRAIN_TXT, 'r') as f:
        train_ids = [line.strip() for line in f if line.strip()]
    
    # Đọc val.txt
    with open(VAL_TXT, 'r') as f:
        val_ids = [line.strip() for line in f if line.strip()]
    
    print(f"\n📊 Đọc từ file txt:")
    print(f"   - train.txt: {len(train_ids)} IDs")
    print(f"   - val.txt: {len(val_ids)} IDs")
    print(f"   - Tổng: {len(train_ids) + len(val_ids)} IDs")
    
    return train_ids, val_ids


def find_file_with_extension(directory, file_id):
    """Tìm file với ID cho trước (bất kể extension)"""
    for ext in ['.JPG', '.jpg', '.JPEG', '.jpeg', '.PNG', '.png']:
        filepath = os.path.join(directory, file_id + ext)
        if os.path.exists(filepath):
            return filepath
    return None


def split_and_copy(train_ids, val_ids):
    """Copy file vào các thư mục theo danh sách ID"""
    
    print(f"\n📂 Phân chia dữ liệu:")
    print(f"   - Train: {len(train_ids)}")
    print(f"   - Validation: {len(val_ids)}")
    
    # Hàm copy/move
    transfer_func = shutil.copy2 if MODE == 'copy' else shutil.move
    action = "Copy" if MODE == 'copy' else "Move"
    
    # Copy train files
    print(f"\n🔄 {action} train files...")
    train_success = 0
    train_missing = []
    for file_id in train_ids:
        img_src = find_file_with_extension(IMAGES_DIR, file_id)
        ann_src = find_file_with_extension(ANNOTATIONS_DIR, file_id)
        
        if img_src and ann_src:
            img_dst = os.path.join(TRAIN_IMG_DIR, os.path.basename(img_src))
            ann_dst = os.path.join(TRAIN_LABEL_DIR, os.path.basename(ann_src))
            transfer_func(img_src, img_dst)
            transfer_func(ann_src, ann_dst)
            train_success += 1
        else:
            train_missing.append(file_id)
    
    # Copy validation files
    print(f"🔄 {action} validation files...")
    val_success = 0
    val_missing = []
    for file_id in val_ids:
        img_src = find_file_with_extension(IMAGES_DIR, file_id)
        ann_src = find_file_with_extension(ANNOTATIONS_DIR, file_id)
        
        if img_src and ann_src:
            img_dst = os.path.join(VAL_IMG_DIR, os.path.basename(img_src))
            ann_dst = os.path.join(VAL_LABEL_DIR, os.path.basename(ann_src))
            transfer_func(img_src, img_dst)
            transfer_func(ann_src, ann_dst)
            val_success += 1
        else:
            val_missing.append(file_id)
    
    # Báo cáo file thiếu
    if train_missing:
        print(f"\n⚠️ Train - Không tìm thấy {len(train_missing)} files: {train_missing[:5]}...")
    if val_missing:
        print(f"⚠️ Validation - Không tìm thấy {len(val_missing)} files: {val_missing[:5]}...")
    
    return train_success, val_success


def verify_split():
    """Kiểm tra kết quả chia"""
    train_img_count = len(os.listdir(TRAIN_IMG_DIR))
    train_label_count = len(os.listdir(TRAIN_LABEL_DIR))
    val_img_count = len(os.listdir(VAL_IMG_DIR))
    val_label_count = len(os.listdir(VAL_LABEL_DIR))
    
    total = train_img_count + val_img_count
    train_pct = train_img_count / total * 100 if total > 0 else 0
    val_pct = val_img_count / total * 100 if total > 0 else 0
    
    print(f"\n✅ Kết quả cuối cùng:")
    print(f"   Train:")
    print(f"      - Images: {train_img_count} ({train_pct:.1f}%)")
    print(f"      - Labels: {train_label_count}")
    print(f"   Validation:")
    print(f"      - Images: {val_img_count} ({val_pct:.1f}%)")
    print(f"      - Labels: {val_label_count}")
    print(f"   Tổng: {total}")
    
    # Kiểm tra tính nhất quán
    if train_img_count == train_label_count and val_img_count == val_label_count:
        print(f"\n🎉 Chia dữ liệu thành công!")
    else:
        print(f"\n⚠️ Cảnh báo: Số lượng Image và Label không khớp!")


def main():
    print("=" * 50)
    print("   CHIA DỮ LIỆU OTU_2D: TRAIN & VALIDATION")
    print("   (Theo file train.txt và val.txt)")
    print("=" * 50)
    
    # Kiểm tra thư mục nguồn
    if not os.path.exists(IMAGES_DIR) or not os.path.exists(ANNOTATIONS_DIR):
        print("❌ Lỗi: Không tìm thấy thư mục images/ hoặc annotations/")
        return
    
    # Kiểm tra file txt
    if not os.path.exists(TRAIN_TXT) or not os.path.exists(VAL_TXT):
        print("❌ Lỗi: Không tìm thấy file train.txt hoặc val.txt")
        return
    
    # Xóa và tạo lại thư mục
    clear_directories()
    
    # Đọc danh sách ID từ file
    train_ids, val_ids = get_file_ids()
    
    if not train_ids and not val_ids:
        print("❌ Lỗi: Không tìm thấy ID hợp lệ!")
        return
    
    # Copy files
    train_count, val_count = split_and_copy(train_ids, val_ids)
    
    # Kiểm tra kết quả
    verify_split()
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
