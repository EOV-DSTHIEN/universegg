# Báo Cáo Phân Tích Cơ Sở Dữ Liệu OTU_2D

## 1. Tổng Quan

**OTU_2D** (Ovarian Tumor Ultrasound 2D) là bộ dữ liệu hình ảnh siêu âm khối u buồng trứng 2D, được sử dụng cho bài toán **phân đoạn** và **phân loại** ảnh y tế.

| Thông tin | Giá trị |
|-----------|---------|
| **Tổng số ảnh gốc** | 1,469 |
| **Số lớp phân loại** | 8 (0-7) |
| **Định dạng ảnh** | JPG |
| **Định dạng nhãn phân đoạn** | PNG |
| **Tổng dung lượng** | ~293 MB |

---

## 2. Cấu Trúc Thư Mục

```
OTU_2D/
├── images/               # Ảnh gốc (1,469 files JPG) - 139 MB
├── annotations/          # Nhãn phân đoạn gốc (1,469 files PNG) - 7.3 MB
├── train.txt             # Danh sách ID train (1,000 mẫu)
├── val.txt               # Danh sách ID validation (469 mẫu)
├── train_cls.txt         # ID + nhãn phân loại (train)
├── val_cls.txt           # ID + nhãn phân loại (validation)
├── train1/               # ~100 MB
│   ├── Image/            # Ảnh huấn luyện
│   └── Label/            # Nhãn huấn luyện
└── validation1/          # ~47 MB
    ├── Image/            # Ảnh validation
    └── Label/            # Nhãn validation
```

---

## 3. Phân Chia Dữ Liệu (Data Split)

| Tập dữ liệu | Số lượng | Tỷ lệ (%) |
|-------------|----------|-----------|
| **Train** | 1,000 | 68.1% |
| **Validation** | 469 | 31.9% |
| **Tổng** | **1,469** | **100%** |

### Biểu đồ phân chia:
```
Train:      ████████████████████████████████████ 68.1% (1,000)
Validation: █████████████████ 31.9% (469)
```

---

## 4. Phân Loại (Classification Labels)

Dataset có **8 lớp** phân loại khối u (Class 0-7):

### 4.1. Phân bố lớp trong tập Train (1,000 mẫu)

| Lớp | Số lượng | Tỷ lệ (%) | Biểu đồ |
|-----|----------|-----------|---------|
| 0 | 226 | 22.6% | ████████████████████████ |
| 1 | 153 | 15.3% | ████████████████ |
| 2 | 228 | 22.8% | ████████████████████████ |
| 3 | 57 | 5.7% | ██████ |
| 4 | 47 | 4.7% | █████ |
| 5 | 180 | 18.0% | ███████████████████ |
| 6 | 71 | 7.1% | ████████ |
| 7 | 38 | 3.8% | ████ |

### 4.2. Phân bố lớp trong tập Validation (469 mẫu)

| Lớp | Số lượng | Tỷ lệ (%) | Biểu đồ |
|-----|----------|-----------|---------|
| 0 | 110 | 23.5% | ████████████████████████ |
| 1 | 66 | 14.1% | ███████████████ |
| 2 | 108 | 23.0% | ████████████████████████ |
| 3 | 31 | 6.6% | ███████ |
| 4 | 19 | 4.1% | ████ |
| 5 | 87 | 18.6% | ███████████████████ |
| 6 | 33 | 7.0% | ████████ |
| 7 | 15 | 3.2% | ███ |

### 4.3. Tổng hợp phân bố lớp (Toàn bộ dataset)

| Lớp | Train | Val | Tổng | Tỷ lệ |
|-----|-------|-----|------|-------|
| 0 | 226 | 110 | **336** | 22.9% |
| 1 | 153 | 66 | **219** | 14.9% |
| 2 | 228 | 108 | **336** | 22.9% |
| 3 | 57 | 31 | **88** | 6.0% |
| 4 | 47 | 19 | **66** | 4.5% |
| 5 | 180 | 87 | **267** | 18.2% |
| 6 | 71 | 33 | **104** | 7.1% |
| 7 | 38 | 15 | **53** | 3.6% |

### 4.4. Nhận xét về phân bố lớp
- ⚠️ **Mất cân bằng lớp (Imbalanced)**: Lớp 0, 2, 5 chiếm đa số (~64%)
- ⚠️ **Lớp thiểu số**: Lớp 4, 7 chỉ chiếm ~8%
- ✅ **Tỷ lệ train/val** tương đồng giữa các lớp

---

## 5. Dung Lượng Lưu Trữ

| Thư mục | Dung lượng |
|---------|------------|
| images/ (gốc) | 139 MB |
| annotations/ (gốc) | 7.3 MB |
| train1/ | ~100 MB |
| validation1/ | ~47 MB |
| **Tổng OTU_2D/** | **~293 MB** |

---

## 6. Đặc Điểm Dữ Liệu

### 6.1. Định dạng file
- **Ảnh (Image)**: `.JPG` - Ảnh siêu âm gốc
- **Nhãn phân đoạn (Annotation)**: `.PNG` - Mask phân đoạn
- **Nhãn phân loại**: File `.txt` với format: `<filename> <class>`

### 6.2. Quy ước đặt tên
- Mỗi cặp ảnh-nhãn có cùng tên file (chỉ khác phần mở rộng)
- Ví dụ: `1000.JPG` ↔ `1000.PNG`

### 6.3. Tính nhất quán
- ✅ Số lượng ảnh = Số lượng annotation (1,469)
- ✅ Train + Validation = 1,469 (100%)
- ✅ Tỷ lệ phân bố lớp được duy trì giữa train và val

---

## 7. Ứng Dụng

Bộ dữ liệu này phù hợp cho các bài toán:

1. **Phân đoạn khối u buồng trứng** (Ovarian Tumor Segmentation)
2. **Phân loại khối u** (8 classes classification)
3. **Few-shot Medical Image Segmentation** (với UniverSeg)
4. **Multi-task Learning** (Segmentation + Classification)
5. **Transfer Learning** cho ảnh siêu âm y tế

---

## 8. Ghi Chú Kỹ Thuật

### Để sử dụng với UniverSeg:
```python
# Đường dẫn dữ liệu
train_images = "OTU_2D/train1/Image/"
train_labels = "OTU_2D/train1/Label/"
val_images = "OTU_2D/validation1/Image/"
val_labels = "OTU_2D/validation1/Label/"

# Đọc danh sách file
train_ids = open("OTU_2D/train.txt").read().splitlines()
val_ids = open("OTU_2D/val.txt").read().splitlines()

# Đọc nhãn phân loại
def load_cls_labels(filepath):
    labels = {}
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            filename = parts[0].replace('.JPG', '')
            cls = int(parts[1])
            labels[filename] = cls
    return labels

train_cls = load_cls_labels("OTU_2D/train_cls.txt")
val_cls = load_cls_labels("OTU_2D/val_cls.txt")
```

### Tiền xử lý khuyến nghị:
- Resize về kích thước chuẩn (ví dụ: 128x128 hoặc 256x256)
- Chuẩn hóa giá trị pixel về [0, 1]
- Nhãn phân đoạn: Binary mask (0 = background, 255/1 = foreground)
- **Xử lý mất cân bằng lớp**: Oversampling, class weights, focal loss

---

## 9. Thống Kê Tóm Tắt

```
╔═══════════════════════════════════════════════════════════════╗
║              CƠ SỞ DỮ LIỆU OTU_2D                             ║
╠═══════════════════════════════════════════════════════════════╣
║  📊 Tổng số mẫu:         1,469                                ║
║  🏋️ Tập huấn luyện:      1,000 (68.1%)                        ║
║  ✅ Tập validation:       469 (31.9%)                         ║
║  🏷️ Số lớp phân loại:    8 (Class 0-7)                        ║
║  💾 Tổng dung lượng:     ~293 MB                              ║
║  📁 Định dạng ảnh:       JPG                                  ║
║  🎭 Định dạng nhãn:      PNG (segmentation) + TXT (class)     ║
║  ⚠️ Lưu ý:               Imbalanced classes (0,2,5 dominant)  ║
╚═══════════════════════════════════════════════════════════════╝
```

---

**Ngày cập nhật báo cáo:** 05/01/2026
