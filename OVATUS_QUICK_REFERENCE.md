# OVatus Dataset - Tóm Tắt Nhanh

## 🎯 Dataset Là Gì?

**OVatus** = Dataset Siêu Âm Buồng Trứng với 583 ảnh được annotate thủ công cho 6 loại bệnh lý.

---

## 📊 Con Số Chính

| Thuộc Tính | Giá Trị |
|---|---|
| **Tổng ảnh được annotation** | 583 |
| **Số bệnh nhân** | ~250 |
| **Số lớp bệnh** | 6 |
| **Kích thước ảnh** | 128×128 (sau resize) |
| **Định dạng nhãn** | Polygon (tọa độ điểm) |
| **Format lưu trữ** | JSONL + JPG |

---

## 6️⃣ Các Lớp Bệnh Lý

### Cấu Trúc:
```
Cyst (Nang)     → 4 lớp (dựa trên số buồng + có đặc hay không)
├─ nang_da_thuy (0)      [151 ảnh] ✅
├─ nang_don_thuy (1)     [148 ảnh] ✅
├─ nang_da_thuy_dac (2)  [31 ảnh]  ⚠️
└─ nang_don_thuy_dac (3) [30 ảnh]  ⚠️

Tumor (U)       → 2 lớp (dựa trên tính chất)
├─ u_bi (4)     [116 ảnh] ✅
└─ u_dac (5)    [137 ảnh] ✅
```

### Phân Bố:
```
Phổ Biến (≥100):   4 lớp → 552 ảnh
Hiếm (20-99):      2 lớp → 61 ảnh
```

---

## 📁 Cấu Trúc Dữ Liệu

### JSONL Format (mapping_normalized4.jsonl):
```json
{
  "patient_name": "167_Nguyễn Thị Hoa",
  "images": [
    {
      "image_name": "b754...cf.jpg",
      "imageWidth": 1136,
      "imageHeight": 852,
      "labels": ["nang_da_thuy"],
      "points": [[[x1,y1], [x2,y2], ...]]
    }
  ]
}
```

### File Structure:
```
US_map4/
├─ 167_Nguyễn Thị Hoa/
│  ├─ image1.jpg
│  ├─ image2.jpg
│  └─ ...
├─ 168_Phạm Thị Khánh Vân/
└─ ... (250 bệnh nhân)
```

---

## 🔄 Pipeline Xử Lý

```
JSONL → Load Image → Resize 128×128 → Normalize [0,1]
         ↓
         Polygon → Draw mask → Resize → Stack [6,128,128]
         ↓
         (Image_Tensor, Masks_Tensor, Path)
```

---

## ⚠️ Lưu Ý Quan Trọng

### 1. Multi-Instance
- Một ảnh có thể chứa **2+ khối cùng 1 loại bệnh**
- Xử lý: `mask = np.maximum(mask1, mask2)` (OR operation)

### 2. Multi-Label
- Một ảnh có thể chứa **nhiều loại bệnh khác nhau**
- Có khả năng **overlap** (pixels thuộc 2+ lớp)
- Cần dùng **Sigmoid** chứ không phải **Softmax**

### 3. Class Imbalance
- Các lớp hiếm cần **Adaptive Stratified Sampling**
- Support pool: 90% rare classes, 50% common classes

### 4. Invalid Data
- ~77% ảnh gốc không có annotation hợp lệ
- Lý do: Polygon không đủ 3 điểm, file missing, v.v.
- Kết quả: 583 ảnh hợp lệ từ ~2500 ảnh gốc

---

## 💡 Sử Dụng Nhanh

### Load Dataset:
```python
from make_OVatusData import OvatusDataset

dataset = OvatusDataset(
    annot_path="/path/to/mapping_normalized4.jsonl",
    data_root="/path/to/US_map4",
    resize_to=(128, 128),
    check_overlap=True
)

print(f"Loaded {len(dataset)} images")
```

### Lấy 1 Mẫu:
```python
img, masks, path = dataset[0]
# img:   [3, 128, 128]
# masks: [6, 128, 128]
# path:  string
```

### Split Dataset:
```python
from split_support_test import patient_level_split_60_20_20

support, dev, test = patient_level_split_60_20_20(dataset, seed=42)
# support: 336 ảnh (60%)
# dev:     113 ảnh (20%)
# test:    134 ảnh (20%)
```

---

## 📚 Files Tham Khảo

| File | Mô Tả |
|---|---|
| `OVATUS_DATA_STRUCTURE_ANALYSIS.md` | **Tài liệu chi tiết** (477 dòng) |
| `OVATUS_VISUAL_GUIDE.md` | **Hướng dẫn hình ảnh** (480 dòng) |
| `mapping_normalized4.jsonl` | **Annotation file** |
| `scripts1/make_OVatusData.py` | **Dataset loader code** |
| `universeg_analization_Ovatus.ipynb` | **Phân tích notebook** |

---

## 🎓 Kết Luận

✅ **Ưu điểm**:
- Cân bằng dữ liệu tốt
- Annotation chính xác
- Đa dạng bệnh lý

⚠️ **Thách Thức**:
- Các lớp hiếm cần xử lý đặc biệt
- Multi-label/overlap handling
- Few-shot learning cần thiết

---

**Quick Reference**
- 📊 **583 ảnh**, 6 lớp, ~250 bệnh nhân
- 🎯 **Phân bố**: 4 phổ biến + 2 hiếm
- 📁 **Format**: JSONL + JPG (128×128)
- 🔧 **Process**: Polygon → Mask → Tensor
- 🚀 **Framework**: Few-Shot Learning (UniverSeg)

---

*Last Updated: December 22, 2025*
*For detailed info, see: OVATUS_DATA_STRUCTURE_ANALYSIS.md*
