# Phân Tích Cấu Trúc Dữ Liệu OVatus - Siêu Âm Buồng Trứng

## 📋 Mục Lục
1. [Tổng Quan Dataset](#tổng-quan-dataset)
2. [Cấu Trúc Thư Mục](#cấu-trúc-thư-mục)
3. [Định Dạng Dữ Liệu](#định-dạng-dữ-liệu)
4. [Các Lớp Nhãn (Labels)](#các-lớp-nhãn-labels)
5. [Thống Kê Phân Bố Dữ Liệu](#thống-kê-phân-bố-dữ-liệu)
6. [Pipeline Xử Lý Dữ Liệu](#pipeline-xử-lý-dữ-liệu)
7. [Các Vấn Đề và Giải Pháp](#các-vấn-đề-và-giải-pháp)

---

## 🎯 Tổng Quan Dataset

### Dataset OVatus là gì?
**OVatus** là dataset chuyên biệt về **phân đoạn (Segmentation) siêu âm buồng trứng**. Dataset này chứa các ảnh siêu âm của bệnh nhân có các dạng bệnh lý khác nhau ở buồng trứng.

### Thông tin cơ bản:
- **Tổng số bệnh nhân**: ~250 bệnh nhân
- **Tổng số ảnh được annotation**: **583 ảnh**
- **Định dạng ảnh**: JPG (128×128 pixels sau khi resize)
- **Định dạng nhãn**: Polygon (danh sách các điểm tọa độ)
- **Số lớp bệnh lý**: **6 lớp**

---

## 📁 Cấu Trúc Thư Mục

### Cấu trúc chính:
```
/thiends/hdd2t/UniverSeg/
├── US_map4/                          # Thư mục chứa ảnh siêu âm
│   ├── 167_Nguyễn Thị Hoa/           # Thư mục bệnh nhân 167
│   │   ├── b7547ff0-1c01-42c3.jpg
│   │   ├── c1876762-7728-4edf.jpg
│   │   └── ...
│   ├── 168_Phạm Thị Khánh Vân/       # Thư mục bệnh nhân 168
│   │   └── b15cfe36-db12-4214.jpg
│   ├── 169_2400004224_NH VAN/
│   ├── 170_Nguyễn Thị Thảo/
│   ├── ... (240+ bệnh nhân khác)
│   └── UNK_019/
├── mapping_normalized4.jsonl         # File annotation (JSONL format)
└── [các file khác]
```

### Cấu trúc tên thư mục bệnh nhân:
```
[ID_BN]_[Tên_Bệnh_Nhân]/
```
- **ID_BN**: Mã số bệnh nhân (167, 168, ...)
- **Tên_Bệnh_Nhân**: Tên đầy đủ hoặc mã học sinh (Nguyễn Thị Hoa, 2400004224, v.v.)

---

## 📊 Định Dạng Dữ Liệu

### 1. Tệp Annotation (mapping_normalized4.jsonl)

**Định dạng**: JSONL (JSON Lines - mỗi dòng là một JSON object)

**Cấu trúc của mỗi dòng**:
```json
{
  "patient_name": "167_Nguyễn Thị Hoa",
  "images": [
    {
      "image_name": "b7547ff0-1c01-42c3-bc54-5946129319cf.jpg",
      "imageWidth": 1136,
      "imageHeight": 852,
      "labels": ["nang_da_thuy"],
      "points": [
        [[857.62, 427.88], [841.58, 367.50], [813.28, 319.39], ...]
      ]
    },
    {
      "image_name": "c1876762-7728-4edf-b44a-d6b79faed1bb.jpg",
      "imageWidth": 1136,
      "imageHeight": 852,
      "labels": ["u_bi", "u_bi"],
      "points": [
        [[522.71, 350.52], [530.26, 308.07], ...],
        [[651.01, 508.07], [687.81, 570.33], ...]
      ]
    }
  ]
}
```

**Giải thích các trường**:
| Trường | Kiểu | Mô tả |
|--------|------|-------|
| `patient_name` | String | Mã bệnh nhân + tên |
| `images` | Array | Danh sách ảnh của bệnh nhân |
| `image_name` | String | UUID của ảnh gốc |
| `imageWidth` | Int | Chiều rộng ảnh gốc (pixel) |
| `imageHeight` | Int | Chiều cao ảnh gốc (pixel) |
| `labels` | Array[String] | Danh sách nhãn bệnh lý |
| `points` | Array[Array[Tuple]] | Danh sách polygon cho mỗi nhãn |

### 2. Định dạng Polygon
Mỗi polygon là **danh sách các điểm tọa độ (x, y)** trong hệ tọa độ gốc:
```python
[[x1, y1], [x2, y2], [x3, y3], ...]
```

**Ví dụ**: Một cyst được đánh dấu bằng 12 điểm:
```python
[[857.62, 427.88], [841.58, 367.50], [813.28, 319.39], 
 [782.15, 285.43], [755.73, 270.33], [729.32, 250.52], 
 [698.18, 226.94], [670.83, 203.35], [646.30, 201.47], ...]
```

### 3. Ảnh Gốc
- **Định dạng**: JPG/PNG
- **Độ phân giải ban đầu**: ~1000-1100 × 800 pixels
- **Sau xử lý**: Resize về **128×128 pixels**
- **Kêu màu**: RGB (3 channels)

---

## 🏷️ Các Lớp Nhãn (Labels)

### 6 Lớp Bệnh Lý Chính:

| ID | Tên Tiếng Việt | Tên Trong Code | Mô Tả Lâm Sàng | Ưu Tiên |
|----|---|---|---|---|
| **0** | Nang Đa Nước | `nang_da_thuy` | Cyst nhiều buồng, chứa nước | Trung bình |
| **1** | Nang Đơn Nước | `nang_don_thuy` | Cyst một buồng đơn thuần | Trung bình |
| **2** | Nang Đa Nước Đặc | `nang_da_thuy_dac` | Cyst đa buồng + có phần đặc | Hiếm |
| **3** | Nang Đơn Nước Đặc | `nang_don_thuy_dac` | Cyst đơn buồng + có phần đặc | Hiếm |
| **4** | U Lành (Benign) | `u_bi` | Khối u lành tính | Phổ biến |
| **5** | U Ác (Solid) | `u_dac` | Khối u hoặc thành phần rắn | Phổ biến |

### Đặc điểm của các lớp:

#### Nang (Cyst - ID 0, 1, 2, 3):
- **Đặc điểm**: Có chứa dịch, ranh giới rõ ràng
- **Phân loại**: Dựa trên **số buồng** (đơn/đa) và **có thành phần đặc hay không**
- **Kích thước**: Thường nhỏ hơn u
- **Đặc điểm siêu âm**: Cạnh sáng, nền tối

#### U (Tumor - ID 4, 5):
- **Đặc điểm**: Khối rắn hoặc bán rắn
- **Phân loại**: Dựa trên **tính chất** (lành/ác)
- **Kích thước**: Có thể lớn hơn nang
- **Đặc điểm siêu âm**: Mô đồng nhất hoặc không đồng nhất

---

## 📈 Thống Kê Phân Bố Dữ Liệu

### Phân Bố Toàn Bộ Dataset (583 ảnh):

```
[0] nang_da_thuy         : 151 ảnh (25.9%) ✅ COMMON
[1] nang_don_thuy        : 148 ảnh (25.4%) ✅ COMMON  
[2] nang_da_thuy_dac     :  31 ảnh ( 5.3%) ⚠️ RARE
[3] nang_don_thuy_dac    :  30 ảnh ( 5.1%) ⚠️ RARE
[4] u_bi                 : 116 ảnh (19.9%) ✅ COMMON
[5] u_dac                : 137 ảnh (23.5%) ✅ COMMON
─────────────────────────────────────────────
    TỔNG CỘNG            : 583 ảnh (100%)
```

### Phân Loại Theo Mức Độ Phổ Biến:

| Mức | Tiêu Chí | Lớp |
|-----|----------|------|
| 🟢 **COMMON** | ≥ 100 ảnh | `nang_da_thuy`, `nang_don_thuy`, `u_bi`, `u_dac` |
| 🟡 **MEDIUM** | 50-99 ảnh | (Không có lớp nào ở mức này) |
| 🟠 **RARE** | 20-49 ảnh | `nang_da_thuy_dac`, `nang_don_thuy_dac` |
| 🔴 **VERY RARE** | < 20 ảnh | (Không có) |

### Nhận Xét Quan Trọng:
- **Dữ liệu cân bằng** khá tốt, không quá mất cân bằng
- **Các lớp hiếm** (`nang_da_thuy_dac`, `nang_don_thuy_dac`) cần **Adaptive Stratified Sampling**
- **Các lớp phổ biến** có đủ mẫu để training

---

## 🔄 Pipeline Xử Lý Dữ Liệu

### Bước 1: Load Dataset từ JSONL
```python
# Đọc file annotation
with open("mapping_normalized4.jsonl", "r") as f:
    for line in f:
        item = json.loads(line)  # Parse JSON
        patient_name = item["patient_name"]
        images = item["images"]
```

### Bước 2: Load Ảnh và Tạo Mask

**Quá trình**:
1. Tìm file ảnh trong thư mục bệnh nhân: `{DATA_ROOT}/{patient_name}/{image_name}`
2. Load ảnh JPG → Resize về (128, 128) → Normalize (chia cho 255)
3. Chuyển từ (H,W,C) → (C,H,W) format PyTorch

**Code**:
```python
def process_image(image_path, resize_to=(128, 128)):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(resize_to, Image.BILINEAR)
    img = np.array(img).astype(np.float32) / 255.0
    return np.transpose(img, (2, 0, 1))  # [3, 128, 128]
```

### Bước 3: Chuyển Polygon thành Mask

**Quá trình**:
1. Lấy danh sách điểm polygon từ JSON
2. Tạo ảnh trắng (H,W)
3. Vẽ polygon bên trong bằng PIL.ImageDraw
4. Resize mask về (128, 128) bằng NEAREST interpolation
5. Tạo một mask cho mỗi lớp bệnh lý

**Code**:
```python
def polygon_to_mask(points, image_size, resize_to=(128, 128)):
    mask = Image.new('L', image_size, 0)
    if points is not None and len(points) >= 3:
        ImageDraw.Draw(mask).polygon(points, outline=1, fill=1)
    mask = mask.resize(resize_to, Image.NEAREST)
    return np.array(mask).astype(np.float32)
```

### Bước 4: Xếp Chồng Mask Cho Multi-Label

**Đặc biệt**: Một ảnh có thể chứa nhiều instance của **cùng một lớp**

**Ví dụ**: Ảnh có 2 cyst nước riêng biệt
```python
labels = ["nang_da_thuy", "nang_da_thuy"]  # 2 lần cùng một lớp
points = [[[...]], [[...]]]                 # 2 polygon khác nhau

# Cách xử lý:
for label, poly_points in zip(labels, points):
    class_id = LABEL2ID[label]
    mask = polygon_to_mask(poly_points, ...)
    masks[class_id] = np.maximum(masks[class_id], mask)  # OR operation
```

### Bước 5: Tạo Output Tensor

**Output cho mỗi ảnh**:
```python
(img, masks, img_path)
# img:       torch.Tensor [3, 128, 128] - ảnh RGB
# masks:     torch.Tensor [6, 128, 128] - mask cho 6 lớp
# img_path:  str - đường dẫn file
```

### Sơ Đồ Pipeline Hoàn Chỉnh:

```
mapping_normalized4.jsonl
        ↓
   [JSON Parser]
        ↓
   (patient_name, images metadata)
        ↓
   [Load Image] → {DATA_ROOT}/{patient_name}/{image_name}
        ↓
   (ảnh gốc, W, H)
        ↓
   [Resize + Normalize] → (128, 128), RGB [0-1]
        ↓
   [Load Polygons] → danh sách điểm tọa độ
        ↓
   [Polygon → Mask] → binary mask (H, W)
        ↓
   [Resize Mask] → (128, 128) NEAREST
        ↓
   [Stack Masks] → [6, 128, 128]
        ↓
   (Image, Masks) → PyTorch Dataset
```

---

## ⚠️ Các Vấn Đề và Giải Pháp

### 1️⃣ Multi-Instance Per Class

**Vấn đề**: 
- Một ảnh có thể chứa 2 hoặc nhiều khối của **cùng một loại bệnh**
- Ví dụ: Cùng một ảnh có 2 khối `u_bi` riêng biệt

**Giải Pháp**:
- Sử dụng **OR operation** (Maximum) để merge masks:
  ```python
  masks[class_id] = np.maximum(masks[class_id], new_mask)
  ```
- Điều này đảm bảo tất cả vùng bệnh lý được ghi nhận

### 2️⃣ Multi-Label (Overlapping) Instances

**Vấn Đề**:
- Một ảnh có thể chứa cùng lúc nhiều loại bệnh (VD: vừa nang vừa u)
- Hai khối bệnh lý có thể chồng lấp (overlapping)

**Kiểm tra Overlap**:
```python
total_mask = masks.sum(axis=0)  # Tổng tất cả lớp
overlap_pixels = (total_mask > 1).sum()  # Pixel thuộc 2+ lớp

if overlap_pixels > 0:
    print(f"⚠️ Có {overlap_pixels} pixels chồng lấp!")
```

**Hậu quả**:
- Không thể dùng **Softmax** (mutually exclusive)
- Phải dùng **Sigmoid** (multi-label independent)

### 3️⃣ Dữ Liệu Mất Cân Bằng (Imbalanced Data)

**Vấn đề**:
- 4 lớp phổ biến (100-150 ảnh)
- 2 lớp hiếm (30 ảnh)
- **Class imbalance ratio**: ~5:1

**Giải Pháp**:
- **Adaptive Stratified Sampling**: Lấy mẫu support tùy theo rarity
  - Lớp hiếm: 90% of available
  - Lớp phổ biến: 50% of available
- **Weighted Loss**: Trọng số cao cho lớp hiếm
- **Over-sampling**: Duplicate rare classes

### 4️⃣ Kích Thước Ảnh Không Đồng Nhất

**Vấn đề**:
- Ảnh gốc có độ phân giải khác nhau (~800-1200px)
- Cần Resize để training

**Cách xử lý**:
```python
RESIZE_TO = (128, 128)  # Resize tất cả về kích thước này
img = img.resize(RESIZE_TO, Image.BILINEAR)   # Ảnh
mask = mask.resize(RESIZE_TO, Image.NEAREST)  # Mask
```

**Lưu ý**: 
- ✅ BILINEAR cho ảnh (giữ chi tiết)
- ✅ NEAREST cho mask (tránh artifact)

### 5️⃣ Mặt Nạ Không Hợp Lệ (Invalid Masks)

**Vấn đề**:
- Polygon không hợp lệ (< 3 điểm)
- Polygon nằm ngoài ranh giới ảnh
- Ảnh không tồn tại

**Giải Pháp**:
```python
# Kiểm tra hợp lệ
if not os.path.exists(img_path):
    continue  # Bỏ qua ảnh không tồn tại

if len(poly_points) < 3:
    continue  # Bỏ qua polygon không hợp lệ

if not has_valid_mask:
    continue  # Bỏ qua ảnh không có mask hợp lệ

# Chỉ thêm ảnh nếu có ít nhất 1 mask hợp lệ
self.samples.append((img, masks, img_path))
```

---

## 🔧 Hướng Dẫn Sử Dụng Code

### 1️⃣ Load Dataset

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

### 2️⃣ Lấy Một Mẫu

```python
img, masks, path = dataset[0]

print(f"Image shape:  {img.shape}")    # [3, 128, 128]
print(f"Masks shape:  {masks.shape}")  # [6, 128, 128]
print(f"Image path:   {path}")
```

### 3️⃣ Phân Tích Mẫu

```python
import numpy as np

# Tính bao phủ cho mỗi lớp
for label_id in range(6):
    mask = masks[label_id]
    coverage = 100 * mask.sum() / mask.numel()
    print(f"Label {label_id}: {coverage:.2f}%")
```

### 4️⃣ Split Dataset

```python
from split_support_test import patient_level_split_60_20_20

support_pool, dev_set, test_set = patient_level_split_60_20_20(
    dataset, 
    seed=42
)

print(f"Support: {len(support_pool)} | Dev: {len(dev_set)} | Test: {len(test_set)}")
```

---

## 📚 Tài Liệu Tham Khảo

### Files Liên Quan:
- `mapping_normalized4.jsonl` - Annotation file chính
- `scripts1/make_OVatusData.py` - Dataset loader (phiên bản mới)
- `scripts1/infer.py` - Inference script với Adaptive Stratified Sampling
- `universeg_analization_Ovatus.ipynb` - Jupyter notebook phân tích

### Key Constants:
```python
LABEL2ID = {
    "nang_da_thuy": 0,
    "nang_don_thuy": 1,
    "nang_da_thuy_dac": 2,
    "nang_don_thuy_dac": 3,
    "u_bi": 4,
    "u_dac": 5
}

RESIZE_TO = (128, 128)
NUM_CLASSES = 6
```

---

## 🎓 Kết Luận

### Điểm Mạnh:
✅ Dataset cân bằng tương đối (không quá mất cân bằng)
✅ Annotation chính xác (polygon đầy đủ)
✅ Đủ mẫu cho training (583 ảnh)
✅ Đa dạng bệnh lý (6 loại)

### Điểm Yếu:
⚠️ Hai lớp hiếm cần xử lý đặc biệt
⚠️ Có overlapping instances (cần Sigmoid)
⚠️ Số lượng bệnh nhân hạn chế (250 bệnh nhân)

### Khuyến Nghị:
1. **Dùng Few-Shot Learning** (UniverSeg) để giải quyết class imbalance
2. **Dùng Adaptive Stratified Sampling** để lấy mẫu tối ưu
3. **Dùng Sigmoid activation** (không Softmax) vì multi-label
4. **Cross-validation patient-level** để tránh data leakage

---

**Document Version**: 1.0
**Last Updated**: December 22, 2025
**Dataset**: OVatus - Ovarian Ultrasound Segmentation Dataset
