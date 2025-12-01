import numpy as np
from make_OVatusData import OVatusDataset

def split_dataset(dataset, support_frac=0.4, seed=42):
    N = len(dataset)
    idxs = np.arange(N)
    rng = np.random.default_rng(seed)
    rng.shuffle(idxs)
    split_idx = int(support_frac * N)
    support_idxs = idxs[:split_idx]
    test_idxs = idxs[split_idx:]
    return support_idxs, test_idxs

if __name__ == "__main__":
    dataset = OVatusDataset()
    support_idxs, test_idxs = split_dataset(dataset, support_frac=0.4)
    print(f"Support set: {len(support_idxs)} samples")
    print(f"Test set: {len(test_idxs)} samples")
    # Ví dụ lấy dữ liệu
    support_set = [dataset[i] for i in support_idxs]
    test_set = [dataset[i] for i in test_idxs]
    # Có thể lưu hoặc dùng trực tiếp cho mô hình