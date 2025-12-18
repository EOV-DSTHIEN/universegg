import numpy as np
import os
from make_OVatusData import OVatusDataset

def split_dataset(dataset, support_frac=0.4, seed=42):
    # Group image indices by patient
    patient_to_indices = {}
    for idx in range(len(dataset)):
        _, _, img_path = dataset[idx]
        # Extract patient name from path
        patient = os.path.basename(os.path.dirname(img_path))
        if patient not in patient_to_indices:
            patient_to_indices[patient] = []
        patient_to_indices[patient].append(idx)

    patients = list(patient_to_indices.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    split_idx = int(support_frac * len(patients))
    support_patients = patients[:split_idx]
    test_patients = patients[split_idx:]

    support_idxs = []
    test_idxs = []
    for p in support_patients:
        support_idxs.extend(patient_to_indices[p])
    for p in test_patients:
        test_idxs.extend(patient_to_indices[p])
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