# split_dataset.py
import numpy as np
import os
from torch.utils.data import Dataset, Subset
from typing import Tuple

def patient_level_split_60_20_20(
    dataset: Dataset,
    seed: int = 42
) -> Tuple[Subset, Subset, Subset]:
    """
    Chia theo subject (patient) đúng như paper UniverSeg:
    - 60% support pool → dùng để sample support set động
    - 20% dev → không dùng ở đây (có thể dùng để tune)
    - 20% test → báo cáo cuối cùng
    """
    patient_to_indices = {}
    print("[DEBUG] Bắt đầu duyệt dataset để gom index theo bệnh nhân...")
    for idx in range(len(dataset)):
        _, _, img_path = dataset[idx]
        patient = os.path.basename(os.path.dirname(img_path))
        patient_to_indices.setdefault(patient, []).append(idx)
        print(f"[DEBUG] idx={idx}, img_path={img_path}, patient={patient}")

    patients = list(patient_to_indices.keys())
    print(f"[DEBUG] Danh sách bệnh nhân: {patients}")
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    print(f"[DEBUG] Danh sách bệnh nhân sau khi shuffle: {patients}")

    n = len(patients)
    n_support = int(0.6 * n)
    n_dev = int(0.2 * n)
    print(f"[DEBUG] Tổng số bệnh nhân: {n}, n_support={n_support}, n_dev={n_dev}")

    support_patients = patients[:n_support]
    dev_patients = patients[n_support:n_support + n_dev]
    test_patients = patients[n_support + n_dev:]
    print(f"[DEBUG] support_patients={support_patients}")
    print(f"[DEBUG] dev_patients={dev_patients}")
    print(f"[DEBUG] test_patients={test_patients}")

    def get_idxs(p_list):
        idxs = [i for p in p_list for i in patient_to_indices[p]]
        print(f"[DEBUG] get_idxs({p_list}) -> {idxs}")
        return idxs

    support_idxs = get_idxs(support_patients)
    dev_idxs = get_idxs(dev_patients)
    test_idxs = get_idxs(test_patients)

    print(f"Total patients: {len(patients)}")
    print(f"Support pool : {len(support_patients)} patients → {len(support_idxs)} images")
    print(f"Dev set      : {len(dev_patients)} patients → {len(dev_idxs)} images")
    print(f"Test set     : {len(test_patients)} patients → {len(test_idxs)} images")

    return Subset(dataset, support_idxs), Subset(dataset, dev_idxs), Subset(dataset, test_idxs)