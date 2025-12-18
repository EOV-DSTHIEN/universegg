# normalized.py
import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from make_OVatusData import OVatusDataset

def main():
    print("Kiểm tra OVatusDataset với preprocessing chuẩn UniverSeg...\n")
    dataset = OVatusDataset()

    print(f"Tổng số mẫu: {len(dataset)}")
    for i in range(min(3, len(dataset))):
        img, masks, path = dataset[i]
        print(f"\nMẫu {i}: {os.path.basename(path)}")
        print(f"   Image shape : {img.shape} (dtype: {img.dtype})")
        print(f"   Masks shape : {masks.shape}")

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2 + masks.shape[0], 1)
        plt.imshow(img.permute(1,2,0).numpy())
        plt.title("Image (normalized)")
        plt.axis('off')

        for j in range(masks.shape[0]):
            plt.subplot(1, 2 + masks.shape[0], 2 + j)
            plt.imshow(masks[j], cmap='gray')
            plt.title(f"Label {j+1}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    print("\nDataset sẵn sàng 100% cho UniverSeg!")

if __name__ == "__main__":
    main()