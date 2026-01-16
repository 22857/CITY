import h5py
import matplotlib.pyplot as plt
import numpy as np

# 你的数据路径
H5_PATH = r"/root/autodl-tmp/merged_dataset_512_3d_valid.h5"

with h5py.File(H5_PATH, 'r') as f:
    # 随机选一个样本
    idx = np.random.randint(0, len(f['coord']))

    # 读取
    heatmap = f['heatmap'][idx][0].astype(np.float32) # Input
    mask = f['mask'][idx][0].astype(np.float32)       # Label
    coord = f['coord'][idx]                           # True Pos (normalized)

    print(f"Sample {idx}")
    print(f"True Coordinate: {coord * 5000} m")

    # 画图
    plt.figure(figsize=(12, 5))

    # 1. 输入热力图
    plt.subplot(1, 2, 1)
    plt.title("Input: DPD Heatmap")
    plt.imshow(heatmap, origin='lower', extent=[0, 5000, 0, 5000], cmap='jet')
    plt.colorbar()
    # 画出真值点
    plt.scatter(coord[0]*5000, coord[1]*5000, c='red', marker='x', s=100, label='True Tx')
    plt.legend()

    # 2. 标签掩码
    plt.subplot(1, 2, 2)
    plt.title("Label: Soft Mask")
    plt.imshow(mask, origin='lower', extent=[0, 5000, 0, 5000], cmap='gray')
    plt.scatter(coord[0]*5000, coord[1]*5000, c='red', marker='x', s=100)

    plt.show()