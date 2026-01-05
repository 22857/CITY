import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
from PhysicsGuidedNetwork import PhysicsGuidedNet

# ================= 配置 =================
H5_PATH = r"D:\Dataset\SignalDataset\merged_dataset_512_3d_fast.h5"
# 注意：根据你上一轮的训练日志，最终模型保存为 best_model_final.pth
# 如果你找不到文件，请检查目录下是否有 best_model_chunked.pth
MODEL_PATH = "best_model_chunked.pth"
SCENE_SIZE = 5000.0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def visualize():
    if not os.path.exists(MODEL_PATH):
        print(f"【错误】找不到模型文件: {MODEL_PATH}")
        print("请检查文件名是否正确，或者是否使用了 best_model_chunked.pth")
        return

    print(f"加载模型: {MODEL_PATH}")

    # 1. 获取网络参数
    with h5py.File(H5_PATH, 'r') as f:
        sample_iq = f['iq'][0]
        num_rx = sample_iq.shape[0] // 2
        total_num = len(f['iq'])

        # 2. 随机抽取 3 个样本
        # 即使使用了 os.environ 设置，有时 matplotlib 也会有冲突，
        # 如果再次报错，可以尝试把这行移到 if __name__ == '__main__': 之后
        indices = np.random.choice(total_num, 3, replace=False)
        indices = np.sort(indices)

        print(f"可视化样本索引: {indices}")

        iq = torch.from_numpy(f['iq'][indices]).float().to(DEVICE)
        heatmap = torch.from_numpy(f['heatmap'][indices]).float().to(DEVICE)
        mask = torch.from_numpy(f['mask'][indices]).float().to(DEVICE)
        true_coord = torch.from_numpy(f['coord'][indices]).float().to(DEVICE)

        # 3. 加载模型并预测
        model = PhysicsGuidedNet(num_rx=num_rx, signal_len=2048).to(DEVICE)
        # map_location 确保在只有 CPU 的机器上也能跑
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()

        with torch.no_grad():
            pred_coord, pred_mask = model(iq, heatmap)

    # 4. 绘图
    iq = iq.cpu().numpy()
    heatmap = heatmap.cpu().numpy()
    mask = mask.cpu().numpy()
    pred_mask = torch.sigmoid(pred_mask).cpu().numpy()

    # 坐标反归一化 (0-1 -> 0-5000)
    true_coord = true_coord.cpu().numpy()[:, :2] * SCENE_SIZE
    pred_coord = pred_coord.cpu().numpy() * SCENE_SIZE

    for i in range(3):
        dist_err = np.linalg.norm(true_coord[i] - pred_coord[i])

        plt.figure(figsize=(15, 5))

        # 子图1: Input Heatmap + 坐标
        plt.subplot(1, 3, 1)
        plt.title(f"Input Heatmap\nGT(Red):{true_coord[i].astype(int)}\nPred(Wht):{pred_coord[i].astype(int)}")
        # 显示热力图
        plt.imshow(heatmap[i, 0], origin='lower', extent=[0, SCENE_SIZE, 0, SCENE_SIZE], cmap='jet')
        plt.scatter(true_coord[i, 0], true_coord[i, 1], c='r', marker='x', s=120, linewidths=2, label='GT')
        plt.scatter(pred_coord[i, 0], pred_coord[i, 1], c='white', marker='o', s=120, edgecolors='black', label='Pred')
        plt.legend(loc='upper right')

        # 子图2: Label Mask
        plt.subplot(1, 3, 2)
        plt.title("Ground Truth Mask")
        plt.imshow(mask[i, 0], origin='lower', extent=[0, SCENE_SIZE, 0, SCENE_SIZE], cmap='gray')

        # 子图3: Predict Mask
        plt.subplot(1, 3, 3)
        plt.title(f"Predicted Mask\nDist Error: {dist_err:.1f} m")
        plt.imshow(pred_mask[i, 0], origin='lower', extent=[0, SCENE_SIZE, 0, SCENE_SIZE], cmap='gray')

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    visualize()