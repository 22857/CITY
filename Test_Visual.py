import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
from PhysicsGuidedNetwork import PhysicsGuidedNet

# ================= 配置 =================
H5_PATH = r"D:\Dataset\SignalDataset\merged_dataset_512_3d_fast.h5"
# 使用最新的对称增强模型
MODEL_PATH = "best_model_symmetric.pth"
SCENE_SIZE = 5000.0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def visualize():
    if not os.path.exists(MODEL_PATH):
        print(f"【错误】找不到模型文件: {MODEL_PATH}")
        return

    print(f"加载模型: {MODEL_PATH}")

    # 1. 获取数据
    with h5py.File(H5_PATH, 'r') as f:
        sample_iq = f['iq'][0]
        num_rx = sample_iq.shape[0] // 2
        total_num = len(f['iq'])

        # 随机抽取 3 个样本
        indices = np.random.choice(total_num, 3, replace=False)
        indices = np.sort(indices)
        print(f"可视化样本索引: {indices}")

        iq = torch.from_numpy(f['iq'][indices]).float().to(DEVICE)
        heatmap = torch.from_numpy(f['heatmap'][indices]).float().to(DEVICE)
        mask = torch.from_numpy(f['mask'][indices]).float().to(DEVICE)
        true_coord = torch.from_numpy(f['coord'][indices]).float().to(DEVICE)

        # 2. 加载模型
        model = PhysicsGuidedNet(num_rx=num_rx, signal_len=2048).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()

        # 3. TTA 推理 (关键部分)
        with torch.no_grad():
            # === Pass 1: 原始预测 ===
            pred_coord_1, pred_mask_1 = model(iq, heatmap)

            # === Pass 2: 水平翻转预测 (H-Flip TTA) ===
            # A. 翻转 Heatmap (Width=dim 3)
            heatmap_flip = torch.flip(heatmap, [3])

            # B. 交换 IQ 通道 (与训练时保持一致)
            # 实部: 0<->1, 2<->3 (indices: 1,0,3,2)
            # 虚部: 4<->5, 6<->7 (indices: 5,4,7,6)
            idx_perm = torch.tensor([1, 0, 3, 2, 5, 4, 7, 6], device=DEVICE)
            iq_flip = iq[:, idx_perm, :]

            # 预测
            pred_coord_flip, pred_mask_flip = model(iq_flip, heatmap_flip)

            # === 还原结果 ===
            # Mask 翻转回来
            pred_mask_2 = torch.flip(pred_mask_flip, [3])

            # 坐标 x 还原: 1 - x
            pred_coord_2 = pred_coord_flip.clone()
            pred_coord_2[:, 0] = 1.0 - pred_coord_2[:, 0]

            # === 平均 ===
            # 对 Logits 进行平均 (也可以先Sigmoid再平均，Logits平均更平滑)
            pred_mask_logits = (pred_mask_1 + pred_mask_2) / 2.0
            pred_coord = (pred_coord_1 + pred_coord_2) / 2.0

    # 4. 数据转换与绘图
    iq = iq.cpu().numpy()
    heatmap = heatmap.cpu().numpy()
    mask = mask.cpu().numpy()

    # 手动 Sigmoid 显示概率
    pred_mask = torch.sigmoid(pred_mask_logits).cpu().numpy()

    true_coord = true_coord.cpu().numpy()[:, :2] * SCENE_SIZE
    pred_coord = pred_coord.cpu().numpy() * SCENE_SIZE

    for i in range(3):
        dist_err = np.linalg.norm(true_coord[i] - pred_coord[i])

        plt.figure(figsize=(15, 5))

        # 子图1: Input Heatmap
        plt.subplot(1, 3, 1)
        plt.title(f"Input Heatmap\nGT:{true_coord[i].astype(int)}\nPred:{pred_coord[i].astype(int)}")
        plt.imshow(heatmap[i, 0], origin='lower', extent=[0, SCENE_SIZE, 0, SCENE_SIZE], cmap='jet')
        plt.scatter(true_coord[i, 0], true_coord[i, 1], c='r', marker='x', s=120, linewidths=2, label='GT')
        plt.scatter(pred_coord[i, 0], pred_coord[i, 1], c='white', marker='o', s=120, edgecolors='black',
                    label='Pred (TTA)')
        plt.legend(loc='upper right')

        # 子图2: GT Mask
        plt.subplot(1, 3, 2)
        plt.title("Ground Truth Mask")
        plt.imshow(mask[i, 0], origin='lower', extent=[0, SCENE_SIZE, 0, SCENE_SIZE], cmap='gray')

        # 子图3: Pred Mask (TTA)
        plt.subplot(1, 3, 3)
        plt.title(f"Predicted Mask (TTA)\nDist Error: {dist_err:.1f} m")
        plt.imshow(pred_mask[i, 0], origin='lower', extent=[0, SCENE_SIZE, 0, SCENE_SIZE], cmap='gray')

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    visualize()