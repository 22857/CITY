import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
from PhysicsGuidedNetwork import PhysicsGuidedNet

# ================= 配置 =================
H5_PATH = r"D:\Dataset\SignalDataset\merged_dataset_512_3d_fast.h5"
MODEL_PATH = "best_model_chunked.pth"  # 加载你训练好的最优模型
SCENE_SIZE = 5000.0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def visualize():
    # 1. 加载模型
    print(f"加载模型: {MODEL_PATH}")
    # 临时读取一个样本以获取 num_rx
    with h5py.File(H5_PATH, 'r') as f:
        sample_iq = f['iq'][0]
        num_rx = sample_iq.shape[0] // 2

    model = PhysicsGuidedNet(num_rx=num_rx, signal_len=2048).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # 2. 随机读取几个样本
    with h5py.File(H5_PATH, 'r') as f:
        total_num = len(f['iq'])
        # 随机选 3 个索引
        indices = np.random.choice(total_num, 3, replace=False)
        indices = np.sort(indices)

        iq = torch.from_numpy(f['iq'][indices]).float().to(DEVICE)
        heatmap = torch.from_numpy(f['heatmap'][indices]).float().to(DEVICE)
        mask = torch.from_numpy(f['mask'][indices]).float().to(DEVICE)
        true_coord = torch.from_numpy(f['coord'][indices]).float().to(DEVICE)

        # 3. 预测
        with torch.no_grad():
            pred_coord, pred_mask = model(iq, heatmap)

    # 4. 绘图
    iq = iq.cpu().numpy()
    heatmap = heatmap.cpu().numpy()
    mask = mask.cpu().numpy()
    pred_mask = pred_mask.cpu().numpy()
    true_coord = true_coord.cpu().numpy()[:, :2] * SCENE_SIZE
    pred_coord = pred_coord.cpu().numpy() * SCENE_SIZE

    for i in range(3):
        err = np.linalg.norm(true_coord[i] - pred_coord[i])

        plt.figure(figsize=(15, 4))

        # 子图1: 输入热力图
        plt.subplot(1, 3, 1)
        plt.title(f"Input Heatmap\nGT: {true_coord[i].astype(int)}")
        plt.imshow(heatmap[i, 0], origin='lower', extent=[0, SCENE_SIZE, 0, SCENE_SIZE], cmap='jet')
        plt.scatter(true_coord[i, 0], true_coord[i, 1], c='r', marker='x', s=100, label='GT')
        plt.scatter(pred_coord[i, 0], pred_coord[i, 1], c='w', marker='o', s=100, label='Pred')
        plt.legend()

        # 子图2: 真实 Mask
        plt.subplot(1, 3, 2)
        plt.title("Ground Truth Mask")
        plt.imshow(mask[i, 0], origin='lower', extent=[0, SCENE_SIZE, 0, SCENE_SIZE], cmap='gray')

        # 子图3: 预测 Mask
        plt.subplot(1, 3, 3)
        plt.title(f"Predicted Mask\nError: {err:.1f} m")
        plt.imshow(pred_mask[i, 0], origin='lower', extent=[0, SCENE_SIZE, 0, SCENE_SIZE], cmap='gray')

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    visualize()