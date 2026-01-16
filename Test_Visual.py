import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
from PhysicsGuidedNetwork import PhysicsGuidedNet

# 尝试导入骨架化算法
try:
    from skimage.morphology import skeletonize

    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("【提示】未检测到 scikit-image 库，将只使用二值化显示 (建议 pip install scikit-image)")

# ================= 配置区域 =================
# 确保这里指向的是包含 6Rx 数据的验证集
H5_PATH = r"/root/autodl-tmp/merged_dataset_512_3d_valid.h5"
MODEL_PATH = "best_model_urban_512.pth"

SCENE_SIZE = 5000.0
MAP_SIZE = 512  # 必须与训练时一致
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
        print(f"检测到接收机数量: {num_rx} (通道数: {sample_iq.shape[0]})")

        total_num = len(f['iq'])
        indices = np.random.choice(total_num, 3, replace=False)
        indices = np.sort(indices)
        print(f"可视化样本索引: {indices}")

        iq = torch.from_numpy(f['iq'][indices]).float().to(DEVICE)
        heatmap = torch.from_numpy(f['heatmap'][indices]).float().to(DEVICE)
        mask = torch.from_numpy(f['mask'][indices]).float().to(DEVICE)
        true_coord = torch.from_numpy(f['coord'][indices]).float().to(DEVICE)

        # 2. 加载模型
        model = PhysicsGuidedNet(num_rx=num_rx, signal_len=2048, map_size=MAP_SIZE).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()

        # 3. TTA 推理
        with torch.no_grad():
            pred_coord_1, pred_mask_1 = model(iq, heatmap)

            # TTA 翻转逻辑
            heatmap_flip = torch.flip(heatmap, [3])
            if num_rx == 6:
                idx_perm = torch.tensor([6, 7, 4, 5, 2, 3, 0, 1, 10, 11, 8, 9], device=DEVICE)
                iq_flip = iq[:, idx_perm, :]
            else:
                iq_flip = iq

            pred_coord_flip, pred_mask_flip = model(iq_flip, heatmap_flip)

            pred_mask_2 = torch.flip(pred_mask_flip, [3])
            pred_coord_2 = pred_coord_flip.clone()
            pred_coord_2[:, 0] = 1.0 - pred_coord_2[:, 0]

            pred_mask_logits = (pred_mask_1 + pred_mask_2) / 2.0
            pred_coord = (pred_coord_1 + pred_coord_2) / 2.0

    # 4. 后处理与绘图
    iq = iq.cpu().numpy()
    heatmap = heatmap.cpu().numpy()
    mask = mask.cpu().numpy()
    pred_prob = torch.sigmoid(pred_mask_logits).cpu().numpy()  # [B, 1, H, W]

    true_coord = true_coord.cpu().numpy()[:, :2] * SCENE_SIZE
    pred_coord = pred_coord.cpu().numpy() * SCENE_SIZE

    for i in range(3):
        dist_err = np.linalg.norm(true_coord[i] - pred_coord[i])

        # === 调试：打印概率值的统计信息 ===
        p_min = pred_prob[i, 0].min()
        p_max = pred_prob[i, 0].max()
        p_mean = pred_prob[i, 0].mean()
        print(f"样本 {indices[i]} | 预测概率范围: Min={p_min:.4f}, Max={p_max:.4f}, Mean={p_mean:.4f}")

        plt.figure(figsize=(18, 6))

        # 子图 1: Input
        plt.subplot(1, 3, 1)
        plt.title(f"Input Heatmap\nErr: {dist_err:.1f}m")
        plt.imshow(heatmap[i, 0], origin='lower', extent=[0, SCENE_SIZE, 0, SCENE_SIZE], cmap='jet')
        plt.scatter(true_coord[i, 0], true_coord[i, 1], c='r', marker='x', s=150, linewidths=3, label='GT')
        plt.scatter(pred_coord[i, 0], pred_coord[i, 1], c='w', marker='o', s=150, edgecolors='k', label='Pred')

        # 子图 2: GT
        plt.subplot(1, 3, 2)
        plt.title("Ground Truth Mask")
        plt.imshow(mask[i, 0], origin='lower', extent=[0, SCENE_SIZE, 0, SCENE_SIZE], cmap='gray')

        # 子图 3: 智能处理后的预测
        plt.subplot(1, 3, 3)
        plt.title(f"Processed Prediction\n(Max Prob: {p_max:.2f})")

        # === 核心修改：自适应阈值 ===
        # 如果最大概率太小 (<0.01)，说明模型没输出任何东西，显示黑图
        if p_max < 0.01:
            final_vis = np.zeros_like(pred_prob[i, 0])
            print("  -> 警告：模型输出概率过低，显示全黑")
        else:
            # 动态阈值：取最大值的 40% 作为门槛
            # 比如最大值是 0.2，阈值就是 0.08，这样能保证一定会显示出东西
            dynamic_thresh = p_max * 0.4
            binary_mask = pred_prob[i, 0] > dynamic_thresh

            if HAS_SKIMAGE:
                # 骨架化需要 bool 类型
                final_vis = skeletonize(binary_mask)
            else:
                final_vis = binary_mask

        plt.imshow(final_vis, origin='lower', extent=[0, SCENE_SIZE, 0, SCENE_SIZE], cmap='gray')
        plt.scatter(true_coord[i, 0], true_coord[i, 1], c='r', marker='x', s=100)  # 叠加红叉方便对比

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    visualize()