import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import h5py
import numpy as np
import os
from PhysicsGuidedNetwork import PhysicsGuidedNet

# ================= 配置 =================
# 确保路径正确
H5_PATH = "/root/autodl-tmp/merged_dataset_512_3d_fast_v2.h5"
MODEL_PATH = "best_model_symmetric.pth"  # 确保加载的是你刚刚训练完的模型
BATCH_SIZE = 128  # 评估时不需要反向传播，Batch 可以大一点
SCENE_SIZE = 5000.0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def evaluate_full_tta():
    print(f"🚀 启动全量验证集 TTA 评估...")
    print(f"数据集: {H5_PATH}")
    print(f"模型: {MODEL_PATH}")

    if not os.path.exists(H5_PATH):
        print("找不到数据集文件！")
        return

    # 1. 加载数据
    f = h5py.File(H5_PATH, 'r')
    total_samples = len(f['iq'])
    sample_iq = f['iq'][0]
    num_rx = sample_iq.shape[0] // 2

    # 获取验证集索引 (最后 10%)
    split_idx = int(0.9 * total_samples)
    val_indices = np.arange(split_idx, total_samples)
    print(f"验证集样本数: {len(val_indices)}")

    # 2. 加载模型
    model = PhysicsGuidedNet(num_rx=num_rx, signal_len=2048).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    model.eval()

    # 3. 分块读取并评估
    total_dist_err = 0.0
    processed_samples = 0

    # 每次读取 2000 个样本到内存，避免撑爆 RAM
    CHUNK_SIZE = 2000

    # IQ 通道置换索引 (用于 TTA)
    # H-Flip: 实部 0<->1, 2<->3 | 虚部 4<->5, 6<->7
    idx_perm = torch.tensor([1, 0, 3, 2, 5, 4, 7, 6], device=DEVICE)

    with torch.no_grad():
        pbar = tqdm(total=len(val_indices), desc="Evaluating TTA")

        for chunk_start in range(0, len(val_indices), CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, len(val_indices))
            current_indices = val_indices[chunk_start:chunk_end]
            current_indices = np.sort(current_indices)

            # 读入内存
            iq_ram = torch.from_numpy(f['iq'][current_indices]).float()
            heatmap_ram = torch.from_numpy(f['heatmap'][current_indices]).float()
            coord_ram = torch.from_numpy(f['coord'][current_indices]).float()

            dataset = TensorDataset(iq_ram, heatmap_ram, coord_ram)
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

            for iq, heatmap, true_coord in loader:
                iq, heatmap = iq.to(DEVICE), heatmap.to(DEVICE)
                true_coord = true_coord.to(DEVICE)

                # === TTA 核心逻辑 ===

                # 1. 原始预测
                pred_coord_1, _ = model(iq, heatmap)

                # 2. 翻转预测
                # A. 翻转 Heatmap
                heatmap_flip = torch.flip(heatmap, [3])
                # B. 交换 IQ 通道
                iq_flip = iq[:, idx_perm, :]

                # 预测
                pred_coord_flip, _ = model(iq_flip, heatmap_flip)

                # C. 还原坐标 (x' = 1 - x)
                pred_coord_2 = pred_coord_flip.clone()
                pred_coord_2[:, 0] = 1.0 - pred_coord_2[:, 0]

                # 3. 平均
                pred_coord_final = (pred_coord_1 + pred_coord_2) / 2.0

                # === 计算误差 ===
                true_xy = true_coord[:, :2] * SCENE_SIZE
                pred_xy = pred_coord_final * SCENE_SIZE

                dist = torch.norm(pred_xy - true_xy, dim=1)
                total_dist_err += dist.sum().item()
                processed_samples += iq.size(0)

            pbar.update(len(current_indices))

            # 释放内存
            del iq_ram, heatmap_ram, coord_ram, dataset, loader

    pbar.close()
    f.close()

    avg_error = total_dist_err / processed_samples
    print("\n" + "=" * 40)
    print(f"📊 全量验证集最终评估结果")
    print(f"🧪 测试样本数: {processed_samples}")
    print(f"🎯 TTA 平均定位误差: {avg_error:.4f} 米")
    print("=" * 40)


if __name__ == '__main__':
    evaluate_full_tta()