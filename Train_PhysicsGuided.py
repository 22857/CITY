import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
from PhysicsGuidedDataset import PhysicsGuidedHDF5Dataset
from PhysicsGuidedNetwork import PhysicsGuidedNet

# ================= 配置 =================
# 确保这里的文件名和你生成的一致
H5_PATH = r"D:\Dataset\SignalDataset\merged_dataset_512_3d_fast.h5"
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 50
SCENE_SIZE = 5000.0  # 用于反归一化计算米级误差
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    print(f"启动训练 | 设备: {DEVICE} | Batch: {BATCH_SIZE}")

    # 1. 数据准备
    print(f"打开数据集: {H5_PATH}")
    if not os.path.exists(H5_PATH):
        print(f"错误: 找不到文件 {H5_PATH}")
        return

    full_dataset = PhysicsGuidedHDF5Dataset(H5_PATH)

    # 划分训练集和验证集 (90% / 10%)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    # Windows 下 num_workers 设置过大可能会导致卡顿，建议 0 或 4
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"训练样本: {len(train_set)}, 验证样本: {len(val_set)}")

    # 2. 模型初始化
    # 从 dataset 拿一条数据看看 IQ 通道数
    sample_iq, _, _, _ = full_dataset[0]
    num_rx = sample_iq.shape[0] // 2

    print("初始化网络...")
    model = PhysicsGuidedNet(num_rx=num_rx, signal_len=2048).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 3. 损失函数
    criterion_coord = nn.MSELoss()
    criterion_mask = nn.MSELoss()

    # 4. 训练循环
    best_err = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_loss_sum = 0.0

        # 训练阶段
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]")
        for iq, heatmap, true_coord, mask in pbar:
            iq = iq.to(DEVICE)
            heatmap = heatmap.to(DEVICE)
            true_coord = true_coord.to(DEVICE)
            mask = mask.to(DEVICE)

            optimizer.zero_grad()

            # Forward
            pred_coord, pred_mask = model(iq, heatmap)

            # ===【关键修复】===
            # pred_coord 是 [B, 2] (XY)
            # true_coord 是 [B, 3] (XYZ)
            # 我们只取 true_coord 的前两列 (XY) 来计算 Loss
            true_coord_xy = true_coord[:, :2]

            # Loss 计算
            loss_c = criterion_coord(pred_coord, true_coord_xy)
            loss_m = criterion_mask(pred_mask, mask)

            # 总 Loss
            loss = loss_c + 0.5 * loss_m

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Coord': f"{loss_c.item():.4f}"})

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_coord_err = 0.0

        with torch.no_grad():
            for iq, heatmap, true_coord, mask in val_loader:
                iq = iq.to(DEVICE)
                heatmap = heatmap.to(DEVICE)
                true_coord = true_coord.to(DEVICE)
                mask = mask.to(DEVICE)

                pred_coord, pred_mask = model(iq, heatmap)

                # 同样只取 XY 进行 Loss 验证
                true_coord_xy = true_coord[:, :2]

                loss_c = criterion_coord(pred_coord, true_coord_xy)
                loss_m = criterion_mask(pred_mask, mask)
                val_loss += (loss_c + 0.5 * loss_m).item()

                # 计算实际距离误差 (单位: 米)
                # 此时 pred_coord 和 true_coord_xy 都是归一化的 [0, 1]
                # 欧氏距离 * 场景大小
                dist_norm = torch.norm(pred_coord - true_coord_xy, dim=1)
                dist_meter = dist_norm * SCENE_SIZE
                val_coord_err += dist_meter.mean().item()

        avg_val_loss = val_loss / len(val_loader)
        avg_dist_err = val_coord_err / len(val_loader)

        print(f"Epoch {epoch + 1} | Val Loss: {avg_val_loss:.6f} | Mean Error: {avg_dist_err:.2f} m")

        # 保存最优模型 (以定位误差最小为准)
        if avg_dist_err < best_err:
            best_err = avg_dist_err
            torch.save(model.state_dict(), "best_model.pth")
            print(f">>> New Best Model Saved! Error: {best_err:.2f} m")


if __name__ == '__main__':
    main()