import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import h5py
import numpy as np
import os
from PhysicsGuidedNetwork import PhysicsGuidedNet

# ================= 路径配置 =================
# 请确保该路径指向你生成的 512x512 HDF5 文件
H5_PATH = r"D:\Dataset\SignalDataset\merged_dataset_512_3d_fast.h5"

# ================= 超参数配置 =================
BATCH_SIZE = 32  # GPU 计算时的批次大小
CHUNK_SIZE = 2000  # 每次从硬盘读入内存的样本数
LR = 1e-4  # 初始学习率
EPOCHS = 50  # 总训练轮数
SCENE_SIZE = 5000.0  # 场景物理尺寸 (米)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ================= 验证函数 =================
def validate(model, val_indices, h5_file, criterion_coord, criterion_bce, criterion_dice, chunk_size=1000, batch_size=32):
    """
    分块加载验证数据，并使用小 Batch 推理，防止 OOM。
    """
    model.eval()
    total_loss = 0.0
    total_dist_err = 0.0
    num_samples = 0

    with torch.no_grad():
        # 外层循环：分块从硬盘读入内存
        for i in range(0, len(val_indices), chunk_size):
            # 1. 读取当前块数据
            current_indices = val_indices[i: i + chunk_size]
            current_indices = np.sort(current_indices)  # HDF5 要求升序索引

            # 读入 CPU 内存 (RAM)
            iq_ram = torch.from_numpy(h5_file['iq'][current_indices]).float()
            heatmap_ram = torch.from_numpy(h5_file['heatmap'][current_indices]).float()
            mask_ram = torch.from_numpy(h5_file['mask'][current_indices]).float()
            coord_ram = torch.from_numpy(h5_file['coord'][current_indices]).float()

            # 2. 构造临时 DataLoader (RAM -> GPU)
            temp_dataset = TensorDataset(iq_ram, heatmap_ram, coord_ram, mask_ram)
            temp_loader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

            # 3. 内层循环：小批次推理
            for iq, heatmap, true_coord, mask in temp_loader:
                iq, heatmap = iq.to(DEVICE), heatmap.to(DEVICE)
                mask, true_coord = mask.to(DEVICE), true_coord.to(DEVICE)

                # 预测
                pred_coord, pred_mask = model(iq, heatmap)

                # Loss 计算 (注意 3D -> 2D 切片)
                true_coord_xy = true_coord[:, :2]
                loss_c = criterion_coord(pred_coord, true_coord_xy)
                # 使用传入的混合 Loss 计算验证集损失
                loss_b = criterion_bce(pred_mask, mask)
                loss_d = criterion_dice(pred_mask, mask)

                # 验证集权重固定即可，主要参考 dist_err
                loss_total = loss_c + 0.5 * (loss_b + loss_d)

                # 累加 Loss
                batch_len = iq.size(0)
                total_loss += loss_total.item() * batch_len  # 使用计算出的 loss_total

                # 累加距离误差 (米)
                dist_meter = torch.norm(pred_coord - true_coord_xy, dim=1) * SCENE_SIZE
                total_dist_err += dist_meter.sum().item()

                num_samples += batch_len

            # 手动释放内存
            del iq_ram, heatmap_ram, mask_ram, coord_ram, temp_dataset, temp_loader

    return total_loss / num_samples, total_dist_err / num_samples


# ================= 主程序 =================
def main():
    print(f"🚀 启动增强版训练 | Chunk: {CHUNK_SIZE} | Batch: {BATCH_SIZE} | Device: {DEVICE}")

    if not os.path.exists(H5_PATH):
        print(f"【错误】找不到数据集文件: {H5_PATH}")
        return

    # 1. 打开 HDF5 (只读取元数据)
    f = h5py.File(H5_PATH, 'r')
    total_samples = len(f['iq'])
    print(f"数据集总样本数: {total_samples}")

    # 2. 划分训练集/验证集 (90% / 10%)
    all_indices = np.arange(total_samples)
    split_idx = int(0.9 * total_samples)
    train_indices_all = all_indices[:split_idx]
    val_indices_all = all_indices[split_idx:]

    print(f"训练集: {len(train_indices_all)}, 验证集: {len(val_indices_all)}")

    # 3. 初始化模型
    sample_iq = f['iq'][0]
    num_rx = sample_iq.shape[0] // 2

    model = PhysicsGuidedNet(num_rx=num_rx, signal_len=2048).to(DEVICE)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    # 【修复】移除了 verbose=True，防止报错
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )

    criterion_coord = nn.MSELoss()

    # 1. 定义 Dice Loss 类 (解决形状模糊)
    class DiceLoss(nn.Module):
        def __init__(self, smooth=1.0):
            super(DiceLoss, self).__init__()
            self.smooth = smooth

        def forward(self, pred_logits, target):
            # 将 Logits 转为概率 (0-1)
            pred_probs = torch.sigmoid(pred_logits)

            # 展平所有维度，只计算重叠度
            pred_flat = pred_probs.view(-1)
            target_flat = target.view(-1)

            intersection = (pred_flat * target_flat).sum()

            # Dice 系数 = 2 * 交集 / (并集 + 平滑项)
            dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)

            return 1 - dice

    # 2. 定义加权 BCE (解决正负样本不平衡)
    # 假设线条像素很少，给予 20 倍权重，强迫网络关注白色线条
    pos_weight = torch.tensor([20.0]).to(DEVICE)

    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_dice = DiceLoss()

    best_err = float('inf')

    # ================= 训练循环 =================
    try:
        for epoch in range(EPOCHS):
            model.train()
            train_loss_epoch = 0.0

            # 进度条
            pbar = tqdm(total=len(train_indices_all), desc=f"Epoch {epoch + 1}/{EPOCHS}")

            # --- Chunk Loading: 分块读入内存 ---
            for chunk_start in range(0, len(train_indices_all), CHUNK_SIZE):
                chunk_end = min(chunk_start + CHUNK_SIZE, len(train_indices_all))

                # A. 硬盘 -> 内存 (RAM)
                iq_ram = torch.from_numpy(f['iq'][chunk_start:chunk_end])
                map_ram = torch.from_numpy(f['heatmap'][chunk_start:chunk_end])
                mask_ram = torch.from_numpy(f['mask'][chunk_start:chunk_end])
                coord_ram = torch.from_numpy(f['coord'][chunk_start:chunk_end])

                # B. 内存 -> DataLoader
                mem_dataset = TensorDataset(iq_ram, map_ram, coord_ram, mask_ram)
                train_loader = DataLoader(mem_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

                # C. GPU 训练
                for iq, heatmap, true_coord, mask in train_loader:
                    iq, heatmap = iq.to(DEVICE), heatmap.to(DEVICE)
                    mask, true_coord = mask.to(DEVICE), true_coord.to(DEVICE)

                    optimizer.zero_grad()

                    # 前向传播
                    pred_coord, pred_mask = model(iq, heatmap)

                    # 3D 标签切片为 2D
                    true_coord_xy = true_coord[:, :2]

                    loss_c = criterion_coord(pred_coord, true_coord_xy)
                    # 1. 像素级分类 Loss (带权重)
                    loss_bce = criterion_bce(pred_mask, mask)

                    # 2. 形状级 Dice Loss
                    loss_dice = criterion_dice(pred_mask, mask)

                    # 3. 组合 Mask Loss
                    loss_m = loss_bce + loss_dice

                    # 动态权重调整：前期侧重学形状(Mask)，后期侧重修坐标(Coord)
                    # 如果 epoch 小于 20，Mask 的权重给大一点 (0.5)，让 Mask 先成型
                    mask_weight = 0.5 if epoch < 20 else 0.1

                    loss = loss_c + mask_weight * loss_m

                    loss.backward()
                    optimizer.step()

                    train_loss_epoch += loss.item() * iq.size(0)

                # 更新进度条
                current_lr = optimizer.param_groups[0]['lr']
                pbar.update(chunk_end - chunk_start)
                pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'LR': f"{current_lr:.1e}"})

                # 释放内存
                del iq_ram, map_ram, mask_ram, coord_ram, mem_dataset, train_loader

            pbar.close()

            # --- 验证阶段 ---
            print("正在验证...")
            avg_val_loss, avg_dist_err = validate(
                model,
                val_indices_all,
                f,
                criterion_coord,
                criterion_bce,
                criterion_dice,
                chunk_size=CHUNK_SIZE,
                batch_size=BATCH_SIZE
            )

            avg_train_loss = train_loss_epoch / len(train_indices_all)
            print(
                f"Epoch {epoch + 1} 结果: Train Loss={avg_train_loss:.5f}, Val Loss={avg_val_loss:.5f}, 平均误差={avg_dist_err:.2f}m")

            # --- 学习率调整 (手动实现 Verbose) ---
            last_lr = optimizer.param_groups[0]['lr']
            scheduler.step(avg_dist_err)
            new_lr = optimizer.param_groups[0]['lr']

            if new_lr != last_lr:
                print(f"📉 学习率自动衰减: {last_lr:.1e} -> {new_lr:.1e}")

            # --- 保存模型 ---
            if avg_dist_err < best_err:
                best_err = avg_dist_err
                torch.save(model.state_dict(), "best_model_final.pth")
                print(f">>> 发现新最优模型！误差: {best_err:.2f}m，已保存。")

    except KeyboardInterrupt:
        print("\n训练被手动中断。")
    finally:
        f.close()
        print("HDF5 文件句柄已关闭。")


if __name__ == '__main__':
    main()