import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
# 导入网络和数据集
from PhysicsGuidedNetwork import PhysicsGuidedNet
from PhysicsGuidedDataset import PhysicsGuidedHDF5Dataset

# ================= 配置区域 =================

# 1. 数据集路径 (指向 MakeCsvIQData -> Generate_Multimodal_Data 生成的独立文件)
TRAIN_H5_PATH = "/root/autodl-tmp/merged_dataset_512_3d_train.h5"
VAL_H5_PATH = "/root/autodl-tmp/merged_dataset_512_3d_valid.h5"

# 2. 保存路径
SAVE_PATH = "best_model_urban_512.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 3. 核心训练参数
# 24G 显存 (3090/4090) -> 32
# 16G 显存 (V100/T4)   -> 16
# 12G 显存 (1080Ti)    -> 8
BATCH_SIZE = 32
NUM_WORKERS = 8
LR = 1e-4
EPOCHS = 50
SCENE_SIZE = 5000.0

# 【关键】明确指定分辨率和接收机数量，必须与数据生成一致
MAP_SIZE = 512
NUM_RX = 6
SIGNAL_LEN = 2048


# ================= 工具模块 =================

def apply_augmentation(iq, heatmap, coord, mask):
    """
    数据增强：随机翻转 Heatmap 和 Mask，并同步调整坐标
    注意：暂不翻转 IQ 通道，避免复杂的 6Rx 索引映射问题
    """
    # 随机水平翻转 (H-Flip)
    if np.random.rand() > 0.5:
        heatmap = torch.flip(heatmap, [3])
        mask = torch.flip(mask, [3])
        coord[:, 0] = 1.0 - coord[:, 0]

    # 随机垂直翻转 (V-Flip)
    if np.random.rand() > 0.5:
        heatmap = torch.flip(heatmap, [2])
        mask = torch.flip(mask, [2])
        coord[:, 1] = 1.0 - coord[:, 1]

    return iq, heatmap, coord, mask


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred_logits, target):
        pred_probs = torch.sigmoid(pred_logits)
        # Flatten for Dice calculation
        pred_flat = pred_probs.view(pred_probs.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        intersection = (pred_flat * target_flat).sum(1)
        dice = (2. * intersection + self.smooth) / (pred_flat.sum(1) + target_flat.sum(1) + self.smooth)
        return 1 - dice.mean()


def validate(model, loader):
    """
    验证函数：计算平均距离误差 (米)
    已移除所有边缘过滤逻辑，全量评估
    """
    model.eval()
    total_dist_err = 0.0
    num_samples = 0

    with torch.no_grad():
        for iq, heatmap, coord, mask in loader:
            iq, heatmap, coord = iq.to(DEVICE), heatmap.to(DEVICE), coord.to(DEVICE)

            with torch.cuda.amp.autocast():
                pred_coord, _ = model(iq, heatmap)

            # 计算真实距离误差 (Euclidean Distance)
            # coord[:, :2] 是归一化坐标 (0~1)，需乘 SCENE_SIZE 还原为米
            dist_err = torch.norm(pred_coord - coord[:, :2], dim=1) * SCENE_SIZE

            total_dist_err += dist_err.sum().item()
            num_samples += iq.size(0)

    if num_samples == 0: return 9999.0
    return total_dist_err / num_samples


# ================= 主程序 =================
def main():
    print(f"🚀 启动城市高精定位训练 | {MAP_SIZE}x{MAP_SIZE} | {NUM_RX}Rx | 设备: {DEVICE}")
    print(f"📦 Batch Size: {BATCH_SIZE}")

    # 1. 加载双数据集
    print(f"Loading Train Set: {TRAIN_H5_PATH}")
    if not os.path.exists(TRAIN_H5_PATH):
        raise FileNotFoundError(f"找不到训练文件: {TRAIN_H5_PATH}")
    train_ds = PhysicsGuidedHDF5Dataset(TRAIN_H5_PATH)

    print(f"Loading Val Set:   {VAL_H5_PATH}")
    if not os.path.exists(VAL_H5_PATH):
        raise FileNotFoundError(f"找不到验证文件: {VAL_H5_PATH}")
    val_ds = PhysicsGuidedHDF5Dataset(VAL_H5_PATH)

    print(f"📊 训练样本: {len(train_ds)} | 验证样本: {len(val_ds)}")

    # DataLoader
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    # 2. 模型初始化
    # 显式传入 map_size=512 以匹配 PhysicsGuidedNetwork 中的全连接层定义
    model = PhysicsGuidedNet(num_rx=NUM_RX, signal_len=SIGNAL_LEN, map_size=MAP_SIZE).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)
    scaler = torch.cuda.amp.GradScaler()

    # 3. Loss 定义
    criterion_coord = nn.L1Loss()  # 默认 mean reduction
    # 针对 512x512 的稀疏目标，给予正样本极高权重 (50.0)
    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([50.0]).to(DEVICE))
    criterion_dice = DiceLoss()

    best_err = float('inf')

    # 4. 训练循环
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for iq, heatmap, coord, mask in pbar:
            iq, heatmap, coord, mask = iq.to(DEVICE), heatmap.to(DEVICE), coord.to(DEVICE), mask.to(DEVICE)

            # 数据增强
            iq, heatmap, coord, mask = apply_augmentation(iq, heatmap, coord, mask)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                # Forward
                pred_coord, pred_mask = model(iq, heatmap)

                # --- Loss Calculation ---

                # A. 坐标回归 Loss (核心任务，权重加倍)
                loss_c = criterion_coord(pred_coord, coord[:, :2])

                # B. Mask 分割 Loss (辅助任务，权重降低)
                loss_bce = criterion_bce(pred_mask, mask)
                loss_dice = criterion_dice(pred_mask, mask)
                loss_m = loss_bce + loss_dice

                # C. 一致性 Loss (王者归来：带 IQ 置换的 TTA)
                # 只有加上这个 IQ 置换，TTA 才是对的！
                loss_consistency = torch.tensor(0.0, device=DEVICE)
                if True:
                    # 1. 翻转 Heatmap
                    heatmap_flip = torch.flip(heatmap, [3])

                    # 2. 【关键】翻转 IQ 通道 (6Rx 正六边形)
                    # 索引映射: Rx3, Rx2, Rx1, Rx0, Rx5, Rx4
                    idx_perm = torch.tensor([6, 7, 4, 5, 2, 3, 0, 1, 10, 11, 8, 9], device=DEVICE)
                    iq_flip = iq[:, idx_perm, :]

                    # 3. 传入翻转后的 iq_flip
                    pred_coord_flip, _ = model(iq_flip, heatmap_flip)

                    # 4. 还原坐标
                    pred_restored = pred_coord_flip.clone()
                    pred_restored[:, 0] = 1.0 - pred_restored[:, 0]

                    loss_consistency = torch.nn.functional.l1_loss(pred_coord, pred_restored.detach())

                # D. 总 Loss (重新配比)
                # 强坐标(10.0)，弱绘图(0.1)，强一致性(2.0)
                total_loss = 10.0 * loss_c + 0.1 * loss_m + 2.0 * loss_consistency

            # Backward
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += total_loss.item()
            pbar.set_postfix({
                'Loss': f"{total_loss.item():.3f}",
                'L_c': f"{loss_c.item():.3f}",
                'Consis': f"{loss_consistency.item():.3f}"
            })

        # 验证阶段
        val_err = validate(model, val_loader)
        print(f"Epoch {epoch + 1} 验证误差: {val_err:.2f}m")
        scheduler.step(val_err)

        # 保存最佳模型
        if val_err < best_err:
            best_err = val_err
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"🌟 新纪录: {best_err:.2f}m (已保存)")


if __name__ == '__main__':
    main()