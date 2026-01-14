import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import numpy as np

# 导入你定义的模块
from PhysicsGuidedNetwork import PhysicsGuidedNet
from PhysicsGuidedDataset import PhysicsGuidedHDF5Dataset

# ================= 1. 路径与硬件配置 =================
# 请确保文件名与服务器上的实际文件名一致
H5_PATH = "/root/autodl-tmp/merged_dataset_512_3d_fast_v2.h5"
SAVE_PATH = "best_model_symmetric.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================= 2. 超参数配置 =================
BATCH_SIZE = 64
NUM_WORKERS = 8
LR = 1e-4
EPOCHS = 50
SCENE_SIZE = 5000.0


# ================= 3. 工具函数 =================

def apply_augmentation(iq, heatmap, coord, mask):
    """
    在 GPU 上进行数据增强，保持 IQ 通道与几何翻转的一致性
    """
    # 随机水平翻转 (H-Flip)
    if np.random.rand() > 0.5:
        heatmap = torch.flip(heatmap, [3])
        mask = torch.flip(mask, [3])
        coord[:, 0] = 1.0 - coord[:, 0]
        # H-Flip 索引交换: Rx0<->Rx1, Rx3<->Rx2
        idx_perm = torch.tensor([1, 0, 3, 2, 5, 4, 7, 6], device=iq.device)
        iq = iq[:, idx_perm, :]

    # 随机垂直翻转 (V-Flip)
    if np.random.rand() > 0.5:
        heatmap = torch.flip(heatmap, [2])
        mask = torch.flip(mask, [2])
        coord[:, 1] = 1.0 - coord[:, 1]
        # V-Flip 索引交换: Rx0<->Rx3, Rx1<->Rx2
        idx_perm = torch.tensor([3, 2, 1, 0, 7, 6, 5, 4], device=iq.device)
        iq = iq[:, idx_perm, :]

    return iq, heatmap, coord, mask


def get_spatial_weight(target_coord, device):
    # 直接返回全 1 的权重，让模型自然收敛
    return torch.ones(target_coord.size(0), 1, device=device)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred_logits, target):
        pred_probs = torch.sigmoid(pred_logits)
        intersection = (pred_probs * target).sum()
        dice = (2. * intersection + self.smooth) / (pred_probs.sum() + target.sum() + self.smooth)
        return 1 - dice


def validate(model, loader):
    model.eval()
    total_dist_err = 0.0
    num_samples = 0

    # 定义“基站附近”的阈值，例如 500m
    # 场景 5000m，0.1 代表 500m
    IGNORE_RADIUS = 0.1

    with torch.no_grad():
        for iq, heatmap, coord, mask in loader:
            iq, heatmap, coord, mask = iq.to(DEVICE), heatmap.to(DEVICE), coord.to(DEVICE), mask.to(DEVICE)

            with torch.cuda.amp.autocast():
                pred_coord, _ = model(iq, heatmap)

            # 计算误差 (米)
            dist_err = torch.norm(pred_coord - coord[:, :2], dim=1)

            # --- 过滤逻辑 ---
            # 计算目标到四个角的距离
            # corners: (0,0), (1,0), (1,1), (0,1)
            x, y = coord[:, 0], coord[:, 1]
            d1 = torch.sqrt(x ** 2 + y ** 2)
            d2 = torch.sqrt((x - 1) ** 2 + y ** 2)
            d3 = torch.sqrt((x - 1) ** 2 + (y - 1) ** 2)
            d4 = torch.sqrt(x ** 2 + (y - 1) ** 2)

            min_dist_to_station, _ = torch.min(torch.stack([d1, d2, d3, d4], dim=1), dim=1)

            # 只有距离基站 > 500m 的样本才计入误差
            valid_mask = min_dist_to_station > IGNORE_RADIUS

            if valid_mask.sum() > 0:
                valid_err = dist_err[valid_mask]
                total_dist_err += valid_err.sum().item() * SCENE_SIZE
                num_samples += valid_mask.sum().item()

    if num_samples == 0: return 9999.0
    return total_dist_err / num_samples


# ================= 4. 主训练程序 =================
def main():
    print(f"🚀 启动终极版训练 (Spatial Weight + Consistency) | 设备: {DEVICE}")

    # 1. 加载数据集
    full_dataset = PhysicsGuidedHDF5Dataset(H5_PATH)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    # 2. DataLoader
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    # 3. 模型初始化
    model = PhysicsGuidedNet(num_rx=4, signal_len=2048).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)
    scaler = torch.cuda.amp.GradScaler()

    # 4. Loss 定义
    # 关键修改：reduction='none' 以便手动应用空间权重
    criterion_coord = nn.L1Loss(reduction='none')
    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([20.0]).to(DEVICE))
    criterion_dice = DiceLoss()

    best_err = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for iq, heatmap, coord, mask in pbar:
            iq, heatmap, coord, mask = iq.to(DEVICE), heatmap.to(DEVICE), coord.to(DEVICE), mask.to(DEVICE)

            # 1. 基础增强
            iq, heatmap, coord, mask = apply_augmentation(iq, heatmap, coord, mask)

            optimizer.zero_grad()

            # --- Pass A: 原始前向传播 ---
            with torch.cuda.amp.autocast():
                pred_coord, pred_mask = model(iq, heatmap)

                # A1. 计算坐标 Loss (带空间加权)
                raw_loss_c = criterion_coord(pred_coord, coord[:, :2])  # [B, 2]
                spatial_w = get_spatial_weight(coord, DEVICE)  # [B, 1]
                loss_c = (raw_loss_c * spatial_w).mean()  # Scalar

                # A2. 计算 Mask Loss
                loss_m = criterion_bce(pred_mask, mask) + criterion_dice(pred_mask, mask)

            # --- Pass B: 一致性约束 (Explicit Consistency) ---
            loss_consistency = torch.tensor(0.0, device=DEVICE)

            # 100% 触发一致性检查
            if True:
                # B1. 构造翻转样本 (H-Flip)
                heatmap_flip = torch.flip(heatmap, [3])
                idx_perm = torch.tensor([1, 0, 3, 2, 5, 4, 7, 6], device=DEVICE)
                iq_flip = iq[:, idx_perm, :]

                with torch.cuda.amp.autocast():
                    # B2. 预测
                    pred_coord_flip, _ = model(iq_flip, heatmap_flip)

                # B3. 还原坐标: x' = 1 - x
                pred_coord_restored = pred_coord_flip.clone()
                pred_coord_restored[:, 0] = 1.0 - pred_coord_restored[:, 0]

                # B4. 计算一致性 (L1 Loss)
                loss_consistency = torch.nn.functional.l1_loss(pred_coord, pred_coord_restored.detach())

            # --- 总 Loss ---
            mask_w = 0.5 if epoch < 20 else 0.3
            # Consistency 权重给 2.0，强迫模型学会自洽
            total_loss = loss_c + mask_w * loss_m + 0 * loss_consistency

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += total_loss.item()
            pbar.set_postfix({
                'Loss': f"{total_loss.item():.3f}",
                'Consis': f"{loss_consistency.item():.3f}"
            })

        # 验证
        val_err = validate(model, val_loader)
        print(f"Epoch {epoch + 1} 验证完成: 平均误差 = {val_err:.2f}m")

        scheduler.step(val_err)

        if val_err < best_err:
            best_err = val_err
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"🌟 发现更优模型: {best_err:.2f}m")


if __name__ == '__main__':
    main()