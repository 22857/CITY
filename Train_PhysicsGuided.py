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
        idx_perm = torch.tensor([1, 0, 3, 2, 5, 4, 7, 6], device=iq.device)
        iq = iq[:, idx_perm, :]

    # 随机垂直翻转 (V-Flip)
    if np.random.rand() > 0.5:
        heatmap = torch.flip(heatmap, [2])
        mask = torch.flip(mask, [2])
        coord[:, 1] = 1.0 - coord[:, 1]
        idx_perm = torch.tensor([3, 2, 1, 0, 7, 6, 5, 4], device=iq.device)
        iq = iq[:, idx_perm, :]

    return iq, heatmap, coord, mask


def get_spatial_weight(target_coord, device):
    """
    权重掩码：边缘区域权重为 0，中心区域权重为 1
    """
    x = target_coord[:, 0]
    y = target_coord[:, 1]
    MARGIN = 0.1  # 剔除边缘 10%

    # 也就是：x,y 都在 [0.1, 0.9] 之间时，weight=1，否则=0
    in_center = (x > MARGIN) & (x < 1.0 - MARGIN) & \
                (y > MARGIN) & (y < 1.0 - MARGIN)

    return in_center.float().unsqueeze(1).to(device)


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

    # 矩形切割验证：只统计中心区域
    MARGIN = 0.1

    with torch.no_grad():
        for iq, heatmap, coord, mask in loader:
            iq, heatmap, coord, mask = iq.to(DEVICE), heatmap.to(DEVICE), coord.to(DEVICE), mask.to(DEVICE)

            with torch.cuda.amp.autocast():
                pred_coord, _ = model(iq, heatmap)

            dist_err = torch.norm(pred_coord - coord[:, :2], dim=1) * SCENE_SIZE

            # 过滤逻辑
            x, y = coord[:, 0], coord[:, 1]
            valid_mask = (x > MARGIN) & (x < 1.0 - MARGIN) & \
                         (y > MARGIN) & (y < 1.0 - MARGIN)

            if valid_mask.sum() > 0:
                total_dist_err += dist_err[valid_mask].sum().item()
                num_samples += valid_mask.sum().item()

    if num_samples == 0: return 9999.0
    return total_dist_err / num_samples


# ================= 4. 主训练程序 =================
def main():
    print(f"🚀 启动终极版训练 (Safe Zone Only + Consistency) | 设备: {DEVICE}")

    # 1. 加载数据集
    full_dataset = PhysicsGuidedHDF5Dataset(H5_PATH)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    # 2. 模型初始化
    model = PhysicsGuidedNet(num_rx=4, signal_len=2048).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)
    scaler = torch.cuda.amp.GradScaler()

    # 3. Loss 定义
    criterion_coord = nn.L1Loss(reduction='none')
    # 【关键修改】保留 reduction='none'，以便下面手动处理标量化
    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([20.0]).to(DEVICE), reduction='none')
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

            with torch.cuda.amp.autocast():
                # --- Pass A: 原始前向传播 ---
                pred_coord, pred_mask = model(iq, heatmap)

                # 1. 计算空间权重 (中心为1, 边缘为0)
                spatial_w = get_spatial_weight(coord, DEVICE)  # [B, 1]
                num_valid = spatial_w.sum() + 1e-6

                # A1. 坐标 Loss (只计算有效区域)
                raw_loss_c = criterion_coord(pred_coord, coord[:, :2])  # [B, 2]
                loss_c = (raw_loss_c * spatial_w).sum() / num_valid

                # A2. Mask Loss (手动处理张量 -> 标量)
                # bce_map: [B, 1, 512, 512]
                bce_map = criterion_bce(pred_mask, mask)

                # 应用空间权重: 扩展维度以匹配 [B, 1, 1, 1]
                # 只有中心的样本计算 Loss，边缘样本 Loss 归零
                bce_masked = bce_map * spatial_w.view(-1, 1, 1, 1)

                # 求和并归一化 (除以有效像素总数)
                loss_bce = bce_masked.sum() / (num_valid * 512 * 512 + 1e-6)

                # Dice Loss (这个一般是全局标量，暂时不加权，影响不大)
                loss_dice = criterion_dice(pred_mask, mask)

                loss_m = loss_bce + loss_dice

            # --- Pass B: 一致性约束 (仅在安全区重启) ---
            loss_consistency = torch.tensor(0.0, device=DEVICE)

            if True:
                heatmap_flip = torch.flip(heatmap, [3])
                idx_perm = torch.tensor([1, 0, 3, 2, 5, 4, 7, 6], device=DEVICE)
                iq_flip = iq[:, idx_perm, :]

                with torch.cuda.amp.autocast():
                    pred_coord_flip, _ = model(iq_flip, heatmap_flip)

                pred_coord_restored = pred_coord_flip.clone()
                pred_coord_restored[:, 0] = 1.0 - pred_coord_restored[:, 0]

                # 计算一致性并应用空间权重
                raw_consis = torch.abs(pred_coord - pred_coord_restored.detach())  # [B, 2]
                loss_consistency = (raw_consis * spatial_w).sum() / num_valid

            # --- 总 Loss ---
            # 重启 Consistency (权重 10.0)
            mask_w = 0.2
            total_loss = loss_c + mask_w * loss_m + 10.0 * loss_consistency

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += total_loss.item()
            pbar.set_postfix({
                'Loss': f"{total_loss.item():.3f}",
                'Consis': f"{loss_consistency.item():.3f}"
            })

        val_err = validate(model, val_loader)
        print(f"Epoch {epoch + 1} 验证完成: 平均误差 = {val_err:.2f}m")

        scheduler.step(val_err)

        if val_err < best_err:
            best_err = val_err
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"🌟 发现更优模型: {best_err:.2f}m")


if __name__ == '__main__':
    main()