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
H5_PATH = "/root/autodl-tmp/merged_dataset_512_3d_fast_v2.h5"  # 确保路径与生成脚本一致
SAVE_PATH = "best_model_symmetric.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================= 2. 超参数配置 =================
BATCH_SIZE = 64  # 优化 H5 后可尝试增大至 64 或 128
NUM_WORKERS = 8  # AutoDL 建议设为 8-16
LR = 1e-4
EPOCHS = 50
SCENE_SIZE = 5000.0


# ================= 3. 物理一致性增强函数 (保持) =================
def apply_augmentation(iq, heatmap, coord, mask):
    """
    在 GPU 上进行数据增强，保持 IQ 通道与几何翻转的一致性
    """
    # 随机水平翻转
    if np.random.rand() > 0.5:
        heatmap = torch.flip(heatmap, [3])
        mask = torch.flip(mask, [3])
        coord[:, 0] = 1.0 - coord[:, 0]
        # H-Flip 索引交换: Rx0<->Rx1, Rx3<->Rx2
        idx_perm = torch.tensor([1, 0, 3, 2, 5, 4, 7, 6], device=iq.device)
        iq = iq[:, idx_perm, :]

    # 随机垂直翻转
    if np.random.rand() > 0.5:
        heatmap = torch.flip(heatmap, [2])
        mask = torch.flip(mask, [2])
        coord[:, 1] = 1.0 - coord[:, 1]
        # V-Flip 索引交换: Rx0<->Rx3, Rx1<->Rx2
        idx_perm = torch.tensor([3, 2, 1, 0, 7, 6, 5, 4], device=iq.device)
        iq = iq[:, idx_perm, :]

    return iq, heatmap, coord, mask


# ================= 4. Loss 定义 =================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred_logits, target):
        pred_probs = torch.sigmoid(pred_logits)
        intersection = (pred_probs * target).sum()
        dice = (2. * intersection + self.smooth) / (pred_probs.sum() + target.sum() + self.smooth)
        return 1 - dice


# ================= 5. 验证函数 (优化版) =================
def validate(model, loader, criterion_coord, criterion_bce, criterion_dice):
    model.eval()
    total_dist_err = 0.0
    num_samples = 0

    with torch.no_grad():
        for iq, heatmap, coord, mask in loader:
            iq, heatmap, coord, mask = iq.to(DEVICE), heatmap.to(DEVICE), coord.to(DEVICE), mask.to(DEVICE)

            # 混合精度推理
            with torch.cuda.amp.autocast():
                pred_coord, _ = model(iq, heatmap)

            dist_err = torch.norm(pred_coord - coord[:, :2], dim=1) * SCENE_SIZE
            total_dist_err += dist_err.sum().item()
            num_samples += iq.size(0)

    return total_dist_err / num_samples


# ================= 6. 主训练程序 =================
def main():
    print(f"🚀 启动极速版训练 | 设备: {DEVICE} | Workers: {NUM_WORKERS}")

    # 1. 加载数据集
    full_dataset = PhysicsGuidedHDF5Dataset(H5_PATH)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    # 2. 创建 DataLoader (开启预取加速)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    # 3. 初始化模型与优化器
    model = PhysicsGuidedNet(num_rx=4, signal_len=2048).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)

    # 混合精度缩放器
    scaler = torch.cuda.amp.GradScaler()

    # 4. Loss 定义
    criterion_coord = nn.L1Loss()
    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([20.0]).to(DEVICE))
    criterion_dice = DiceLoss()

    best_err = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for iq, heatmap, coord, mask in pbar:
            iq, heatmap, coord, mask = iq.to(DEVICE), heatmap.to(DEVICE), coord.to(DEVICE), mask.to(DEVICE)

            # 应用数据增强
            iq, heatmap, coord, mask = apply_augmentation(iq, heatmap, coord, mask)

            optimizer.zero_grad()

            # --- 开启混合精度训练 ---
            with torch.cuda.amp.autocast():
                pred_coord, pred_mask = model(iq, heatmap)

                loss_c = criterion_coord(pred_coord, coord[:, :2])
                loss_m = criterion_bce(pred_mask, mask) + criterion_dice(pred_mask, mask)

                # 动态调整 Mask 权重
                mask_w = 0.5 if epoch < 20 else 0.1
                total_loss = loss_c + mask_w * loss_m

            # 反向传播缩放
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += total_loss.item()
            pbar.set_postfix({'loss': f"{total_loss.item():.4f}"})

        # 验证
        val_err = validate(model, val_loader, criterion_coord, criterion_bce, criterion_dice)
        print(f"Epoch {epoch + 1} 验证完成: 平均误差 = {val_err:.2f}m")

        scheduler.step(val_err)

        if val_err < best_err:
            best_err = val_err
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"🌟 发现更优模型: {best_err:.2f}m")


if __name__ == '__main__':
    main()