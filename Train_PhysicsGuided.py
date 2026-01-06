import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import h5py
import numpy as np
import os
from PhysicsGuidedNetwork import PhysicsGuidedNet

# ================= 路径配置 =================
H5_PATH = r"D:\Dataset\SignalDataset\merged_dataset_512_3d_fast.h5"

# ================= 超参数配置 =================
BATCH_SIZE = 32
CHUNK_SIZE = 2000
LR = 1e-4
EPOCHS = 50
SCENE_SIZE = 5000.0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ================= 1. 物理修正版数据增强 (Core Fix) =================
def apply_augmentation(iq, heatmap, coord, mask):
    """
    对 Batch 数据进行随机旋转和翻转 (GPU加速)
    必须同步交换 IQ 通道，以保持物理一致性！

    接收机布局假设 (基于 MakeCsvIQData):
    Rx0:(0,0), Rx1:(5000,0), Rx2:(5000,5000), Rx3:(0,5000)

    IQ 数据结构 (基于 Generate_Multimodal_Data):
    [B, 8, L] -> [Rx0_R, Rx1_R, Rx2_R, Rx3_R, Rx0_I, Rx1_I, Rx2_I, Rx3_I]
    """

    # --- 1. 随机水平翻转 (H-Flip) ---
    # 几何意义：左右互换 -> Rx0<->Rx1, Rx3<->Rx2
    if np.random.rand() > 0.5:
        # A. 图片与标签翻转
        heatmap = torch.flip(heatmap, [3])  # Width is dim 3
        mask = torch.flip(mask, [3])
        coord[:, 0] = 1.0 - coord[:, 0]  # x = 1-x

        # B. IQ 通道交换 (关键修正!)
        # 实部交换: 0<->1, 3<->2
        # 虚部交换: 4<->5, 7<->6
        # 原始索引: [0, 1, 2, 3, 4, 5, 6, 7]
        # 目标索引: [1, 0, 3, 2, 5, 4, 7, 6]
        idx_perm = torch.tensor([1, 0, 3, 2, 5, 4, 7, 6], device=iq.device)
        iq = iq[:, idx_perm, :]

    # --- 2. 随机垂直翻转 (V-Flip) ---
    # 几何意义：上下互换 -> Rx0<->Rx3, Rx1<->Rx2
    if np.random.rand() > 0.5:
        # A. 图片与标签翻转
        heatmap = torch.flip(heatmap, [2])  # Height is dim 2
        mask = torch.flip(mask, [2])
        coord[:, 1] = 1.0 - coord[:, 1]  # y = 1-y

        # B. IQ 通道交换 (关键修正!)
        # 实部交换: 0<->3, 1<->2
        # 虚部交换: 4<->7, 5<->6
        # 原始索引: [0, 1, 2, 3, 4, 5, 6, 7]
        # 目标索引: [3, 2, 1, 0, 7, 6, 5, 4]
        idx_perm = torch.tensor([3, 2, 1, 0, 7, 6, 5, 4], device=iq.device)
        iq = iq[:, idx_perm, :]

    # (可选) 旋转 90度 也可以加了，因为 Rx 是正方形对称的
    # 逆时针90度: (x,y)->(-y,x)。Rx0->Rx1->Rx2->Rx3->Rx0
    # 对应 IQ 通道循环移位即可。为了稳妥，先只用 Flip 试试效果。

    return iq, heatmap, coord, mask


# ================= 2. Dice Loss (保持) =================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred_logits, target):
        pred_probs = torch.sigmoid(pred_logits)
        pred_flat = pred_probs.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        return 1 - dice


# ================= 验证函数 =================
def validate(model, val_indices, h5_file, criterion_coord, criterion_bce, criterion_dice, chunk_size=1000,
             batch_size=32):
    model.eval()
    total_loss = 0.0
    total_dist_err = 0.0
    num_samples = 0

    with torch.no_grad():
        for i in range(0, len(val_indices), chunk_size):
            current_indices = val_indices[i: i + chunk_size]
            current_indices = np.sort(current_indices)

            iq_ram = torch.from_numpy(h5_file['iq'][current_indices]).float()
            heatmap_ram = torch.from_numpy(h5_file['heatmap'][current_indices]).float()
            mask_ram = torch.from_numpy(h5_file['mask'][current_indices]).float()
            coord_ram = torch.from_numpy(h5_file['coord'][current_indices]).float()

            temp_dataset = TensorDataset(iq_ram, heatmap_ram, coord_ram, mask_ram)
            temp_loader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

            for iq, heatmap, true_coord, mask in temp_loader:
                iq, heatmap = iq.to(DEVICE), heatmap.to(DEVICE)
                mask, true_coord = mask.to(DEVICE), true_coord.to(DEVICE)

                pred_coord, pred_mask = model(iq, heatmap)

                true_coord_xy = true_coord[:, :2]
                loss_c = criterion_coord(pred_coord, true_coord_xy)

                loss_b = criterion_bce(pred_mask, mask)
                loss_d = criterion_dice(pred_mask, mask)
                loss_total = loss_c + 0.5 * (loss_b + loss_d)

                batch_len = iq.size(0)
                total_loss += loss_total.item() * batch_len

                dist_meter = torch.norm(pred_coord - true_coord_xy, dim=1) * SCENE_SIZE
                total_dist_err += dist_meter.sum().item()

                num_samples += batch_len

            del iq_ram, heatmap_ram, mask_ram, coord_ram, temp_dataset, temp_loader

    return total_loss / num_samples, total_dist_err / num_samples


# ================= 主程序 =================
def main():
    print(f"🚀 启动物理修正增强版训练 (Symmetric Rx Augmentation) | Device: {DEVICE}")

    if not os.path.exists(H5_PATH):
        print(f"【错误】找不到数据集文件: {H5_PATH}")
        return

    f = h5py.File(H5_PATH, 'r')
    total_samples = len(f['iq'])

    # 划分数据集
    all_indices = np.arange(total_samples)
    split_idx = int(0.9 * total_samples)
    train_indices_all = all_indices[:split_idx]
    val_indices_all = all_indices[split_idx:]

    # 初始化模型
    sample_iq = f['iq'][0]
    num_rx = sample_iq.shape[0] // 2
    model = PhysicsGuidedNet(num_rx=num_rx, signal_len=2048).to(DEVICE)

    # 优化器 & 调度器
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )

    # Loss 定义
    criterion_coord = nn.MSELoss()
    pos_weight = torch.tensor([20.0]).to(DEVICE)
    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_dice = DiceLoss()

    best_err = float('inf')

    try:
        for epoch in range(EPOCHS):
            model.train()
            train_loss_epoch = 0.0
            pbar = tqdm(total=len(train_indices_all), desc=f"Epoch {epoch + 1}/{EPOCHS}")

            for chunk_start in range(0, len(train_indices_all), CHUNK_SIZE):
                chunk_end = min(chunk_start + CHUNK_SIZE, len(train_indices_all))

                # Load to RAM
                iq_ram = torch.from_numpy(f['iq'][chunk_start:chunk_end])
                map_ram = torch.from_numpy(f['heatmap'][chunk_start:chunk_end])
                mask_ram = torch.from_numpy(f['mask'][chunk_start:chunk_end])
                coord_ram = torch.from_numpy(f['coord'][chunk_start:chunk_end])

                mem_dataset = TensorDataset(iq_ram, map_ram, coord_ram, mask_ram)
                train_loader = DataLoader(mem_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

                for iq, heatmap, true_coord, mask in train_loader:
                    iq, heatmap = iq.to(DEVICE), heatmap.to(DEVICE)
                    mask, true_coord = mask.to(DEVICE), true_coord.to(DEVICE)

                    # ================= 应用修正后的增强 =================
                    iq, heatmap, true_coord, mask = apply_augmentation(iq, heatmap, true_coord, mask)
                    # =================================================

                    optimizer.zero_grad()
                    pred_coord, pred_mask = model(iq, heatmap)

                    true_coord_xy = true_coord[:, :2]

                    # Loss 计算
                    loss_c = criterion_coord(pred_coord, true_coord_xy)
                    loss_b = criterion_bce(pred_mask, mask)
                    loss_d = criterion_dice(pred_mask, mask)

                    loss_m = loss_b + loss_d

                    # 动态权重
                    mask_weight = 0.5 if epoch < 20 else 0.1
                    loss = loss_c + mask_weight * loss_m

                    loss.backward()
                    optimizer.step()

                    train_loss_epoch += loss.item() * iq.size(0)

                pbar.update(chunk_end - chunk_start)
                pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
                del iq_ram, map_ram, mask_ram, coord_ram, mem_dataset, train_loader

            pbar.close()

            print("正在验证...")
            avg_val_loss, avg_dist_err = validate(
                model, val_indices_all, f,
                criterion_coord, criterion_bce, criterion_dice,
                chunk_size=CHUNK_SIZE, batch_size=BATCH_SIZE
            )

            avg_train_loss = train_loss_epoch / len(train_indices_all)
            print(
                f"Epoch {epoch + 1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Err={avg_dist_err:.2f}m")

            scheduler.step(avg_dist_err)

            if avg_dist_err < best_err:
                best_err = avg_dist_err
                torch.save(model.state_dict(), "best_model_symmetric.pth")
                print(f">>> 新最优模型 (Symmetric): {best_err:.2f}m")

    except KeyboardInterrupt:
        print("\n训练中断。")
    finally:
        f.close()


if __name__ == '__main__':
    main()