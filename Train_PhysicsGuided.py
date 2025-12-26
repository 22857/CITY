import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

# === 1. 环境与路径配置 ===
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 动态添加项目根目录，确保能找到 NetworkFunction
current_dir = os.path.dirname(os.path.abspath(__file__))  # .../MultiEmitters
project_root = os.path.dirname(current_dir)  # .../DPV-main
if project_root not in sys.path:
    sys.path.append(project_root)

# 尝试导入网络架构
try:
    from NetworkFunction.PhysicsGuidedNetwork import PhysicsGuidedNet

    print("成功导入 PhysicsGuidedNet")
except ImportError as e:
    print(f"【错误】无法导入网络模型: {e}")
    sys.exit(1)

# ================= 2. 全局配置 (Config) =================
CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    # --- 路径配置 (请确保指向 Step 1.5 生成的 H5 文件) ---
    'h5_path': r"D:\Dataset\SignalDataset\merged_dataset_512.h5",

    'save_dir': './checkpoints_hdf5',
    'vis_dir': './training_vis_hdf5',

    # --- 训练参数 (针对 RTX 4060 优化) ---
    'epochs': 50,
    'batch_size': 32,  # 【加速】加大 Batch Size 跑满 GPU
    'learning_rate': 3e-4,  # 【稳定】降低 LR 防止震荡
    'num_workers': 4,  # 【并行】HDF5 支持多进程读取

    # --- 物理参数 ---
    'scene_size': 5000.0,
    'map_size': 512,
    'signal_len': 2048,
    'num_rx': 4,

    # --- 损失权重 (策略调整) ---
    'w_coord': 500.0,  # 【策略】强力优化坐标精度
    'w_mask': 1.0,  # 辅助优化
}


# ================= 3. HDF5 数据集类 =================
class PhysicsGuidedHDF5Dataset(Dataset):
    def __init__(self, h5_path):
        """
        专用于读取合并后的 HDF5 文件
        """
        self.h5_path = h5_path
        self.dataset_len = 0
        self.h5_file = None

        # 预检文件
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"找不到 HDF5 文件: {h5_path}\n请先运行合并脚本！")

        # 打开一次获取长度
        with h5py.File(self.h5_path, 'r') as f:
            self.dataset_len = len(f['coord'])

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        # 【关键】懒加载：确保每个 Worker 进程有自己的文件句柄
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')

        # HDF5 切片读取 (自动处理 float16 -> numpy)
        iq_np = self.h5_file['iq'][idx]  # [8, 2048]
        map_np = self.h5_file['heatmap'][idx]  # [1, 64, 64]
        mask_np = self.h5_file['mask'][idx]  # [1, 64, 64]
        coord_np = self.h5_file['coord'][idx]  # [2]

        # 转为 Tensor (必须转 float32 才能进入网络计算)
        return (
            torch.from_numpy(iq_np).float(),
            torch.from_numpy(map_np).float(),
            torch.from_numpy(coord_np).float(),
            torch.from_numpy(mask_np).float()
        )


# ================= 4. 可视化工具 =================
def save_visualization(iq, rough_map, true_mask, pred_mask, true_coord, pred_coord, epoch, save_dir):
    rough_map = rough_map.squeeze().cpu().detach().numpy()
    true_mask = true_mask.squeeze().cpu().detach().numpy()
    pred_mask = pred_mask.squeeze().cpu().detach().numpy()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title(f"Input Heatmap (Pre-calc)\nTrue: {true_coord.cpu().numpy()}")
    plt.imshow(rough_map, origin='lower', cmap='jet')

    plt.subplot(1, 3, 2)
    plt.title(f"Pred Mask (Reconstructed)\nPred: {pred_coord.cpu().numpy()}")
    plt.imshow(pred_mask, origin='lower', cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title("Ground Truth Mask")
    plt.imshow(true_mask, origin='lower', cmap='gray')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"epoch_{epoch}.png"))
    plt.close()


# ================= 5. 主训练流程 =================
def main():
    # 准备目录
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    os.makedirs(CONFIG['vis_dir'], exist_ok=True)
    print(f"启动训练 | 设备: {CONFIG['device']} | Batch: {CONFIG['batch_size']}")

    # --- A. 加载数据 ---
    print(f"打开数据集: {CONFIG['h5_path']}")
    full_ds = PhysicsGuidedHDF5Dataset(CONFIG['h5_path'])

    # 划分数据集 (90% Train, 10% Valid)
    total_len = len(full_ds)
    train_len = int(0.9 * total_len)
    val_len = total_len - train_len
    train_ds, val_ds = random_split(full_ds, [train_len, val_len])

    print(f"训练样本: {train_len}, 验证样本: {val_len}")

    # DataLoader (开启 persistent_workers 优化 HDF5 读取)
    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=True,
        persistent_workers=(CONFIG['num_workers'] > 0)
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True,
        persistent_workers=(CONFIG['num_workers'] > 0)
    )

    # --- B. 初始化模型 ---
    model = PhysicsGuidedNet(
        num_rx=CONFIG['num_rx'],
        signal_len=CONFIG['signal_len'],
        map_size=CONFIG['map_size']
    ).to(CONFIG['device'])

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-4)

    # 【修复】移除了 verbose=True 以兼容新版 PyTorch
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    criterion_coord = nn.MSELoss()
    criterion_mask = nn.BCELoss()

    # --- C. 训练循环 ---
    best_dist_error = float('inf')

    for epoch in range(CONFIG['epochs']):
        # ====== TRAIN ======
        model.train()
        train_loss_log = []

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']} [Train]")

        for batch in loop:
            # 数据上 GPU
            iq, r_map, true_coord, true_mask = [item.to(CONFIG['device']) for item in batch]

            # 前向
            pred_coord, pred_mask = model(iq, r_map)

            # 损失计算
            loss_c = criterion_coord(pred_coord, true_coord)
            loss_m = criterion_mask(pred_mask, true_mask)
            loss = CONFIG['w_coord'] * loss_c + CONFIG['w_mask'] * loss_m

            # 反向
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_log.append(loss.item())
            loop.set_postfix(loss=loss.item())

        # ====== VALID ======
        model.eval()
        val_loss_total = 0.0
        dist_errors = []

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                iq, r_map, true_coord, true_mask = [item.to(CONFIG['device']) for item in batch]

                pred_coord, pred_mask = model(iq, r_map)

                # 计算 Loss
                loss_c = criterion_coord(pred_coord, true_coord)
                loss_m = criterion_mask(pred_mask, true_mask)
                val_loss_total += (CONFIG['w_coord'] * loss_c + CONFIG['w_mask'] * loss_m).item()

                # 计算物理误差 (米)
                diff_m = (pred_coord - true_coord) * CONFIG['scene_size']
                dist_err = torch.norm(diff_m, dim=1).cpu().numpy()
                dist_errors.extend(dist_err)

                # 保存第一张图用于观察
                if i == 0:
                    save_visualization(
                        iq[0], r_map[0], true_mask[0], pred_mask[0],
                        true_coord[0] * CONFIG['scene_size'],
                        pred_coord[0] * CONFIG['scene_size'],
                        epoch + 1, CONFIG['vis_dir']
                    )

        # ====== 统计 ======
        avg_train_loss = np.mean(train_loss_log)
        avg_val_loss = val_loss_total / len(val_loader)
        avg_dist_error = np.mean(dist_errors)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\n[Epoch {epoch + 1} Summary]")
        print(f"  LR: {current_lr:.2e}")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"  > Mean Error: {avg_dist_error:.2f} meters")

        # 调度器步进
        scheduler.step(avg_val_loss)

        # 保存最优
        if avg_dist_error < best_dist_error:
            best_dist_error = avg_dist_error
            torch.save(model.state_dict(), os.path.join(CONFIG['save_dir'], 'best_model.pth'))
            print(f"New Best Model Saved! ({best_dist_error:.2f}m)")

        torch.save(model.state_dict(), os.path.join(CONFIG['save_dir'], 'last_model.pth'))


if __name__ == '__main__':
    main()