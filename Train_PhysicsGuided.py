import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import h5py
import numpy as np
import sys
import os

# 引入你的网络定义
sys.path.append('DataLoader')
# 假设 PhysicsGuidedNetwork.py 和 Train 脚本在同一级或能被 python path 找到
try:
    from PhysicsGuidedNetwork import PhysicsGuidedNet
except ImportError:
    # 尝试直接从当前目录导入
    from PhysicsGuidedNetwork import PhysicsGuidedNet

# ================= 配置区域 =================
H5_PATH = r"D:\Dataset\SignalDataset\merged_dataset_512_3d_fast.h5"
BATCH_SIZE = 32
CHUNK_SIZE = 4000  # 【关键】每次读入内存的样本数。2000个样本约占 4GB 内存。根据你的内存大小调整。
LR = 1e-4
EPOCHS = 50
SCENE_SIZE = 5000.0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def validate(model, val_indices, h5_file, criterion_coord, criterion_mask, chunk_size=1000, batch_size=32):
    """
    修复后的验证函数：加载大块数据后，使用 DataLoader 分小批次验证，防止爆显存
    """
    model.eval()
    total_loss = 0.0
    total_dist_err = 0.0
    num_samples = 0

    # 验证集分块加载 (HDD -> RAM)
    with torch.no_grad():
        for i in range(0, len(val_indices), chunk_size):
            # 1. 获取当前块的索引
            current_indices = val_indices[i: i + chunk_size]
            current_indices = np.sort(current_indices)  # HDF5 要求升序

            # 2. 加载到 CPU 内存 (RAM)
            # 注意：不要在这里直接 .to(DEVICE)，否则 1000 条数据会占满显存
            iq_ram = torch.from_numpy(h5_file['iq'][current_indices]).float()
            heatmap_ram = torch.from_numpy(h5_file['heatmap'][current_indices]).float()
            mask_ram = torch.from_numpy(h5_file['mask'][current_indices]).float()
            coord_ram = torch.from_numpy(h5_file['coord'][current_indices]).float()

            # 3. 创建临时 DataLoader (RAM -> GPU Mini-batch)
            # 这样每次只喂 32 条给 GPU
            temp_dataset = TensorDataset(iq_ram, heatmap_ram, coord_ram, mask_ram)
            temp_loader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

            # 4. 小批次推理
            for iq, heatmap, true_coord, mask in temp_loader:
                iq, heatmap = iq.to(DEVICE), heatmap.to(DEVICE)
                mask, true_coord = mask.to(DEVICE), true_coord.to(DEVICE)

                # 预测
                pred_coord, pred_mask = model(iq, heatmap)

                # Loss
                true_coord_xy = true_coord[:, :2]
                loss_c = criterion_coord(pred_coord, true_coord_xy)
                loss_m = criterion_mask(pred_mask, mask)

                # 累加误差 (乘以当前 batch 大小)
                batch_len = iq.size(0)
                total_loss += (loss_c + 0.5 * loss_m).item() * batch_len

                dist_meter = torch.norm(pred_coord - true_coord_xy, dim=1) * SCENE_SIZE
                total_dist_err += dist_meter.sum().item()

                num_samples += batch_len

            # 释放 RAM
            del iq_ram, heatmap_ram, mask_ram, coord_ram, temp_dataset, temp_loader

    return total_loss / num_samples, total_dist_err / num_samples


def main():
    print(f"🚀 启动分块训练 | Chunk Size: {CHUNK_SIZE}")

    if not os.path.exists(H5_PATH):
        print("找不到数据集文件！")
        return

    # 1. 打开 HDF5 (只读取元数据，不读内容)
    f = h5py.File(H5_PATH, 'r')
    total_samples = len(f['iq'])
    print(f"总样本数: {total_samples}")

    # 2. 划分训练/验证集 (索引划分)
    all_indices = np.arange(total_samples)
    # 不打乱总索引，直接按前90%后10%切分，保证训练集在硬盘上是连续的，读取最快
    split_idx = int(0.9 * total_samples)
    train_indices_all = all_indices[:split_idx]
    val_indices_all = all_indices[split_idx:]

    print(f"训练集: {len(train_indices_all)}, 验证集: {len(val_indices_all)}")

    # 3. 初始化模型
    sample_iq = f['iq'][0]
    num_rx = sample_iq.shape[0] // 2
    model = PhysicsGuidedNet(num_rx=num_rx, signal_len=2048).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    criterion_coord = nn.MSELoss()
    criterion_mask = nn.MSELoss()

    best_err = float('inf')

    # ================= 训练循环 =================
    for epoch in range(EPOCHS):
        model.train()
        train_loss_epoch = 0.0
        processed_samples = 0

        # 进度条
        pbar = tqdm(total=len(train_indices_all), desc=f"Epoch {epoch + 1}/{EPOCHS}")

        # --- 分块加载循环 (Chunk Loading) ---
        # 每次只处理 train_indices_all 中的一部分
        # 为了保证 I/O 最快，我们按顺序切片读取

        for chunk_start in range(0, len(train_indices_all), CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, len(train_indices_all))

            # A. 【加载阶段】从硬盘读入内存
            # 使用切片 f['key'][start:end] 是最快的顺序读取方式
            # 注意：这里的索引是相对于 HDF5 文件的绝对索引
            # 因为我们在上面是按顺序划分的，所以可以直接切片

            # print(f"  Loading chunk {chunk_start}-{chunk_end} to RAM...")
            iq_ram = torch.from_numpy(f['iq'][chunk_start:chunk_end])
            map_ram = torch.from_numpy(f['heatmap'][chunk_start:chunk_end])
            mask_ram = torch.from_numpy(f['mask'][chunk_start:chunk_end])
            coord_ram = torch.from_numpy(f['coord'][chunk_start:chunk_end])

            # B. 【构造内存 DataLoader】
            # 数据已经在内存里了，TensorDataset 包装一下
            # num_workers=0, 因为内存读取不需要多进程，多进程反而慢
            mem_dataset = TensorDataset(iq_ram, map_ram, coord_ram, mask_ram)
            train_loader = DataLoader(mem_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

            # C. 【GPU 训练阶段】
            for iq, heatmap, true_coord, mask in train_loader:
                iq, heatmap, mask, true_coord = iq.to(DEVICE), heatmap.to(DEVICE), mask.to(DEVICE), true_coord.to(
                    DEVICE)

                optimizer.zero_grad()
                pred_coord, pred_mask = model(iq, heatmap)

                true_coord_xy = true_coord[:, :2]
                loss_c = criterion_coord(pred_coord, true_coord_xy)
                loss_m = criterion_mask(pred_mask, mask)
                loss = loss_c + 0.5 * loss_m

                loss.backward()
                optimizer.step()

                train_loss_epoch += loss.item() * iq.size(0)
                processed_samples += iq.size(0)

            # 更新总进度条
            pbar.update(chunk_end - chunk_start)
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

            # D. 【释放内存】
            # 进入下一次循环前，iq_ram 等变量会被覆盖或销毁，Python GC 会自动回收
            del iq_ram, map_ram, mask_ram, coord_ram, mem_dataset, train_loader

        pbar.close()

        # --- 验证阶段 ---
        print("Validating...")
        # 修改调用方式，传入 batch_size
        avg_val_loss, avg_dist_err = validate(
            model,
            val_indices_all,
            f,
            criterion_coord,
            criterion_mask,
            chunk_size=CHUNK_SIZE,  # 使用和训练一样的 Chunk 大小读取硬盘
            batch_size=BATCH_SIZE  # 使用和训练一样的 Batch 大小进行推理
        )

        avg_train_loss = train_loss_epoch / len(train_indices_all)
        print(
            f"Epoch {epoch + 1} Result: Train Loss={avg_train_loss:.5f}, Val Loss={avg_val_loss:.5f}, Err={avg_dist_err:.2f}m")

        if avg_dist_err < best_err:
            best_err = avg_dist_err
            torch.save(model.state_dict(), "best_model_chunked.pth")
            print(">>> Model Saved!")

    f.close()


if __name__ == '__main__':
    main()