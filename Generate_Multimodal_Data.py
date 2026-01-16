import os
import sys
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ================= 配置区域 (针对 AutoDL 优化) =================
DATASET_ROOT = "/root/autodl-tmp/SignalDataset_6Rx_SafeZone/SNR[6, 6]/valid"
OUTPUT_H5_PATH = "/root/autodl-tmp/merged_dataset_snr6_valid.h5"

# 物理参数
SCENE_SIZE = 5000.0
MAP_SIZE = 512
SIGNAL_LEN = 2048
FS = 50e6
C = 299792458.0

# --- 核心性能优化参数 ---
BATCH_SIZE = 32         # 减小 Batch Size 释放显存
NUM_WORKERS = 8
GRID_CHUNK_SIZE = 2048  # 减小 Chunk Size，防止 64GB 溢出
PREFETCH_FACTOR = 2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ================= 数据加载器 (保持原有逻辑) =================
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'DataLoader'))
try:
    from ProcessedSampleDataset import ProcessedSampleDataset
except ImportError:
    print("【错误】找不到 ProcessedSampleDataset。")
    sys.exit(1)


class SignalRawDataset(Dataset):
    def __init__(self, root_path, signal_len=2048):
        self.ds = ProcessedSampleDataset(root_path, dataType='IQ')
        self.signal_len = signal_len
        self.total_num = self.ds.getDataNum()

    def __len__(self):
        return self.total_num

    def __getitem__(self, idx):
        (iq, rx_info), tx_info = self.ds.getData(idx, isGetFsFc=False)
        if iq.shape[1] > self.signal_len:
            iq = iq[:, :self.signal_len]
        elif iq.shape[1] < self.signal_len:
            iq = np.pad(iq, ((0, 0), (0, self.signal_len - iq.shape[1])))
        N_rx = iq.shape[0]
        rx_pos = rx_info[:N_rx * 6].reshape(N_rx, 6)[:, :3]
        tx_pos = tx_info[:3]
        return iq.astype(np.complex64), rx_pos.astype(np.float32), tx_pos.astype(np.float32)


# ================= GPU 优化版: DPD 热力图 =================
def generate_dpd_map_batch_optimized(iq_batch, rx_pos_batch, map_size, scene_size, fs, device):
    B, N_rx, L = iq_batch.shape
    sig_tensor = iq_batch.to(device)
    rx_pos_tensor = rx_pos_batch.to(device)

    # 1. 频域预处理
    sig_freq = torch.fft.fftshift(torch.fft.fft(sig_tensor, dim=2), dim=2)
    # 形状: [B, 1, N_rx, L]
    sig_freq = sig_freq.view(B, 1, N_rx, L)

    # 预计算频率轴
    freqs = torch.linspace(-fs / 2, fs / 2, L, device=device).view(1, 1, 1, L)
    two_pi_f = 2 * torch.pi * freqs

    # 2. 网格构建
    x = torch.linspace(0, scene_size, map_size, device=device)
    y = torch.linspace(0, scene_size, map_size, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    grid_points_3d = torch.stack([grid_x.flatten(), grid_y.flatten(), torch.zeros_like(grid_x.flatten())], dim=1)
    N_grid = grid_points_3d.shape[0]

    heatmaps = torch.zeros(B, N_grid, device=device)
    rx_exp_base = rx_pos_tensor.view(B, 1, N_rx, 3)

    # 3. 分块计算 (优化显存分配)
    for i in range(0, N_grid, GRID_CHUNK_SIZE):
        sub_grid = grid_points_3d[i: i + GRID_CHUNK_SIZE]
        sub_N = sub_grid.shape[0]

        # [B, Chunk, Rx]
        dists = torch.norm(sub_grid.view(1, sub_N, 1, 3) - rx_exp_base, dim=3)
        taus = dists / C

        # 核心显存消耗点：通过减小 GRID_CHUNK_SIZE 降至 ~4GB
        # 显存计算: B * sub_N * Rx * L * 8 bytes
        phase = torch.exp(1j * two_pi_f * taus.unsqueeze(-1))

        # V = sig * phase
        V = sig_freq * phase

        # Q = V @ V.mH (相关矩阵)
        Q = torch.matmul(V, V.mH)

        # 释放 phase 和 V 的显存 (显式删除可选，PyTorch 会自动回收)
        del phase, V

        # 取最大特征值
        max_eigs = torch.linalg.eigvalsh(Q)[..., -1].abs()
        heatmaps[:, i:i + sub_N] = max_eigs

        del Q

    # 4. 归一化
    heatmaps = heatmaps.view(B, map_size, map_size)
    h_min = heatmaps.amin(dim=(1, 2), keepdim=True)
    h_max = heatmaps.amax(dim=(1, 2), keepdim=True)
    heatmaps = (heatmaps - h_min) / (h_max - h_min + 1e-8)
    return heatmaps


# ================= GPU 优化版: Mask 生成 =================
def generate_ideal_mask_batch_optimized(rx_pos_batch, tx_pos_batch, map_size, scene_size, device):
    B, N_rx, _ = rx_pos_batch.shape
    rx_pos = rx_pos_batch.to(device)
    tx_pos = tx_pos_batch.to(device)

    x = torch.linspace(0, scene_size, map_size, device=device)
    y = torch.linspace(0, scene_size, map_size, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    grid_3d = torch.stack([grid_x, grid_y, torch.zeros_like(grid_x)], dim=-1).unsqueeze(0)

    masks = torch.zeros(B, map_size, map_size, device=device)
    threshold = (scene_size / map_size) * 1.5

    # 预计算所有 Tx 到 Rx 的真实距离差
    # 这里通过向量化进一步加速
    for i in range(N_rx):
        for j in range(i + 1, N_rx):
            d_true = torch.norm(tx_pos - rx_pos[:, i], dim=1) - \
                     torch.norm(tx_pos - rx_pos[:, j], dim=1)
            d_true = d_true.view(B, 1, 1)

            dist_grid_i = torch.norm(grid_3d - rx_pos[:, i].view(B, 1, 1, 3), dim=3)
            dist_grid_j = torch.norm(grid_3d - rx_pos[:, j].view(B, 1, 1, 3), dim=3)
            d_grid = dist_grid_i - dist_grid_j

            masks = torch.maximum(masks, ((d_grid - d_true).abs() < threshold).float())
    return masks


# ================= 主程序 (带 H5 分块优化) =================
if __name__ == '__main__':
    raw_dataset = SignalRawDataset(DATASET_ROOT, signal_len=SIGNAL_LEN)
    total_num = len(raw_dataset)
    if total_num == 0: sys.exit(1)

    dataloader = DataLoader(raw_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            prefetch_factor=PREFETCH_FACTOR)

    sample_iq, _, _ = raw_dataset[0]
    num_rx = sample_iq.shape[0]

    with h5py.File(OUTPUT_H5_PATH, 'w') as f:
        # 优化分块策略：1 样本 1 分块，极大提升训练读取速度
        dset_iq = f.create_dataset('iq', shape=(total_num, num_rx * 2, SIGNAL_LEN), dtype='float32')
        dset_map = f.create_dataset('heatmap', shape=(total_num, 1, MAP_SIZE, MAP_SIZE),
                                    dtype='float32', chunks=(1, 1, MAP_SIZE, MAP_SIZE), compression="lzf")
        dset_mask = f.create_dataset('mask', shape=(total_num, 1, MAP_SIZE, MAP_SIZE),
                                     dtype='float32', chunks=(1, 1, MAP_SIZE, MAP_SIZE), compression="lzf")
        dset_coord = f.create_dataset('coord', shape=(total_num, 3), dtype='float32')

        current_idx = 0
        for iq_batch, rx_batch, tx_batch in tqdm(dataloader, desc="Turbo Generating"):
            batch_len = iq_batch.shape[0]

            # 并行计算
            heatmaps = generate_dpd_map_batch_optimized(iq_batch, rx_batch, MAP_SIZE, SCENE_SIZE, FS, DEVICE)
            masks = generate_ideal_mask_batch_optimized(rx_batch, tx_batch, MAP_SIZE, SCENE_SIZE, DEVICE)

            # 整理并写入
            iq_real = torch.cat([iq_batch.real, iq_batch.imag], dim=1).numpy()
            end_idx = current_idx + batch_len

            dset_iq[current_idx:end_idx] = iq_real
            dset_map[current_idx:end_idx] = heatmaps.cpu().numpy()[:, np.newaxis, ...]
            dset_mask[current_idx:end_idx] = masks.cpu().numpy()[:, np.newaxis, ...]
            dset_coord[current_idx:end_idx] = (tx_batch / SCENE_SIZE).numpy()

            current_idx = end_idx
    print(f"\n✅ 完成！文件已保存至: {OUTPUT_H5_PATH}")