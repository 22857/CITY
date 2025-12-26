import os
import sys
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ================= 配置区域 =================
DATASET_ROOT = r"D:\Dataset\SignalDataset\SNR[5, 5]\train"
OUTPUT_H5_PATH = r"D:\Dataset\SignalDataset\merged_dataset_512_3d_fast.h5"

# 物理参数
SCENE_SIZE = 5000.0
MAP_SIZE = 512  # 512x512
SIGNAL_LEN = 2048
FS = 10e6
C = 299792458.0

# 性能参数
BATCH_SIZE = 32  # 适当增大，利用 GPU 并行能力
NUM_WORKERS = 4  # 读取数据的进程数 (根据你的 CPU 核心数调整，通常 4-8)
GRID_CHUNK_SIZE = 512  # DPD 计算分块大小

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ================= 数据加载器封装 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'DataLoader'))
try:
    from ProcessedSampleDataset import ProcessedSampleDataset
except ImportError:
    print("【错误】找不到 ProcessedSampleDataset，请检查路径。")
    sys.exit(1)


class SignalRawDataset(Dataset):
    """
    封装原始数据集，以便使用 PyTorch DataLoader 进行多进程加速读取
    """

    def __init__(self, root_path, signal_len=2048):
        self.ds = ProcessedSampleDataset(root_path, dataType='IQ')
        self.signal_len = signal_len
        self.total_num = self.ds.getDataNum()

    def __len__(self):
        return self.total_num

    def __getitem__(self, idx):
        # 读取数据
        (iq, rx_info), tx_info = self.ds.getData(idx, isGetFsFc=False)

        # 1. 处理 IQ 长度
        if iq.shape[1] > self.signal_len:
            iq = iq[:, :self.signal_len]
        elif iq.shape[1] < self.signal_len:
            iq = np.pad(iq, ((0, 0), (0, self.signal_len - iq.shape[1])))

        # 2. 解析坐标
        N_rx = iq.shape[0]
        rx_pos = rx_info[:N_rx * 6].reshape(N_rx, 6)[:, :3]
        tx_pos = tx_info[:3]

        # 返回 Tensor 兼容格式 (numpy)
        return iq.astype(np.complex64), rx_pos.astype(np.float32), tx_pos.astype(np.float32)


# ================= GPU 核心函数 1: DPD 热力图 =================
def generate_dpd_map_batch(iq_batch, rx_pos_batch, map_size, scene_size, fs, device):
    # iq_batch: [B, Rx, L]
    B, N_rx, L = iq_batch.shape

    # 预处理
    # 注意：DataLoader 出来的已经是 Tensor，但可能在 CPU，需移动到 GPU
    sig_tensor = iq_batch.to(device)
    rx_pos_tensor = rx_pos_batch.to(device)

    # FFT
    sig_freq = torch.fft.fftshift(torch.fft.fft(sig_tensor, dim=2), dim=2)
    sig_freq = sig_freq.view(B, 1, N_rx, L)

    freqs = torch.linspace(-fs / 2, fs / 2, L, device=device).view(1, 1, 1, L)

    # 构建网格
    x = torch.linspace(0, scene_size, map_size, device=device)
    y = torch.linspace(0, scene_size, map_size, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
    grid_points_2d = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

    # 3D 扩展 (Z=0)
    N_grid = grid_points_2d.shape[0]
    zeros = torch.zeros(N_grid, 1, device=device)
    grid_points_3d = torch.cat([grid_points_2d, zeros], dim=1)

    # 分块计算
    heatmap_chunks = []

    # 预先 reshape rx 避免循环内重复操作
    rx_exp_base = rx_pos_tensor.view(B, 1, N_rx, 3)

    for i in range(0, N_grid, GRID_CHUNK_SIZE):
        sub_grid = grid_points_3d[i: i + GRID_CHUNK_SIZE]
        sub_N = sub_grid.shape[0]

        # [1, Chunk, 1, 3] - [B, 1, Rx, 3] -> [B, Chunk, Rx, 3]
        dists = torch.norm(sub_grid.view(1, sub_N, 1, 3) - rx_exp_base, dim=3)
        taus = dists / C

        # Phase & Correlation
        phase = torch.exp(1j * 2 * torch.pi * freqs * taus.unsqueeze(-1))
        V = sig_freq * phase
        Q = torch.matmul(V, V.mH)

        # Max Eigenvalue
        max_eigs = torch.linalg.eigvalsh(Q)[..., -1].abs()
        heatmap_chunks.append(max_eigs)

    # 拼接 & 归一化
    full_flat = torch.cat(heatmap_chunks, dim=1)
    heatmaps = full_flat.view(B, map_size, map_size)
    h_min = heatmaps.amin(dim=(1, 2), keepdim=True)
    h_max = heatmaps.amax(dim=(1, 2), keepdim=True)
    heatmaps = (heatmaps - h_min) / (h_max - h_min + 1e-8)

    return heatmaps


# ================= GPU 核心函数 2: 极速 Mask 生成 =================
def generate_ideal_mask_batch_gpu(rx_pos_batch, tx_pos_batch, map_size, scene_size, device):
    """
    完全在 GPU 上并行的 Mask 生成函数
    """
    B, N_rx, _ = rx_pos_batch.shape
    rx_pos = rx_pos_batch.to(device)
    tx_pos = tx_pos_batch.to(device)

    # 1. 构建网格 [1, H, W, 3]
    x = torch.linspace(0, scene_size, map_size, device=device)
    y = torch.linspace(0, scene_size, map_size, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
    zeros = torch.zeros_like(grid_x)
    grid_3d = torch.stack([grid_x, grid_y, zeros], dim=-1).unsqueeze(0)  # [1, H, W, 3]

    # 2. 初始化 Mask
    masks = torch.zeros(B, map_size, map_size, device=device, dtype=torch.float32)

    pixel_res = scene_size / map_size
    threshold = pixel_res * 1.5

    # 3. 并行计算每一对接收机
    for i in range(N_rx):
        for j in range(i + 1, N_rx):
            # A. 真实距离差 TDOA [B]
            d_true = torch.norm(tx_pos - rx_pos[:, i], dim=1) - \
                     torch.norm(tx_pos - rx_pos[:, j], dim=1)
            # 扩展为 [B, 1, 1] 以便广播
            d_true = d_true.view(B, 1, 1)

            # B. 网格距离差 [B, H, W]
            # rx_pos[:, i]: [B, 3] -> [B, 1, 1, 3]
            dist_grid_i = torch.norm(grid_3d - rx_pos[:, i].view(B, 1, 1, 3), dim=3)
            dist_grid_j = torch.norm(grid_3d - rx_pos[:, j].view(B, 1, 1, 3), dim=3)
            d_grid = dist_grid_i - dist_grid_j

            # C. 判定双曲线
            diff = (d_grid - d_true).abs()
            # 累加到 Mask (逻辑或)
            masks = torch.maximum(masks, (diff < threshold).float())

    return masks


# ================= 主程序 =================
if __name__ == '__main__':
    # Windows 下多进程必须在 if __name__ == '__main__': 下运行
    print(f"模式: {DEVICE}")
    print(f"分辨率: {MAP_SIZE}x{MAP_SIZE}")
    print(f"Workers: {NUM_WORKERS} (IO加速)")

    # 1. 准备目录
    output_dir = os.path.dirname(OUTPUT_H5_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. 创建 DataLoader
    raw_dataset = SignalRawDataset(DATASET_ROOT, signal_len=SIGNAL_LEN)
    total_num = len(raw_dataset)
    print(f"总样本数: {total_num}")

    # 获取 num_rx 用于初始化 H5
    sample_iq, _, _ = raw_dataset[0]
    num_rx = sample_iq.shape[0]

    # pin_memory=True 可以加速从 CPU 到 GPU 的数据传输
    dataloader = DataLoader(raw_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    # 3. 初始化 HDF5
    iq_shape = (total_num, num_rx * 2, SIGNAL_LEN)
    map_shape = (total_num, 1, MAP_SIZE, MAP_SIZE)
    coord_shape = (total_num, 3)

    with h5py.File(OUTPUT_H5_PATH, 'w') as f:
        print("初始化 HDF5 (启用 LZF 压缩)...")
        dset_iq = f.create_dataset('iq', shape=iq_shape, dtype='float32')
        dset_map = f.create_dataset('heatmap', shape=map_shape, dtype='float32', compression="lzf")
        dset_mask = f.create_dataset('mask', shape=map_shape, dtype='float32', compression="lzf")
        dset_coord = f.create_dataset('coord', shape=coord_shape, dtype='float32')

        # 4. 流水线循环
        current_idx = 0

        for iq_batch, rx_batch, tx_batch in tqdm(dataloader, desc="Accelerated Generating"):
            # 此时 iq_batch 已经在内存中 (由 Workers 预取)
            batch_len = iq_batch.shape[0]

            # --- GPU 计算 ---
            # 1. Heatmap
            heatmaps = generate_dpd_map_batch(
                iq_batch, rx_batch, MAP_SIZE, SCENE_SIZE, FS, DEVICE
            )

            # 2. Mask (现在也在 GPU 上算!)
            masks = generate_ideal_mask_batch_gpu(
                rx_batch, tx_batch, MAP_SIZE, SCENE_SIZE, DEVICE
            )

            # --- 数据整理 & 写入 ---
            # 移回 CPU 并转 numpy
            heatmaps_np = heatmaps.cpu().numpy()[:, np.newaxis, :, :]
            masks_np = masks.cpu().numpy()[:, np.newaxis, :, :]

            # IQ 处理: Complex -> Real Stack
            # iq_batch 是 [B, Rx, L] complex
            iq_real = torch.cat([iq_batch.real, iq_batch.imag], dim=1).numpy()

            # 坐标归一化
            coords_np = (tx_batch / SCENE_SIZE).numpy()

            # 写入 H5
            end_idx = current_idx + batch_len
            dset_iq[current_idx:end_idx] = iq_real
            dset_map[current_idx:end_idx] = heatmaps_np
            dset_mask[current_idx:end_idx] = masks_np
            dset_coord[current_idx:end_idx] = coords_np

            current_idx = end_idx

    print(f"\n✅ 生成完毕: {OUTPUT_H5_PATH}")