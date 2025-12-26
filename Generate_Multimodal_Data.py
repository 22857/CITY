import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 路径设置 (请修改为你实际的路径)
DATASET_ROOT = r"D:\Dataset\SignalDataset\SNR[5, 5]\train"  # 建议先生成 train 集
OUTPUT_ROOT = r"D:\Dataset\SignalDataset\Preprocessed_v1"  # 输出保存路径

# 2. 物理参数
SCENE_SIZE = 5000.0  # 场景大小 5000m x 5000m (对应 MakeCsvIQData 中的配置)
MAP_SIZE = 64  # 输出热力图的分辨率 (64x64 对于 CNN 足够了)
SIGNAL_LEN = 2048  # 用于生成热力图的信号截断长度 (短一点速度快)
FS = 10e6  # 采样率 20MHz
C = 299792458.0  # 光速

# 3. GPU 设置
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 100  # 每次处理多少条数据 (显存不够改小)

# ================= 导入 Dataset =================
# 动态添加路径以防报错
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'DataLoader'))
try:
    from ProcessedSampleDataset import ProcessedSampleDataset
except ImportError:
    # 假设 ProcessedSampleDataset.py 在同级目录
    try:
        from ProcessedSampleDataset import ProcessedSampleDataset
    except ImportError:
        print("【错误】找不到 ProcessedSampleDataset，请检查路径配置。")
        sys.exit(1)


# ================= 核心函数 1: 生成粗糙 DPD 热力图 (GPU) =================
def generate_dpd_map_batch(iq_batch, rx_pos_batch, map_size, scene_size, fs, device):
    """
    输入:
        iq_batch: [Batch, Rx, Len] (Complex)
        rx_pos_batch: [Batch, Rx, 2] (XY coords)
    输出:
        heatmaps: [Batch, map_size, map_size]
    """
    B, N_rx, L = iq_batch.shape

    # 1. 预处理信号 (FFT)
    # 转为 Tensor 并移至 GPU
    sig_tensor = torch.as_tensor(iq_batch, dtype=torch.complex64, device=device)
    rx_pos_tensor = torch.as_tensor(rx_pos_batch, dtype=torch.float32, device=device)

    # FFT & Shift
    sig_freq = torch.fft.fftshift(torch.fft.fft(sig_tensor, dim=2), dim=2)

    # 频率轴
    freqs = torch.linspace(-fs / 2, fs / 2, L, device=device).reshape(1, 1, L)

    # 2. 构建网格 (Grid)
    # 生成 map_size x map_size 的网格坐标
    x = torch.linspace(0, scene_size, map_size, device=device)
    y = torch.linspace(0, scene_size, map_size, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')  # 注意 xy 顺序

    # Flatten: [M*M, 2]
    grid_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    N_grid = grid_points.shape[0]

    # 3. 向量化计算 (Vectorized DPD)
    # 为了避免 B * Grid 显存爆炸，我们对 grid 进行分块或者直接广播
    # 这里假设 B=100, Grid=64*64=4096，总数 40w，显存应该够用

    # 维度扩展用于广播:
    # grid: [1, N_grid, 1, 2]  (Broadcast over Batch and Rx)
    # rx:   [B, 1,      N_rx, 2]
    grid_exp = grid_points.view(1, N_grid, 1, 2)
    rx_exp = rx_pos_tensor.view(B, 1, N_rx, 2)

    # 距离: [B, N_grid, N_rx]
    dists = torch.norm(grid_exp - rx_exp, dim=3)
    taus = dists / C

    # 相位补偿: [B, N_grid, N_rx, L]
    # exp(j * 2pi * f * tau)
    # taus: [B, G, R, 1]
    # freqs: [1, 1, 1, L]
    phase = torch.exp(1j * 2 * torch.pi * freqs.unsqueeze(0) * taus.unsqueeze(-1))

    # 补偿信号 V: [B, N_grid, N_rx, L]
    # sig: [B, 1, N_rx, L]
    V = sig_freq.view(B, 1, N_rx, L) * phase

    # 相关矩阵 Q: [B, N_grid, N_rx, N_rx]
    # matmul 作用于最后两维: [..., Rx, L] @ [..., L, Rx]
    Q = torch.matmul(V, V.mH)

    # 特征值分解
    # eigvalsh 用于 Hermitian 矩阵
    eigvals = torch.linalg.eigvalsh(Q)

    # 取最大特征值 (Spectrum Power): [B, N_grid]
    max_eigs = eigvals[..., -1].abs()

    # Reshape 回图像: [B, H, W]
    # 注意 meshgrid 的 indexing='xy' 对应的 reshape 顺序
    heatmaps = max_eigs.view(B, map_size, map_size)

    # 归一化 (每张图归一化到 0-1)
    # min: [B, 1, 1], max: [B, 1, 1]
    h_min = heatmaps.amin(dim=(1, 2), keepdim=True)
    h_max = heatmaps.amax(dim=(1, 2), keepdim=True)
    heatmaps = (heatmaps - h_min) / (h_max - h_min + 1e-8)

    return heatmaps.cpu().numpy()


# ================= 核心函数 2: 生成完美双曲线 Mask (CPU) =================
def generate_ideal_mask_batch(rx_pos_batch, tx_pos_batch, map_size, scene_size):
    """
    输入:
        rx_pos_batch: [Batch, Rx, 3] (只用前两维)
        tx_pos_batch: [Batch, 3] (XY coords)
    输出:
        masks: [Batch, map_size, map_size]
    """
    B = rx_pos_batch.shape[0]
    masks = np.zeros((B, map_size, map_size), dtype=np.float32)

    # 构建网格 (CPU numpy)
    x = np.linspace(0, scene_size, map_size)
    y = np.linspace(0, scene_size, map_size)
    gv, uv = np.meshgrid(x, y)  # [H, W]

    # 像素代表的物理距离 (用于设定线宽)
    pixel_res = scene_size / map_size
    threshold = pixel_res * 1.5  # 线宽容差

    for b in range(B):
        rx_pos = rx_pos_batch[b]  # [Rx, 3]
        tx_pos = tx_pos_batch[b]  # [3]
        num_rx = rx_pos.shape[0]

        # 累加双曲线
        canvas = np.zeros((map_size, map_size), dtype=np.float32)

        for i in range(num_rx):
            for j in range(i + 1, num_rx):
                # 真值 TDOA 距离差
                d_true = np.linalg.norm(tx_pos - rx_pos[i]) - np.linalg.norm(tx_pos - rx_pos[j])

                # 网格 TDOA 距离差
                dist_i = np.sqrt((gv - rx_pos[i][0]) ** 2 + (uv - rx_pos[i][1]) ** 2)
                dist_j = np.sqrt((gv - rx_pos[j][0]) ** 2 + (uv - rx_pos[j][1]) ** 2)
                d_grid = dist_i - dist_j

                # 绘制
                # 使用高斯模糊的思路或者硬阈值，这里用硬阈值方便分割
                mask_line = np.abs(d_grid - d_true) < threshold
                canvas[mask_line] = 1.0

        masks[b] = canvas

    return masks


# ================= 主程序 =================
if __name__ == '__main__':
    print(f"模式: {DEVICE}")
    print(f"数据源: {DATASET_ROOT}")
    print(f"输出目录: {OUTPUT_ROOT}")

    # 1. 准备目录
    heatmap_dir = os.path.join(OUTPUT_ROOT, "Heatmaps_Input")
    mask_dir = os.path.join(OUTPUT_ROOT, "Masks_Label")
    os.makedirs(heatmap_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # 2. 加载数据集
    dataset = ProcessedSampleDataset(DATASET_ROOT, dataType='IQ')
    total_num = dataset.getDataNum()
    print(f"总样本数: {total_num}")

    # 3. 批处理循环
    # 我们手动分 Batch 读取，避免 DataLoader 的复杂性
    indices = np.arange(total_num)

    for start_idx in tqdm(range(0, total_num, BATCH_SIZE), desc="Generating"):
        end_idx = min(start_idx + BATCH_SIZE, total_num)
        batch_indices = indices[start_idx:end_idx]
        current_bs = len(batch_indices)

        # --- A. 读取数据 ---
        batch_iq = []
        batch_rx_pos = []
        batch_tx_pos = []

        for idx in batch_indices:
            # 读取单条数据
            (iq, rx_info), tx_info = dataset.getData(idx, isGetFsFc=False)

            # 截断 IQ
            if iq.shape[1] > SIGNAL_LEN:
                iq = iq[:, :SIGNAL_LEN]
            elif iq.shape[1] < SIGNAL_LEN:
                # 补零 (虽不常见)
                iq = np.pad(iq, ((0, 0), (0, SIGNAL_LEN - iq.shape[1])))

            # 解析位置
            # rx_info: [Rx*6] -> [Rx, 6] -> [Rx, 3] (XYZ)
            N_rx = iq.shape[0]
            rx_pos = rx_info[:N_rx * 6].reshape(N_rx, 6)[:, :3]

            # tx_info: [Tx*6] -> [3] (XYZ)
            tx_pos = tx_info[:3]

            batch_iq.append(iq)
            batch_rx_pos.append(rx_pos[:, :2])  # 仅用 XY 用于 DPD 生成
            batch_tx_pos.append(tx_pos)  # 包含 Z 用于计算真实距离(更准)或仅用XY

        batch_iq = np.array(batch_iq)  # [B, Rx, L]
        batch_rx_pos = np.array(batch_rx_pos)  # [B, Rx, 2]
        batch_tx_pos = np.array(batch_tx_pos)  # [B, 3]

        # --- B. 生成粗糙热力图 (Input) ---
        # 调用 GPU 函数
        heatmaps = generate_dpd_map_batch(
            batch_iq, batch_rx_pos,
            MAP_SIZE, SCENE_SIZE, FS, DEVICE
        )

        # --- C. 生成完美掩码 (Label) ---
        # 调用 CPU 函数
        masks = generate_ideal_mask_batch(
            batch_rx_pos, batch_tx_pos,  # 注意这里 mask生成最好用三维坐标算距离，然后投影
            MAP_SIZE, SCENE_SIZE
        )

        # --- D. 保存 ---
        for i, idx in enumerate(batch_indices):
            # 保存热力图 (float32, 0-1)
            np.save(os.path.join(heatmap_dir, f"{idx}.npy"), heatmaps[i].astype(np.float32))

            # 保存掩码 (float32, 0 or 1)
            np.save(os.path.join(mask_dir, f"{idx}.npy"), masks[i].astype(np.float32))

    print("\n所有数据生成完毕！")
    print(f"热力图位置: {heatmap_dir}")
    print(f"掩码图位置: {mask_dir}")

    # --- 可视化检查 (看最后一条) ---
    print("正在展示最后一条样本的可视化结果...")
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Rough DPD Heatmap (Input)")
    plt.imshow(heatmaps[-1], origin='lower', extent=[0, SCENE_SIZE, 0, SCENE_SIZE], cmap='jet')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("Ideal Hyperbola Mask (Label)")
    plt.imshow(masks[-1], origin='lower', extent=[0, SCENE_SIZE, 0, SCENE_SIZE], cmap='gray')

    # 画出真实位置
    tx_x, tx_y = batch_tx_pos[-1][:2]
    plt.scatter(tx_x, tx_y, c='r', marker='x', label='True Tx')
    plt.legend()

    plt.show()