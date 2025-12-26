import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from ProcessedSampleDataset import ProcessedSampleDataset

# ================= 配置区域 =================
# 1. 路径设置 (请修改为你实际的路径)
DATASET_ROOT = r"D:\Dataset\SignalDataset\SNR[5, 5]\train"  # 数据集路径
OUTPUT_ROOT = r"D:\Dataset\SignalDataset\Preprocessed_v2"  # 输出保存路径 (建议改名以示区别)

# 2. 物理参数
SCENE_SIZE = 5000.0  # 场景大小 5000m x 5000m
MAP_SIZE = 64  # 输出热力图的分辨率
SIGNAL_LEN = 2048  # 信号截断长度
FS = 10e6  # 采样率 10MHz (需与仿真一致)
C = 299792458.0  # 光速

# 3. GPU 设置
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 100

# ================= 核心函数 1: 生成粗糙 DPD 热力图 (GPU, 3D适配) =================
def generate_dpd_map_batch(iq_batch, rx_pos_batch, map_size, scene_size, fs, device):
    """
    输入:
        iq_batch: [Batch, Rx, Len] (Complex)
        rx_pos_batch: [Batch, Rx, 3] (XYZ coords, 必须包含高度)
    输出:
        heatmaps: [Batch, map_size, map_size]
    """
    B, N_rx, L = iq_batch.shape

    # 1. 预处理信号 (FFT)
    sig_tensor = torch.as_tensor(iq_batch, dtype=torch.complex64, device=device)
    rx_pos_tensor = torch.as_tensor(rx_pos_batch, dtype=torch.float32, device=device)

    # FFT & Shift
    sig_freq = torch.fft.fftshift(torch.fft.fft(sig_tensor, dim=2), dim=2)

    # 频率轴
    freqs = torch.linspace(-fs / 2, fs / 2, L, device=device).reshape(1, 1, L)

    # 2. 构建 2D 搜索网格 (假设我们在地面 Z=0 处搜索发射机)
    x = torch.linspace(0, scene_size, map_size, device=device)
    y = torch.linspace(0, scene_size, map_size, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')

    # Flatten: [N_grid, 2]
    grid_points_2d = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    N_grid = grid_points_2d.shape[0]

    # 【修复点】：将 2D 网格扩展为 3D 点 (Z=0)，以便计算到空中接收机的斜距
    zeros = torch.zeros(N_grid, 1, device=device)
    grid_points_3d = torch.cat([grid_points_2d, zeros], dim=1)  # [N_grid, 3]

    # 3. 向量化计算 (Vectorized DPD)
    # 维度扩展用于广播:
    # grid: [1, N_grid, 1, 3]  (Broadcast over Batch and Rx)
    # rx:   [B, 1,      N_rx, 3]
    grid_exp = grid_points_3d.view(1, N_grid, 1, 3)
    rx_exp = rx_pos_tensor.view(B, 1, N_rx, 3)

    # 3D 距离: [B, N_grid, N_rx]
    # 这里计算的是从"地面假设点"到"空中接收机"的直线距离
    dists = torch.norm(grid_exp - rx_exp, dim=3)
    taus = dists / C

    # 相位补偿: [B, N_grid, N_rx, L]
    # exp(j * 2pi * f * tau)
    phase = torch.exp(1j * 2 * torch.pi * freqs.unsqueeze(0) * taus.unsqueeze(-1))

    # 补偿信号 V: [B, N_grid, N_rx, L]
    V = sig_freq.view(B, 1, N_rx, L) * phase

    # 相关矩阵 Q: [B, N_grid, N_rx, N_rx]
    Q = torch.matmul(V, V.mH)

    # 特征值分解
    eigvals = torch.linalg.eigvalsh(Q)

    # 取最大特征值: [B, N_grid]
    max_eigs = eigvals[..., -1].abs()

    # Reshape 回图像: [B, H, W]
    heatmaps = max_eigs.view(B, map_size, map_size)

    # 归一化 (0-1)
    h_min = heatmaps.amin(dim=(1, 2), keepdim=True)
    h_max = heatmaps.amax(dim=(1, 2), keepdim=True)
    heatmaps = (heatmaps - h_min) / (h_max - h_min + 1e-8)

    return heatmaps.cpu().numpy()


# ================= 核心函数 2: 生成完美双曲线 Mask (CPU, 3D适配) =================
def generate_ideal_mask_batch(rx_pos_batch, tx_pos_batch, map_size, scene_size):
    """
    输入:
        rx_pos_batch: [Batch, Rx, 3] (XYZ)
        tx_pos_batch: [Batch, 3] (XYZ)
    输出:
        masks: [Batch, map_size, map_size]
    """
    B = rx_pos_batch.shape[0]
    masks = np.zeros((B, map_size, map_size), dtype=np.float32)

    # 构建地面网格 (CPU numpy)
    x = np.linspace(0, scene_size, map_size)
    y = np.linspace(0, scene_size, map_size)
    gv, uv = np.meshgrid(x, y)  # [H, W]

    # 像素分辨率与线宽
    pixel_res = scene_size / map_size
    threshold = pixel_res * 1.5

    for b in range(B):
        rx_pos = rx_pos_batch[b]  # [Rx, 3]
        tx_pos = tx_pos_batch[b]  # [3]
        num_rx = rx_pos.shape[0]

        canvas = np.zeros((map_size, map_size), dtype=np.float32)

        for i in range(num_rx):
            for j in range(i + 1, num_rx):
                # 【修复点1】计算真实的 3D TDOA 距离差 (Ground Truth)
                # 信号在空中传播，必须用 3D 坐标算距离
                d_true = np.linalg.norm(tx_pos - rx_pos[i]) - np.linalg.norm(tx_pos - rx_pos[j])

                # 【修复点2】计算网格点到接收机的 3D 距离
                # 网格点在地面 Z=0，接收机在空中 Z=h
                # dist = sqrt((x-rx)^2 + (y-ry)^2 + (0-rz)^2)
                dist_i = np.sqrt((gv - rx_pos[i][0]) ** 2 + (uv - rx_pos[i][1]) ** 2 + (0 - rx_pos[i][2]) ** 2)
                dist_j = np.sqrt((gv - rx_pos[j][0]) ** 2 + (uv - rx_pos[j][1]) ** 2 + (0 - rx_pos[j][2]) ** 2)

                d_grid = dist_i - dist_j

                # 绘制双曲线投影
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
    indices = np.arange(total_num)

    for start_idx in tqdm(range(0, total_num, BATCH_SIZE), desc="Generating"):
        end_idx = min(start_idx + BATCH_SIZE, total_num)
        batch_indices = indices[start_idx:end_idx]

        # --- A. 读取数据 ---
        batch_iq = []
        batch_rx_pos = []
        batch_tx_pos = []

        for idx in batch_indices:
            # 读取单条数据
            (iq, rx_info), tx_info = dataset.getData(idx, isGetFsFc=False)

            # 截断或补零 IQ
            if iq.shape[1] > SIGNAL_LEN:
                iq = iq[:, :SIGNAL_LEN]
            elif iq.shape[1] < SIGNAL_LEN:
                iq = np.pad(iq, ((0, 0), (0, SIGNAL_LEN - iq.shape[1])))

            # 解析位置 (保留 3D)
            N_rx = iq.shape[0]
            # rx_info 结构: [Rx, 6] -> 取前3个为位置 XYZ
            rx_pos = rx_info[:N_rx * 6].reshape(N_rx, 6)[:, :3]
            # tx_info 结构: [Tx, 6] -> 取前3个为位置 XYZ
            tx_pos = tx_info[:3]

            batch_iq.append(iq)
            # 【修复点】这里不再切片 [:2]，而是传递完整的 [Rx, 3]
            batch_rx_pos.append(rx_pos)
            batch_tx_pos.append(tx_pos)

        batch_iq = np.array(batch_iq)  # [B, Rx, L]
        batch_rx_pos = np.array(batch_rx_pos)  # [B, Rx, 3]
        batch_tx_pos = np.array(batch_tx_pos)  # [B, 3]

        # --- B. 生成粗糙热力图 (Input) ---
        heatmaps = generate_dpd_map_batch(
            batch_iq, batch_rx_pos,
            MAP_SIZE, SCENE_SIZE, FS, DEVICE
        )

        # --- C. 生成完美掩码 (Label) ---
        masks = generate_ideal_mask_batch(
            batch_rx_pos, batch_tx_pos,
            MAP_SIZE, SCENE_SIZE
        )

        # --- D. 保存 ---
        for i, idx in enumerate(batch_indices):
            np.save(os.path.join(heatmap_dir, f"{idx}.npy"), heatmaps[i].astype(np.float32))
            np.save(os.path.join(mask_dir, f"{idx}.npy"), masks[i].astype(np.float32))

    print("\n所有数据生成完毕！")

    # --- 可视化检查 ---
    print("正在展示最后一条样本的可视化结果...")
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Rough DPD Heatmap (Input)")
    plt.imshow(heatmaps[-1], origin='lower', extent=[0, SCENE_SIZE, 0, SCENE_SIZE], cmap='jet')
    plt.colorbar()
    # 画出真实位置 (2D投影)
    plt.scatter(batch_tx_pos[-1][0], batch_tx_pos[-1][1], c='r', marker='x', label='True Tx')

    plt.subplot(1, 2, 2)
    plt.title("Ideal Hyperbola Mask (Label)")
    plt.imshow(masks[-1], origin='lower', extent=[0, SCENE_SIZE, 0, SCENE_SIZE], cmap='gray')
    plt.scatter(batch_tx_pos[-1][0], batch_tx_pos[-1][1], c='r', marker='x')

    plt.legend()
    plt.show()