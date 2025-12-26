import torch


def dpd_search_torch(rcvPos, sig_rcv_time, init_pos, edge, lamda, fs, device='cuda'):
    """
    DPD 搜索的 PyTorch GPU 加速版
    """
    # 0. 检查设备
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, switching to CPU.")
        device = 'cpu'

    dev = torch.device(device)
    c = 299792458.0

    # 1. 数据预处理 (To GPU)
    # rcvPos: (N_rx, 3) -> (N_rx, 2)
    # 注意：PyTorch 默认 float32，如果精度不够需转 float64 (double)
    rx_pos_tensor = torch.tensor(rcvPos[:, :2], dtype=torch.float32, device=dev)

    # 信号转频域
    # sig_rcv_time: (N_rx, Samples)
    sig_tensor = torch.tensor(sig_rcv_time, dtype=torch.complex64, device=dev)
    # FFT & Shift
    sig_freq = torch.fft.fftshift(torch.fft.fft(sig_tensor, dim=1), dim=1)

    N_rx, N_samples = sig_freq.shape

    # 构造频率轴
    freqs = torch.linspace(-fs / 2, fs / 2, N_samples, device=dev)
    # (1, 1, N_samples) 用于广播
    freqs_grid = freqs.view(1, 1, -1)

    # 2. 构建搜索网格
    # 使用 torch 生成网格
    x_min, x_max = init_pos[0] - edge, init_pos[0] + edge
    y_min, y_max = init_pos[1] - edge, init_pos[1] + edge

    # 注意：arange 的终点处理
    x_vec = torch.arange(x_min, x_max + lamda / 10.0, lamda, device=dev)
    y_vec = torch.arange(y_min, y_max + lamda / 10.0, lamda, device=dev)

    # 生成网格坐标点
    grid_x, grid_y = torch.meshgrid(x_vec, y_vec, indexing='ij')
    # Flatten to (N_grid, 2)
    grid_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

    n_grid = grid_points.shape[0]

    # 3. 批量处理 (Batch Processing)
    # 显存优化：一次处理太多点会 OOM (Out of Memory)
    # 1050Ti/1660Ti 级别建议 1000-2000，3090/4090 可以 5000+
    batch_size = 2000

    max_eig_global = -1.0
    best_pos = init_pos

    # 预先扩展维度避免循环中重复操作
    # rx_pos_tensor: (1, N_rx, 2)
    rx_pos_batch = rx_pos_tensor.unsqueeze(0)
    # sig_freq: (1, N_rx, N_samples)
    sig_freq_batch = sig_freq.unsqueeze(0)

    with torch.no_grad():  # 这一步很关键，推理模式不计算梯度，省显存提速
        for i in range(0, n_grid, batch_size):
            # 当前 Batch: (Batch, 2)
            points_batch = grid_points[i: i + batch_size]
            curr_bs = points_batch.shape[0]

            # (Batch, 1, 2) - (1, N_rx, 2) -> (Batch, N_rx, 2)
            dists = torch.norm(points_batch.unsqueeze(1) - rx_pos_batch, dim=2)

            # 时延: (Batch, N_rx)
            taus = dists / c

            # 相位补偿因子: (Batch, N_rx, N_samples)
            # exp(j * 2pi * f * tau)
            phase_shifts = torch.exp(1j * 2 * torch.pi * freqs_grid * taus.unsqueeze(-1))

            # 补偿信号 V: (Batch, N_rx, N_samples)
            # sig_freq_batch 自动广播
            V = sig_freq_batch * phase_shifts

            # 相关矩阵 Q = V @ V^H
            # (Batch, N_rx, N_samples) @ (Batch, N_samples, N_rx) -> (Batch, N_rx, N_rx)
            # PyTorch 的 matmul 自动处理最后两维的矩阵乘法
            # .mH 是 PyTorch 的共轭转置操作 (Adjoint)
            Q = torch.matmul(V, V.mH)

            # 特征值分解 (Batch, N_rx)
            # linalg.eigvalsh 用于 Hermitian 矩阵，更快且稳定
            eigvals = torch.linalg.eigvalsh(Q)

            # 取最大特征值 (最后一列)
            max_eigs = torch.abs(eigvals[:, -1])

            # 找本 Batch 最大
            curr_max_val, curr_idx = torch.max(max_eigs, dim=0)

            if curr_max_val > max_eig_global:
                max_eig_global = curr_max_val.item()
                # 获取对应的坐标
                best_pos = points_batch[curr_idx].cpu().numpy()

    return best_pos, max_eig_global