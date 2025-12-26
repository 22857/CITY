import numpy as np
import scipy.fft
import DroneSignal


def generate_urban_multipath_params(max_delay_spread=3e-6, num_taps=6):
    """
    生成城市环境下的多径参数 (相对时延, 相对复数增益)
    :return: relative_delays (s), relative_gains (complex)
    """
    # 1. 相对延迟 (首径为 0)
    # 随机化延迟分布，模拟不同反射体距离
    delays = np.sort(np.random.uniform(0, max_delay_spread, num_taps))
    delays[0] = 0.0  # 强制首径为 LOS 或主径

    # 2. 平均功率衰减 (指数衰减模型 PDP)
    # 城市环境常见 RMS delay spread ~ 1us
    tau_rms = max_delay_spread / 3.0
    avg_powers = np.exp(-delays / tau_rms)

    # 3. 瑞利衰落 (Rayleigh Fading) 生成复数增益
    # 实部虚部服从 N(0, 1/2 * Power)
    gains = (np.random.randn(num_taps) + 1j * np.random.randn(num_taps)) / np.sqrt(2)
    gains = gains * np.sqrt(avg_powers)

    # 4. 能量归一化 (使得多径信道总增益期望为 1)
    total_energy = np.sum(np.abs(gains) ** 2)
    gains = gains / np.sqrt(total_energy)

    return delays, gains


def apply_fractional_delay_and_doppler(sig, delay_s, doppler_hz, fs):
    """
    同时应用亚采样时延(频域相移)和多普勒频移(时域旋转)
    """
    N = len(sig)

    # --- 1. 多普勒频移 (时域) ---
    # f_doppler 导致相位随时间线性旋转
    t = np.arange(N) / fs
    doppler_phasor = np.exp(1j * 2 * np.pi * doppler_hz * t)
    sig_doppler = sig * doppler_phasor

    # --- 2. 亚采样时延 (频域) ---
    # 时域延迟 tau <-> 频域相移 exp(-j * 2pi * f * tau)
    # 使用 scipy.fft 处理，它比 numpy.fft 在某些版本下更快
    freqs = scipy.fft.fftfreq(N, d=1 / fs)
    # 构造相位旋转向量
    phase_shift = np.exp(-1j * 2 * np.pi * freqs * delay_s)

    # FFT -> 相位旋转 -> IFFT
    sig_spec = scipy.fft.fft(sig_doppler)
    sig_delayed = scipy.fft.ifft(sig_spec * phase_shift)

    return sig_delayed


def ReceSignalSimple(time, samplingRate, emitterPos, emitterVel, receiverPos, receiverVel, SNR, asynchrony_par_sigma):
    '''
    修正后的物理仿真接收函数

    主要修复：
    1. 噪声模型：固定噪声底噪，而非根据接收功率动态调整。
    2. 时延精度：使用频域方法实现亚采样精度。
    3. 多径模型：显式叠加不同时延/多普勒的路径。
    '''

    # --- 1. 初始化 ---
    c = 299792458.0
    num_samples = int(time * samplingRate)
    num_receivers = receiverPos.shape[0]
    num_emitters = emitterPos.shape[0]

    emitSignals_record = []
    deltaTs_record = []
    deltaFs_record = []
    fcs_record = []

    # 最终混合信号缓存 (Rx, Samples)
    mixed_signal_clean = np.zeros((num_receivers, num_samples), dtype=complex)

    # 预计算噪声功率的参考值 (基于发射机0 -> 接收机0 的视距传播)
    # 假设参考距离 100m (或者直接用 Rx0 的真实距离)
    # 用于确定一个全局统一的 "Noise Floor"
    reference_noise_power = None

    # --- 2. 遍历发射机 ---
    for k in range(num_emitters):

        # A. 生成基带信号
        safe_bandwidth = min(10e6, samplingRate * 0.8)
        base_sig, _, fc = DroneSignal.generate_realistic_drone_signal(
            fs=samplingRate, duration=time, bandwidth=safe_bandwidth
        )
        fcs_record.append(fc)

        # 截断或补零
        if len(base_sig) > num_samples:
            base_sig = base_sig[:num_samples]
        else:
            base_sig = np.pad(base_sig, (0, num_samples - len(base_sig)), 'constant')

        # 记录 (兼容旧接口)
        emit_sig_matrix = np.tile(base_sig, (num_receivers, 1))
        emitSignals_record.append(emit_sig_matrix)

        # B. 几何计算
        e_pos = emitterPos[k]
        e_vel = emitterVel[k]

        # 向量: Tx -> Rx
        diff_vec = receiverPos - e_pos  # (N_rx, 3)
        dist = np.sqrt(np.sum(diff_vec ** 2, axis=1))  # (N_rx,)

        # 理论值记录 (Label)
        deltaT_truth = dist / c

        rel_vel = receiverVel - e_vel
        unit_vec = diff_vec / (dist[:, None] + 1e-9)
        radial_vel = np.sum(rel_vel * unit_vec, axis=1)
        deltaF_truth = (radial_vel / c) * fc

        deltaTs_record.append(deltaT_truth)
        deltaFs_record.append(deltaF_truth)

        # C. 接收机误差注入
        effective_deltaT = deltaT_truth.copy()
        effective_deltaF = deltaF_truth.copy()

        if asynchrony_par_sigma is not None:
            dt_err = np.random.normal(0, asynchrony_par_sigma[0], num_receivers)
            df_err = np.random.normal(0, asynchrony_par_sigma[1], num_receivers)
            # 假设 Rx0 是参考时钟，无误差 (可选)
            dt_err[0] = 0;
            df_err[0] = 0

            effective_deltaT += dt_err
            effective_deltaF += df_err

        # --- 3. 信号传播 (核心修正部分) ---
        for r_idx in range(num_receivers):

            # --- Path Loss 计算 ---
            # 城市环境路损指数 alpha ~ 3.5
            alpha = 3.5
            d = dist[r_idx] if dist[r_idx] > 1.0 else 1.0

            # 阴影衰落 (Log-normal Shadowing)
            shadowing_std_db = 6.0
            shadowing_db = np.random.normal(0, shadowing_std_db)
            shadowing_linear = 10 ** (shadowing_db / 20.0)

            # 综合幅度增益
            # 1e5 是缩放因子，避免数值下溢 (归一化时会抵消)
            path_loss_gain = (1.0 / (d ** (alpha / 2.0))) * shadowing_linear * 1e5

            # 如果是参考链路 (Tx0 -> Rx0)，记录下来用于计算噪声
            if k == 0 and r_idx == 0:
                # 记录参考信号功率 (假设无多径时的视距功率)
                ref_sig_power = np.mean(np.abs(base_sig * path_loss_gain) ** 2)
                # 根据目标 SNR 计算噪声功率
                # Noise = Signal / 10^(SNR/10)
                reference_noise_power = ref_sig_power / (10 ** (SNR[0] / 10.0))

            # --- 多径信道生成 ---
            # 获取当前链路的多径参数 (Delay, Gain)
            # 这里的 delays 是相对于视距径(LOS)的额外延迟
            mp_delays, mp_gains = generate_urban_multipath_params(max_delay_spread=2e-6, num_taps=6)

            # 接收信号累加器
            rx_sig_acc = np.zeros_like(base_sig)

            # 主视距时延和多普勒
            main_delay = effective_deltaT[r_idx]
            main_doppler = effective_deltaF[r_idx]

            # 遍历每一径 (LOS + NLOS)
            for tap_i in range(len(mp_delays)):
                # 1. 每一径的总绝对时延
                tap_total_delay = main_delay + mp_delays[tap_i]

                # 2. 每一径的多普勒
                # 简化：假设所有径的多普勒主要由 Tx 运动主导，近似相等
                # (严谨仿真需要根据到达角计算，但在无几何地图时此近似可接受)
                tap_total_doppler = main_doppler

                # 3. 每一径的复数幅度
                tap_complex_amp = path_loss_gain * mp_gains[tap_i]

                # 4. 生成该径的信号分量 (频域延迟 + 时域多普勒)
                # 这是最耗时的步骤，但精度最高
                tap_sig = apply_fractional_delay_and_doppler(
                    base_sig, tap_total_delay, tap_total_doppler, samplingRate
                )

                # 5. 叠加
                rx_sig_acc += tap_sig * tap_complex_amp

            # 叠加到总混合信号 (支持多发射机)
            mixed_signal_clean[r_idx] += rx_sig_acc

    # --- 4. 添加噪声 (Fixed Noise Floor) ---
    final_rec_signals = np.zeros_like(mixed_signal_clean)

    # 兜底：如果 reference_noise_power 未计算 (例如没有发射机)，设为极小值
    if reference_noise_power is None or reference_noise_power == 0:
        reference_noise_power = 1e-12

    # 生成噪声矩阵 (Rx, Samples) - 所有接收机独立噪声
    # 实部虚部各分担一半功率 -> / sqrt(2)
    noise_std = np.sqrt(reference_noise_power)
    noise_matrix = (np.random.randn(num_receivers, num_samples) +
                    1j * np.random.randn(num_receivers, num_samples)) / np.sqrt(2)
    noise_matrix = noise_matrix * noise_std

    # 信号 + 噪声
    final_rec_signals = mixed_signal_clean + noise_matrix

    return emitSignals_record, final_rec_signals, deltaTs_record, deltaFs_record, fcs_record