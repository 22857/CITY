import numpy as np
from scipy.signal import lfilter
import sys
sys.path.append('D:/PycharmProjects/DPV-main/SignalGeneration')
import DroneSignal


def generate_urban_channel_taps(sampling_rate, max_delay_spread=3e-6, num_taps=6):
    """
    生成城市环境下的多径信道抽头 (Rayleigh Fading)
    :param sampling_rate: 采样率 (Hz)
    :param max_delay_spread: 最大时延扩展 (s), 城市环境典型值为 1us ~ 5us
    :param num_taps: 多径数量
    :return: channel_taps (复数系数)
    """
    # 1. 定义每一径的相对延迟 (这里简单均分，也可以随机)
    delays = np.linspace(0, max_delay_spread, num_taps)

    # 2. 定义每一径的平均功率 (指数衰减: 越晚到达的径，功率越小)
    # 衰减因子 tau
    tau = max_delay_spread / 3
    avg_powers = np.exp(-delays / tau)

    # 3. 生成瑞利衰落系数 (复高斯变量)
    # 实部和虚部均服从 N(0, 1/2 * Power)
    taps = (np.random.randn(num_taps) + 1j * np.random.randn(num_taps)) / np.sqrt(2)

    # 4. 根据功率包络调整幅度
    taps = taps * np.sqrt(avg_powers)

    # 5. 归一化总能量 (保证信道本身不放大信号总功率，只改变波形)
    taps = taps / np.sqrt(np.sum(np.abs(taps) ** 2))

    return taps

def ReceSignalSimple(time, samplingRate, emitterPos, emitterVel, receiverPos, receiverVel, SNR, asynchrony_par_sigma):
    '''
    物理仿真的信号接收函数 (适配新的 DroneSignal)

    :param time: 仿真总时长 (s)
    :param samplingRate: 仿真采样率 (Hz) -> 建议 > 5MHz 以获得米级分辨率
    :param emitterPos: 发射机位置 [N_emit, 3]
    :param emitterVel: 发射机速度 [N_emit, 3]
    :param receiverPos: 接收机位置 [N_recv, 3]
    :param receiverVel: 接收机速度 [N_recv, 3]
    :param SNR: 信噪比列表 [N_emit] (这里我们以第0个发射机为基准设定噪声基底)
    :param asynchrony_par_sigma: 接收机不同步参数 [sigma_t, sigma_f]
    '''

    # --- 1. 初始化 ---
    c = 299792458.0  # 光速
    num_samples = int(time * samplingRate)
    num_receivers = receiverPos.shape[0]
    num_emitters = emitterPos.shape[0]

    # 用于存储返回结果
    emitSignals_record = []  # 记录发射源信号 (纯净)
    deltaTs_record = []  # 记录理论时延
    deltaFs_record = []  # 记录理论频偏
    fcs_record = []  # 记录载波频率

    # 初始化接收机的混合信号缓存 (纯净，未加噪)
    # Shape: [Rx_Num, Samples]
    mixed_clean_signal = np.zeros((num_receivers, num_samples), dtype=complex)

    # --- 2. 遍历每个发射机 (叠加信号) ---
    for k in range(num_emitters):

        # A. 生成该发射机的基带信号
        safe_bandwidth = min(10e6, samplingRate * 0.8)

        base_sig, _, fc = DroneSignal.generate_realistic_drone_signal(
            fs=samplingRate,
            duration=time,
            bandwidth=safe_bandwidth  # 传入安全带宽
        )

        # 记录载波
        fcs_record.append(fc)

        # 确保长度匹配 (截断或补零)
        if len(base_sig) > num_samples:
            base_sig = base_sig[:num_samples]
        else:
            base_sig = np.pad(base_sig, (0, num_samples - len(base_sig)), 'constant')

        # 记录发射源信号 (复制 N_recv 份，为了保持和你原有返回值格式一致)
        # 实际上发射信号只有一份，但为了兼容旧代码结构，我们存成 [Rx, Samples]
        emit_sig_matrix = np.tile(base_sig, (num_receivers, 1))
        emitSignals_record.append(emit_sig_matrix)

        # B. 计算物理参数 (距离, TDOA, FDOA)
        # 发射机 k 的位置和速度
        e_pos = emitterPos[k]
        e_vel = emitterVel[k]

        # 向量计算
        # diff_vec: [Rx_Num, 3] (从发射机指向接收机)
        diff_vec = receiverPos - e_pos
        # dist: [Rx_Num]
        dist = np.sqrt(np.sum(diff_vec ** 2, axis=1))

        # 理论时延 TDOA
        deltaT = dist / c

        # 理论频偏 FDOA (多普勒)
        # 相对速度: Rx_vel - Tx_vel
        rel_vel = receiverVel - e_vel
        # 径向速度投影 = rel_vel dot unit_vec
        # unit_vec = diff_vec / dist
        unit_vec = diff_vec / (dist[:, None] + 1e-9)  # 防止除零
        radial_vel = np.sum(rel_vel * unit_vec, axis=1)
        deltaF = (radial_vel / c) * fc

        # 记录理论值
        deltaTs_record.append(deltaT)
        deltaFs_record.append(deltaF)

        # --- 3. 模拟接收机不同步误差 (Error Injection) ---
        # 假设第0个接收机是主时钟，其他接收机有误差
        # 仅影响信号生成，不影响 label (deltaTs_record)

        # 复制一份用于加误差
        effective_deltaT = deltaT.copy()
        effective_deltaF = deltaF.copy()

        if asynchrony_par_sigma is not None:
            dt_err = np.random.normal(0, asynchrony_par_sigma[0], num_receivers)
            df_err = np.random.normal(0, asynchrony_par_sigma[1], num_receivers)
            # 第0个保持完美同步 (或看你需求)
            dt_err[0] = 0
            df_err[0] = 0

            effective_deltaT += dt_err
            effective_deltaF += df_err

            # --- 4. 生成该发射机到达每个接收机的信号 (URBAN MODIFIED) ---
            for r_idx in range(num_receivers):

                # === 修改 A: 城市路径损耗 (Path Loss + Shadowing) ===
                # 城市环境路损指数 alpha 通常为 3.0 到 4.0 (LOS是2.0)
                alpha = 3.5
                # 参考距离 d0 = 1m
                # PL(dB) = 10 * alpha * log10(d) + Shadowing

                # 计算距离带来的幅度衰减 (线性尺度)
                dist_val = dist[r_idx]
                if dist_val < 1.0: dist_val = 1.0

                # 阴影衰落 (Log-normal Shadowing): 城市环境标准差通常 4-8 dB
                shadowing_std_db = 6.0
                shadowing_db = np.random.normal(0, shadowing_std_db)
                shadowing_linear = 10 ** (shadowing_db / 20.0)  # 转换为电压增益

                # 综合增益: 距离衰减 * 阴影
                # 注意：为了避免数值过小，我们假设 100m 处归一化，或者保留你之前的 1000 系数逻辑
                # 这里使用更物理的衰减模型 (相对值)
                path_loss_gain = (1.0 / (dist_val ** (alpha / 2.0))) * shadowing_linear * 1e5
                # * 1e5 是为了防止数值过小导致下溢，最后靠 SNR 归一化拉回来，没关系

                # === 修改 B: 几何时延 (TDOA) ===
                # 这部分代表"视距传播"的主要到达时间，保持不变
                delay_samples = effective_deltaT[r_idx] * samplingRate
                delay_int = int(round(delay_samples))

                # === 修改 C: 多径传播 (Multipath) ===
                # 1. 生成该链路的特定多径信道
                # 注意：对于同一对 Tx-Rx，在短时间内信道是固定的；
                # 但如果是不同位置的 Rx，信道完全独立。
                channel_h = generate_urban_channel_taps(samplingRate, max_delay_spread=2e-6, num_taps=8)

                # 2. 施加多普勒 (Doppler)
                # 在城市中，多普勒通常施加在每一径上。
                # 简单起见，我们假设多普勒主要由发射机运动引起，对所有径影响近似相同（窄带假设）
                # 或者先加多普勒再过信道
                t_axis = np.arange(num_samples) / samplingRate
                doppler_phasor = np.exp(1j * 2 * np.pi * effective_deltaF[r_idx] * t_axis)

                # 信号先经过多普勒旋转
                sig_with_doppler = base_sig * doppler_phasor

                # 3. 通过多径信道 (卷积)
                # lfilter: 相当于 FIR 滤波，b=channel_h, a=1
                sig_multipath = lfilter(channel_h, 1.0, sig_with_doppler)

                # 4. 施加几何延迟 (TDOA)
                # 这里逻辑不变：先构造空数组，再位移
                rx_sig = np.zeros_like(base_sig)
                if delay_int < num_samples:
                    if delay_int > 0:
                        rx_sig[delay_int:] = sig_multipath[:-delay_int]
                    else:
                        rx_sig[:] = sig_multipath[:]

                # 5. 应用路径损耗
                rx_sig = rx_sig * path_loss_gain

                # 叠加到总混合信号中
                mixed_clean_signal[r_idx] += rx_sig

    # --- 5. 添加噪声 (Receiver Thermal Noise) ---
    # 我们以第0个发射机在第0个接收机上的信号强度为基准，设定 SNR
    # 这样能保证信噪比定义的一致性

    final_rec_signals = np.zeros_like(mixed_clean_signal)

    for r_idx in range(num_receivers):
        clean_sig = mixed_clean_signal[r_idx]

        # 计算纯净信号功率
        signal_power = np.mean(np.abs(clean_sig) ** 2)

        if signal_power == 0:
            final_rec_signals[r_idx] = clean_sig
            continue

        # 计算噪声功率
        # 使用 SNR[0] 作为主要参考，或者你可以取 SNR 的平均值
        target_snr = SNR[0]
        noise_power = signal_power / (10 ** (target_snr / 10.0))

        # 生成复高斯白噪声
        noise = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) / np.sqrt(2)
        noise = noise * np.sqrt(noise_power)

        # 最终信号 = 纯净混合信号 + 噪声
        final_rec_signals[r_idx] = clean_sig + noise

    # --- 6. 整理返回值 ---
    # 你的旧代码有一个 "混合所有 emitter list" 的步骤，但这里我们已经混合好了
    # 为了保持接口兼容，我们返回:
    # emitSignals_record: list of [Rx, Samples] (纯净源)
    # final_rec_signals: [Rx, Samples] (最终混合加噪)

    return emitSignals_record, final_rec_signals, deltaTs_record, deltaFs_record, fcs_record