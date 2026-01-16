import numpy as np
import scipy.fft
import DroneSignal

def generate_urban_multipath_params(max_delay_spread=1e-6, num_taps=4):
    """
    生成城市环境下的多径参数 (相对时延, 相对复数增益)
    基线修改：Dense Urban
    - max_delay_spread: 5e-6 (5us)
    - num_taps: 12
    """
    # 1. 相对延迟 (首径为 0)
    delays = np.sort(np.random.uniform(0, max_delay_spread, num_taps))
    delays[0] = 0.0  # 强制首径为 LOS 或主径

    # 2. 平均功率衰减 (指数衰减模型 PDP)
    # 城市环境常见 RMS delay spread ~ 1us (max / 3~5)
    tau_rms = max_delay_spread / 4.0
    avg_powers = np.exp(-delays / tau_rms)

    # 3. 瑞利衰落 (Rayleigh Fading)
    gains = (np.random.randn(num_taps) + 1j * np.random.randn(num_taps)) / np.sqrt(2)
    gains = gains * np.sqrt(avg_powers)

    # 4. 能量归一化
    total_energy = np.sum(np.abs(gains) ** 2)
    gains = gains / np.sqrt(total_energy)

    return delays, gains

def apply_fractional_delay_and_doppler(sig, delay_s, doppler_hz, fs):
    """
    应用亚采样时延和多普勒
    """
    N = len(sig)
    # 1. 多普勒 (时域旋转)
    t = np.arange(N) / fs
    doppler_phasor = np.exp(1j * 2 * np.pi * doppler_hz * t)
    sig_doppler = sig * doppler_phasor

    # 2. 亚采样时延 (频域相移)
    freqs = scipy.fft.fftfreq(N, d=1 / fs)
    phase_shift = np.exp(-1j * 2 * np.pi * freqs * delay_s)
    sig_spec = scipy.fft.fft(sig_doppler)
    sig_delayed = scipy.fft.ifft(sig_spec * phase_shift)

    return sig_delayed

def ReceSignalSimple(time, samplingRate, emitterPos, emitterVel, receiverPos, receiverVel, SNR, asynchrony_par_sigma):
    '''
    修正后的物理仿真接收函数 (Urban Baseline)
    '''
    c = 299792458.0
    num_samples = int(time * samplingRate)
    num_receivers = receiverPos.shape[0]
    num_emitters = emitterPos.shape[0]

    emitSignals_record = []
    deltaTs_record = []
    deltaFs_record = []
    fcs_record = []

    mixed_signal_clean = np.zeros((num_receivers, num_samples), dtype=complex)
    reference_noise_power = None

    for k in range(num_emitters):
        # A. 生成基带信号
        safe_bandwidth = min(10e6, samplingRate * 0.8)
        base_sig, _, fc = DroneSignal.generate_realistic_drone_signal(
            fs=samplingRate, duration=time, bandwidth=safe_bandwidth
        )
        fcs_record.append(fc)

        if len(base_sig) > num_samples:
            base_sig = base_sig[:num_samples]
        else:
            base_sig = np.pad(base_sig, (0, num_samples - len(base_sig)), 'constant')

        emit_sig_matrix = np.tile(base_sig, (num_receivers, 1))
        emitSignals_record.append(emit_sig_matrix)

        # B. 几何计算
        e_pos = emitterPos[k]
        e_vel = emitterVel[k]
        diff_vec = receiverPos - e_pos
        dist = np.sqrt(np.sum(diff_vec ** 2, axis=1))

        deltaT_truth = dist / c
        rel_vel = receiverVel - e_vel
        unit_vec = diff_vec / (dist[:, None] + 1e-9)
        radial_vel = np.sum(rel_vel * unit_vec, axis=1)
        deltaF_truth = (radial_vel / c) * fc

        deltaTs_record.append(deltaT_truth)
        deltaFs_record.append(deltaF_truth)

        effective_deltaT = deltaT_truth.copy()
        effective_deltaF = deltaF_truth.copy()

        if asynchrony_par_sigma is not None:
            dt_err = np.random.normal(0, asynchrony_par_sigma[0], num_receivers)
            df_err = np.random.normal(0, asynchrony_par_sigma[1], num_receivers)
            dt_err[0] = 0; df_err[0] = 0
            effective_deltaT += dt_err
            effective_deltaF += df_err

        # C. 信号传播
        for r_idx in range(num_receivers):
            # [修改点] 路损指数 alpha = 3.8 (Urban)
            alpha = 3.8
            d = dist[r_idx] if dist[r_idx] > 1.0 else 1.0

            # 阴影衰落
            shadowing_std_db = 6.0
            shadowing_db = np.random.normal(0, shadowing_std_db)
            shadowing_linear = 10 ** (shadowing_db / 20.0)

            path_loss_gain = (1.0 / (d ** (alpha / 2.0))) * shadowing_linear * 1e5

            if k == 0 and r_idx == 0:
                ref_sig_power = np.mean(np.abs(base_sig * path_loss_gain) ** 2)
                reference_noise_power = ref_sig_power / (10 ** (SNR[0] / 10.0))

            # [修改点] 获取城市多径参数 (12径, 5us)
            mp_delays, mp_gains = generate_urban_multipath_params(
                max_delay_spread=5e-6, 
                num_taps=12
            )

            rx_sig_acc = np.zeros_like(base_sig)
            main_delay = effective_deltaT[r_idx]
            main_doppler = effective_deltaF[r_idx]

            for tap_i in range(len(mp_delays)):
                tap_total_delay = main_delay + mp_delays[tap_i]
                tap_total_doppler = main_doppler
                tap_complex_amp = path_loss_gain * mp_gains[tap_i]

                tap_sig = apply_fractional_delay_and_doppler(
                    base_sig, tap_total_delay, tap_total_doppler, samplingRate
                )
                rx_sig_acc += tap_sig * tap_complex_amp

            mixed_signal_clean[r_idx] += rx_sig_acc

    # D. 添加噪声
    if reference_noise_power is None or reference_noise_power == 0:
        reference_noise_power = 1e-12

    noise_std = np.sqrt(reference_noise_power)
    noise_matrix = (np.random.randn(num_receivers, num_samples) +
                    1j * np.random.randn(num_receivers, num_samples)) / np.sqrt(2)
    noise_matrix = noise_matrix * noise_std

    final_rec_signals = mixed_signal_clean + noise_matrix

    return emitSignals_record, final_rec_signals, deltaTs_record, deltaFs_record, fcs_record