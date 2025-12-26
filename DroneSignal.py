import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# --- 1. 用户设定参数 (默认值，可被覆盖) ---
TARGET_FREQ_START = 5725e6  # 5.725 GHz
TARGET_FREQ_END = 5850e6  # 5.850 GHz
DEFAULT_BW = 5e6  # 5 MHz 带宽


def generate_realistic_drone_signal(fs=25e6, duration=1e-3, bandwidth=10e6):
    """
    生成高拟真的无人机信号 (QPSK调制 + 成型滤波 + 射频损伤)

    :param fs: 采样率 (Hz)
    :param duration: 信号时长 (s)
    :param bandwidth: 信号有效带宽 (Hz)
    :return: (baseband_signal, rx_center_freq, actual_carrier_freq)
    """
    total_samples = int(fs * duration)

    # --- A. 频率设定 ---
    # 随机选择一个载波频率 (模拟跳频)
    freq_min = TARGET_FREQ_START + bandwidth / 2
    freq_max = TARGET_FREQ_END - bandwidth / 2
    actual_carrier_freq = np.random.uniform(freq_min, freq_max)

    # 设定接收机中心频率 (LO)
    # 真实场景中，接收机通常固定在一个中心频点，或者只有粗略的对准
    # 这里模拟接收机中心频率与真实载波有一定偏差 (例如最大偏差 5MHz)
    rx_center_freq = actual_carrier_freq + np.random.uniform(-5e6, 5e6)

    # 计算基带频率偏移
    freq_offset = actual_carrier_freq - rx_center_freq

    # --- B. 生成基带通信信号 (QPSK 调制) ---
    # 1. 计算符号数 (假设符号率为带宽的 0.8 倍，预留保护间隔)
    symbol_rate = bandwidth * 0.8
    num_symbols = int(duration * symbol_rate)

    # 2. 生成随机比特并映射到 QPSK 星座图 (1+j, 1-j, -1+j, -1-j)
    # 相比高斯噪声，这具有明显的通信特征
    bits = np.random.randint(0, 4, num_symbols)
    phase = (bits * np.pi / 2) + (np.pi / 4)
    symbols = np.exp(1j * phase)

    # 3. 脉冲成型 (Root Raised Cosine - RRC 滤波器模拟)
    # 这里使用简化的插值滤波来模拟成型
    osr = int(fs / symbol_rate)  # 过采样率
    if osr < 1: osr = 1

    # 上采样 (插零)
    symbols_upsampled = np.zeros(total_samples, dtype=complex)
    idx = np.arange(0, min(total_samples, num_symbols * osr), osr)
    symbols_upsampled[idx] = symbols[:len(idx)]

    # 设计 RRC 滤波器 (或简单的 Hamming 窗低通滤波器作为近似)
    # 截止频率设为带宽的一半
    nyquist = fs / 2
    cutoff = (bandwidth / 2) / nyquist
    num_taps = 127
    b = signal.firwin(num_taps, cutoff, window='hamming')

    # 滤波得到基带波形
    baseband_sig = signal.lfilter(b, 1.0, symbols_upsampled)

    # --- C. 添加射频损伤 (RF Impairments) ---
    t = np.arange(total_samples) / fs

    # 1. 添加载波频率偏差 (CFO) - 模拟晶振漂移
    # 加上之前计算的 freq_offset (物理频偏) + 随机微小漂移 (PPM 误差)
    cfo_drift = np.random.normal(0, 1000)  # 1kHz 随机漂移
    total_freq_shift = freq_offset + cfo_drift
    carrier_wave = np.exp(1j * 2 * np.pi * total_freq_shift * t)

    # 2. 添加相位噪声 (Phase Noise) - 模拟振荡器抖动
    # 使用随机游走模型 (Random Walk)
    phase_jitter = np.random.randn(total_samples) * 0.05  # 抖动强度
    phase_noise = np.cumsum(phase_jitter) * 0.01
    phase_term = np.exp(1j * phase_noise)

    # 3. 组合信号
    full_signal = baseband_sig * carrier_wave * phase_term

    # 4. 归一化幅度
    full_signal = full_signal / np.max(np.abs(full_signal))

    return full_signal, rx_center_freq, actual_carrier_freq


# 保持接口兼容，供外部调用 (使用默认参数)
def generate_dwell_signal():
    return generate_realistic_drone_signal(fs=25e6, duration=1e-3)


# --- 调试与可视化 (直接运行此文件时执行) ---
if __name__ == '__main__':
    fs_test = 50e6  # 50 MHz 采样率测试
    sig, fc_center, f_carrier = generate_realistic_drone_signal(fs=fs_test)

    print(f"载波频率: {f_carrier / 1e6:.2f} MHz")
    print(f"接收中心: {fc_center / 1e6:.2f} MHz")
    print(f"频偏: {(f_carrier - fc_center) / 1e6:.2f} MHz")

    plt.figure(figsize=(10, 8))

    # 时域
    plt.subplot(2, 1, 1)
    plt.plot(np.real(sig[:1000]), label='I')
    plt.plot(np.imag(sig[:1000]), label='Q')
    plt.title("Time Domain (First 1000 samples)")
    plt.legend()

    # 频域 (PSD)
    plt.subplot(2, 1, 2)
    f, Pxx = signal.welch(sig, fs_test, nperseg=1024, return_onesided=False)
    f = np.fft.fftshift(f)
    Pxx = np.fft.fftshift(Pxx)
    # 将 X 轴转换为绝对频率
    freq_axis = (f + fc_center) / 1e6

    plt.plot(freq_axis, 10 * np.log10(Pxx))
    plt.axvline(f_carrier / 1e6, color='r', linestyle='--', label='True Carrier')
    plt.axvline(fc_center / 1e6, color='g', linestyle='--', label='Rx Center')
    plt.title("Power Spectral Density")
    plt.xlabel("Frequency (MHz)")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()