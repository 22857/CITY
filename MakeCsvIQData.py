import os
import time
import numpy as np
from ReceSignalSimple import ReceSignalSimple
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# ================= 配置区域 =================
CONFIG = {
    'samplingRate': 50e6,
    'num_points': 4096,   # 信号长度
    'SnrRange': [6, 6],   # 信噪比范围 (dB) - 稍微调高一点下限，保证数据质量
    'dataNum': 10000,     # 总样本数量
    'validRate': 0.1,      # 验证集比例
    
    # 接收机数量固定为 6
    'receiverNumRange': [6, 6],
    'emitterNumRange': [1, 1],
    'receiverVelRange': [0, 0], # 接收机静止
    'emitterVelRange': [0, 10], # 发射机慢速移动
    'asynchrony_par_sigma': [0, 0],  # 理想同步
    
    # 输出路径 (建议修改文件夹名以区分不同配置)
    'root_dir': r"/root/autodl-tmp/SignalDataset_6Rx_SafeZone" 
}

# 计算保存路径
SNR_STR = str(CONFIG['SnrRange'])
SAVE_ROOT = os.path.join(CONFIG['root_dir'], "SNR" + SNR_STR)


# ================= 辅助函数：定义固定站点 =================
def get_fixed_receiver_positions():
    """
    【修改点1】定义写死的接收机位置
    布局：正六边形分布
    中心：(2500, 2500)
    半径：2000m
    """
    center_x, center_y = 2500.0, 2500.0
    radius = 2000.0
    height = 10.0
    num_rx = 6
    
    pos = np.zeros((num_rx, 3))
    angles = np.linspace(0, 2 * np.pi, num_rx, endpoint=False)
    
    for i in range(num_rx):
        pos[i, 0] = center_x + radius * np.cos(angles[i])
        pos[i, 1] = center_y + radius * np.sin(angles[i])
        pos[i, 2] = height
        
    return pos, radius, (center_x, center_y)


# ================= 核心工作函数 =================
def generate_single_sample(args):
    """
    单个样本生成函数
    """
    idx, mode, conf = args

    # 设置独立的随机种子
    np.random.seed(int(time.time() * 1000) % 2 ** 32 + idx)

    # --- 1. 获取固定接收机位置 ---
    receiverPos, rx_radius, rx_center = get_fixed_receiver_positions()
    receiverNum = receiverPos.shape[0]
    receiverVel = np.zeros_like(receiverPos) # 静止

    # --- 2. 生成发射机位置 (避开 GDOP 恶劣区域) ---
    # 【修改点2】限制发射机在接收机包围圈内部
    # 策略：在接收机半径的 85% 范围内生成 (Safe Zone)
    # 这样可以保证目标永远被基站包围，GDOP 极佳
    
    emitterNum = int(np.random.uniform(conf['emitterNumRange'][0], conf['emitterNumRange'][1] + 1))
    emitterPos = np.zeros([emitterNum, 3])
    emitterVel = np.zeros_like(emitterPos)
    
    safe_margin_ratio = 0.85 # 安全系数，只在半径 85% 内生成
    max_tx_radius = rx_radius * safe_margin_ratio

    for j in range(emitterNum):
        # 使用极坐标生成均匀分布在圆内的点
        # r = R * sqrt(random()) 是为了保证面积均匀分布
        r = max_tx_radius * np.sqrt(np.random.uniform(0, 1))
        theta = np.random.uniform(0, 2 * np.pi)
        
        emitterPos[j, 0] = rx_center[0] + r * np.cos(theta)
        emitterPos[j, 1] = rx_center[1] + r * np.sin(theta)
        # 高度随机 2m ~ 100m (无人机或地面设备)
        emitterPos[j, 2] = np.random.uniform(2, 100)

        # 速度生成
        emitVel = np.random.uniform(conf['emitterVelRange'][0], conf['emitterVelRange'][1])
        angle_vel = np.random.uniform(0, 2 * np.pi)
        angle_climb = np.random.uniform(-np.pi/6, np.pi/6) # 主要是水平运动
        
        emitterVel[j] = [
            emitVel * np.cos(angle_climb) * np.cos(angle_vel),
            emitVel * np.cos(angle_climb) * np.sin(angle_vel),
            emitVel * np.sin(angle_climb)
        ]

    # SNR 和 采样参数
    samplingTime = conf['num_points'] / conf['samplingRate']
    Snrs = [np.random.uniform(conf['SnrRange'][0], conf['SnrRange'][1]) for _ in range(emitterNum)]

    # --- 3. 信号生成 ---
    # 调用信号模型
    emitSignals, receSignal, deltaTs, deltaFs, fcs = ReceSignalSimple(
        samplingTime, conf['samplingRate'],
        emitterPos, emitterVel,
        receiverPos, receiverVel,
        Snrs, conf['asynchrony_par_sigma']
    )

    # --- 4. 保存信号数据 (.npy) ---
    base_save_path = os.path.join(SAVE_ROOT, mode, "Npy", "IQ")
    max_rx_num = conf['receiverNumRange'][1]
    
    # 保存接收机数据
    for k in range(max_rx_num):
        rx_dir = os.path.join(base_save_path, f"Receiver{k}")
        os.makedirs(rx_dir, exist_ok=True)
        save_file = os.path.join(rx_dir, f"{idx}.npy")

        if k < receiverNum:
            sig_data = receSignal[k].astype(np.complex64).reshape(-1, 1)
        else:
            sig_data = np.zeros((conf['num_points'], 1), dtype=np.complex64)
        np.save(save_file, sig_data)

    # 保存发射机数据
    max_tx_num = conf['emitterNumRange'][1]
    for k in range(max_tx_num):
        tx_dir = os.path.join(base_save_path, f"Emitter{k}")
        os.makedirs(tx_dir, exist_ok=True)
        save_file = os.path.join(tx_dir, f"{idx}.npy")

        if k < emitterNum:
            sig_data = emitSignals[k][0].astype(np.complex64).reshape(-1, 1)
        else:
            sig_data = np.zeros((conf['num_points'], 1), dtype=np.complex64)
        np.save(save_file, sig_data)

    # --- 5. 构造 Information 标签 ---
    # 格式严格对齐原代码，确保兼容性
    label = [idx, receiverNum, emitterNum]

    # SNR & FC
    for j in range(max_tx_num):
        if j < emitterNum:
            label.extend([Snrs[j], fcs[j]])
        else:
            label.extend([0, 0])

    # 环境范围 (填入固定值以便 loader 读取)
    # 注意：虽然这里填的是 Range，但实际数据是固定的，这只是为了格式兼容
    rx_range_mock = [0, 5000] 
    label.extend([
        samplingTime, conf['samplingRate'],
        rx_range_mock[0], rx_range_mock[1], # Rx X range
        rx_range_mock[0], rx_range_mock[1], # Rx Y range
        0, 50,                              # Rx Z range
        conf['receiverVelRange'][0], conf['receiverVelRange'][1],
        0, 5000,                            # Tx X range
        0, 5000,                            # Tx Y range
        0, 200,                             # Tx Z range
        conf['emitterVelRange'][0], conf['emitterVelRange'][1]
    ])

    # Delay / Doppler
    for j in range(max_rx_num):
        for k in range(max_tx_num):
            if j < receiverNum and k < emitterNum:
                label.extend([deltaTs[k][j], deltaFs[k][j]])
            else:
                label.extend([0, 0])

    # Rx Pos/Vel (真实值)
    for j in range(max_rx_num):
        if j < receiverNum:
            label.extend(list(receiverPos[j]) + list(receiverVel[j]))
        else:
            label.extend([0] * 6)

    # Tx Pos/Vel (真实值)
    for j in range(max_tx_num):
        if j < emitterNum:
            label.extend(list(emitterPos[j]) + list(emitterVel[j]))
        else:
            label.extend([0] * 6)

    return label


# ================= 主程序 =================
if __name__ == '__main__':
    # 1. 准备目录
    print(f"数据将生成至: {SAVE_ROOT}")

    for mode in ['train', 'valid']:
        path = os.path.join(SAVE_ROOT, mode, "Npy", "IQ")
        if not os.path.exists(path):
            os.makedirs(path)

    # 2. 生成任务列表
    train_num = int(CONFIG['dataNum'] * (1 - CONFIG['validRate']))
    valid_num = int(CONFIG['dataNum'] * CONFIG['validRate'])

    tasks = []
    for i in range(train_num):
        tasks.append((i, 'train', CONFIG))
    for i in range(valid_num):
        tasks.append((i, 'valid', CONFIG))

    print(f"开始生成 {len(tasks)} 条数据 (6站点固定 - 发射机安全区分布)...")

    # 3. 并行执行
    labels_train = []
    labels_valid = []

    start_time = time.time()

    # 这里的 max_workers 可以根据你的 CPU 核数调整
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(generate_single_sample, tasks), total=len(tasks)))

    # 4. 保存
    print("\n正在保存标签文件...")

    for res, task in zip(results, tasks):
        idx, mode, _ = task
        if mode == 'train':
            labels_train.append(res)
        else:
            labels_valid.append(res)

    if labels_train:
        train_info_path = os.path.join(SAVE_ROOT, 'train', 'Npy', 'IQ', 'Information.npy')
        np.save(train_info_path, np.array(labels_train))
        print(f"Train Information saved: {train_info_path}")

    if labels_valid:
        valid_info_path = os.path.join(SAVE_ROOT, 'valid', 'Npy', 'IQ', 'Information.npy')
        np.save(valid_info_path, np.array(labels_valid))
        print(f"Valid Information saved: {valid_info_path}")

    end_time = time.time()
    print(f"\n全部完成! 耗时: {end_time - start_time:.2f} 秒")