import os
import time
import numpy as np
from ReceSignalSimple import ReceSignalSimple
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# ================= 配置区域 (与原文件保持一致) =================
CONFIG = {
    'samplingRate': 20e6,
    'num_points': 10000,  # 长度
    'SnrRange': [5, 5],
    'dataNum': 100000,  # 总数量
    'validRate': 0.1,
    'receiverNumRange': [4, 4],
    'emitterNumRange': [1, 1],
    'receiverPosRange': 1e3 * np.array([[0, 5], [0, 5], [0, 0]]),  # z轴后续代码会处理
    'receiverVelRange': [0, 10],
    'emitterPosRange': 1e3 * np.array([[0, 5], [0, 5], [0.12, 0.6]]),
    'emitterVelRange': [0, 10],
    'asynchrony_par_sigma': [0, 0],  # [sigma_t, sigma_f]
    'root_dir': r"D:\Dataset\SignalDataset"
}

# 计算保存路径
SNR_STR = str(CONFIG['SnrRange'])
SAVE_ROOT = os.path.join(CONFIG['root_dir'], "SNR" + SNR_STR)


# ================= 核心工作函数 (用于多进程) =================
def generate_single_sample(args):
    """
    单个样本生成函数，将被分配到不同的CPU核心执行
    """
    idx, mode, conf = args

    # 随机数种子需要在每个进程中独立设置，否则多进程可能生成重复数据
    # 使用 idx 作为种子的一部分确保唯一性
    np.random.seed(int(time.time() * 1000) % 2 ** 32 + idx)

    # --- 1. 参数随机化 (照搬原逻辑) ---
    receiverNum = int(np.random.uniform(conf['receiverNumRange'][0], conf['receiverNumRange'][1] + 1))
    emitterNum = int(np.random.uniform(conf['emitterNumRange'][0], conf['emitterNumRange'][1] + 1))

    # 接收机位置/速度
    receiverPos = np.zeros([receiverNum, 3])
    # 高度一致逻辑
    recePosZ = np.random.uniform(conf['receiverPosRange'][2][0], conf['receiverPosRange'][2][1])
    receiverVel = np.zeros_like(receiverPos)

    for j in range(receiverNum):
        receiverPos[j, 0] = np.random.uniform(conf['receiverPosRange'][0][0], conf['receiverPosRange'][0][1])
        receiverPos[j, 1] = np.random.uniform(conf['receiverPosRange'][1][0], conf['receiverPosRange'][1][1])
        receiverPos[j, 2] = recePosZ

        # 速度
        receVel = np.random.uniform(conf['receiverVelRange'][0], conf['receiverVelRange'][1])
        theta = np.random.uniform(0, np.pi / 2)
        fa = np.random.uniform(0, 2 * np.pi)
        receiverVel[j] = [
            receVel * np.sin(theta) * np.cos(fa),
            receVel * np.sin(theta) * np.sin(fa),
            receVel * np.cos(theta)
        ]

    # todo 如果需要固定位置 (如之前建议的)，在这里覆盖 receiverPos
    receiverPos = np.array([[0,0,0], [3500,0,0], [3500,3500,0], [0,3500,0]])
    receiverVel = np.zeros_like(receiverPos)

    # 发射机位置/速度
    emitterPos = np.zeros([emitterNum, 3])
    emitterVel = np.zeros_like(emitterPos)
    for j in range(emitterNum):
        emitterPos[j, 0] = np.random.uniform(conf['emitterPosRange'][0][0], conf['emitterPosRange'][0][1])
        emitterPos[j, 1] = np.random.uniform(conf['emitterPosRange'][1][0], conf['emitterPosRange'][1][1])
        emitterPos[j, 2] = np.random.uniform(conf['emitterPosRange'][2][0], conf['emitterPosRange'][2][1])

        emitVel = np.random.uniform(conf['emitterVelRange'][0], conf['emitterVelRange'][1])
        theta = np.pi / 2  # 原代码设定
        fa = np.random.uniform(0, 2 * np.pi)
        emitterVel[j] = [
            emitVel * np.sin(theta) * np.cos(fa),
            emitVel * np.sin(theta) * np.sin(fa),
            emitVel * np.cos(theta)
        ]

    # SNR 和 采样时间
    samplingTime = conf['num_points'] / conf['samplingRate']
    Snrs = [np.random.uniform(conf['SnrRange'][0], conf['SnrRange'][1]) for _ in range(emitterNum)]

    # --- 2. 信号生成 (最耗时的部分) ---
    emitSignals, receSignal, deltaTs, deltaFs, fcs = ReceSignalSimple(
        samplingTime, conf['samplingRate'],
        emitterPos, emitterVel,
        receiverPos, receiverVel,
        Snrs, conf['asynchrony_par_sigma']
    )

    # --- 3. 保存信号数据 (直接存 NPY，跳过 CSV) ---
    # 路径结构: output_dir/Npy/IQ/ReceiverX/idx.npy
    base_save_path = os.path.join(SAVE_ROOT, mode, "Npy", "IQ")

    # 保存接收机数据
    for k in range(conf['receiverNumRange'][1]):
        rx_dir = os.path.join(base_save_path, f"Receiver{k}")
        os.makedirs(rx_dir, exist_ok=True)  # 确保目录存在
        save_file = os.path.join(rx_dir, f"{idx}.npy")

        if k < receiverNum:
            # 转换为 complex64 节省空间且速度更快
            sig_data = receSignal[k].astype(np.complex64)
            # 形状调整为 (N, 1) 保持兼容性，或者直接保存一维
            sig_data = sig_data.reshape(-1, 1)
        else:
            # 填充 0
            sig_data = np.zeros((conf['num_points'], 1), dtype=np.complex64)

        np.save(save_file, sig_data)

    # 保存发射机数据 (如果需要)
    for k in range(conf['emitterNumRange'][1]):
        tx_dir = os.path.join(base_save_path, f"Emitter{k}")
        os.makedirs(tx_dir, exist_ok=True)
        save_file = os.path.join(tx_dir, f"{idx}.npy")

        if k < emitterNum:
            # emitSignals 里的结构是 list of [Rx, Samples]，取第一个Rx的作为参考
            sig_data = emitSignals[k][0].astype(np.complex64).reshape(-1, 1)
        else:
            sig_data = np.zeros((conf['num_points'], 1), dtype=np.complex64)
        np.save(save_file, sig_data)

    # --- 4. 构造标签行 (Information) ---
    # 按照原代码的顺序拼接 list
    # 注意：这里我们返回 list，在主进程中统一写入，避免多进程写同一个文件的锁问题

    label = [idx, receiverNum, emitterNum]

    # SNR & FC
    for j in range(conf['emitterNumRange'][1]):
        if j < emitterNum:
            label.extend([Snrs[j], fcs[j]])
        else:
            label.extend([0, 0])

    # 环境参数
    label.extend([
        samplingTime, conf['samplingRate'],
        conf['receiverPosRange'][0][0], conf['receiverPosRange'][0][1],
        conf['receiverPosRange'][1][0], conf['receiverPosRange'][1][1],
        conf['receiverPosRange'][2][0], conf['receiverPosRange'][2][1],
        conf['receiverVelRange'][0], conf['receiverVelRange'][1],
        conf['emitterPosRange'][0][0], conf['emitterPosRange'][0][1],
        conf['emitterPosRange'][1][0], conf['emitterPosRange'][1][1],
        conf['emitterPosRange'][2][0], conf['emitterPosRange'][2][1],
        conf['emitterVelRange'][0], conf['emitterVelRange'][1]
    ])

    # Delay T/F
    for j in range(conf['receiverNumRange'][1]):
        for k in range(conf['emitterNumRange'][1]):
            if j < receiverNum and k < emitterNum:
                label.extend([deltaTs[k][j], deltaFs[k][j]])
            else:
                label.extend([0, 0])

    # Rx Pos/Vel
    for j in range(conf['receiverNumRange'][1]):
        if j < receiverNum:
            label.extend(list(receiverPos[j]) + list(receiverVel[j]))
        else:
            label.extend([0] * 6)

    # Tx Pos/Vel
    for j in range(conf['emitterNumRange'][1]):
        if j < emitterNum:
            label.extend(list(emitterPos[j]) + list(emitterVel[j]))
        else:
            label.extend([0] * 6)

    return label


# ================= 主程序 =================
if __name__ == '__main__':
    # 1. 准备目录
    print(f"数据将生成至: {SAVE_ROOT}")

    # 为了避免路径冲突，先建立基础文件夹
    for mode in ['train', 'valid']:
        path = os.path.join(SAVE_ROOT, mode, "Npy", "IQ")
        if not os.path.exists(path):
            os.makedirs(path)
        # 清理旧数据 (可选，建议手动清理以防误删)
        # DataIO.DelFile(path)

    # 2. 生成任务列表
    train_num = int(CONFIG['dataNum'] * (1 - CONFIG['validRate']))
    valid_num = int(CONFIG['dataNum'] * CONFIG['validRate'])

    tasks = []
    # Train tasks
    for i in range(train_num):
        tasks.append((i, 'train', CONFIG))
    # Valid tasks
    for i in range(valid_num):
        tasks.append((i, 'valid', CONFIG))

    print(f"开始生成 {len(tasks)} 条数据，使用 {os.cpu_count()} 个核心...")

    # 3. 多进程并行执行
    # max_workers=None 默认使用所有CPU核心
    labels_train = []
    labels_valid = []

    start_time = time.time()

    with ProcessPoolExecutor() as executor:
        # 使用 tqdm 显示进度
        results = list(tqdm(executor.map(generate_single_sample, tasks), total=len(tasks)))

    # 4. 整理标签并保存 (Information.npy / Information.csv)
    print("\n正在保存标签文件...")

    for res, task in zip(results, tasks):
        idx, mode, _ = task
        if mode == 'train':
            labels_train.append(res)
        else:
            labels_valid.append(res)

    # 保存 Train 标签
    if labels_train:
        train_info_path = os.path.join(SAVE_ROOT, 'train', 'Npy', 'IQ', 'Information.npy')
        # 你的 Dataset Loader 读取的是 .npy 格式的 Information
        np.save(train_info_path, np.array(labels_train))
        print(f"Train Information saved: {train_info_path}")

    # 保存 Valid 标签
    if labels_valid:
        valid_info_path = os.path.join(SAVE_ROOT, 'valid', 'Npy', 'IQ', 'Information.npy')
        np.save(valid_info_path, np.array(labels_valid))
        print(f"Valid Information saved: {valid_info_path}")

    end_time = time.time()
    print(f"\n全部完成! 耗时: {end_time - start_time:.2f} 秒")
    print(f"平均速度: {len(tasks) / (end_time - start_time):.2f} 样本/秒")