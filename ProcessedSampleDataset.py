import os
import sys

# from matplotlib import pyplot as plt

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DataStructure')
sys.path.append(base_dir)
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)
import numpy as np
import DataIO


# 纯模拟数据(tf版本)
# 接收机文件夹需包含'Receiver'
class FileBatchLoader:
    def __init__(self, dataFileRoot):

        # 保存根目录
        self.dataFileRoot = dataFileRoot

        # 加载文件夹名称
        self.receDirs = DataIO.GetDirName(dataFileRoot)
        # 接收机文件夹
        self.receDirs = [n for n in self.receDirs if 'Receiver' in n] if self.receDirs else []
        self.receDataFileNames = []
        for dir in self.receDirs:
            temp = DataIO.GetFileName(dataFileRoot + '/' + dir)
            # 将文件名按数字部分排序
            temp.sort(key=lambda x: int(x[:-4]))
            self.receDataFileNames.append(temp)

        # 加载文件夹名称
        self.emitDirs = DataIO.GetDirName(dataFileRoot)
        # 发射机文件夹
        self.emitDirs = [n for n in self.emitDirs if 'Emitter' in n] if self.emitDirs else []
        self.emitDataFileNames = []
        for dir in self.emitDirs:
            temp = DataIO.GetFileName(dataFileRoot + '/' + dir)
            # 将文件名按指定方式排序
            temp.sort(key=lambda x: int(x[:-4]))
            self.emitDataFileNames.append(temp)

        # 加载标签文件
        inforPath = self.dataFileRoot + '/Information.npy'
        if os.path.exists(inforPath):
            temp = np.load(inforPath)
            self.information = np.zeros(temp.shape)
            for i in range(self.information.shape[0]):
                self.information[i] = np.array([float(x) for x in temp[i]])

        self.receiverNum = self.receDirs.__len__()
        self.emitterNum = self.emitDirs.__len__()

        # 数据个数
        self.dataNum = self.information.shape[0] if os.path.exists(inforPath) else 0
        # 已经使用的数据编号
        self.usedDataNum = 0
        # 数据集打乱的编号
        self.dataPermutation = np.random.permutation(self.dataNum)

        print('find {} data'.format(self.receDataFileNames[0].__len__() if os.path.exists(inforPath) else 0))

    # 用于加载发射机的 IQ 数据
    def GetEmitSignalIQData(self, batchSize, singleDataShape, batchIndex, isNormal=False):
        data = np.zeros((batchSize,) + singleDataShape, dtype=complex)
        for k in range(data.shape[0]):
            for m in range(data.shape[1]):
                if m >= self.emitDirs.__len__():
                    continue
                data[k, m, :, 0] = np.squeeze(np.load(
                    self.dataFileRoot + '/' + self.emitDirs[m] + '/' + self.emitDataFileNames[m][batchIndex[k]])[
                                   :data.shape[2]])
                if isNormal and np.abs(data[k, :, :, m]).max() != 0:
                    data[k, m, :, 0] /= np.abs(data[k, m, :, 0]).max()
        return data

    # 用于加载接收机的 IQ 数据
    def GetSignalIQData(self, batchSize, singleDataShape, batchIndex, isNormal=False):
        data = np.zeros((batchSize,) + singleDataShape, dtype=complex)
        for k in range(data.shape[0]):
            for m in range(data.shape[1]):
                if m >= self.receDirs.__len__():
                    continue
                # try:
                #     data[k, m, :, 0] = np.load(
                #         self.dataFileRoot + '/' + self.receDirs[m] + '/' + self.receDataFileNames[m][batchIndex[k]])[
                #                        :data.shape[2]]
                # except:
                #     print('Waring: the required data is unsupported!')
                #     exit(-1)

                try:
                    # --- 修改开始 ---
                    # 加载原始数据
                    raw_data = np.load(
                        self.dataFileRoot + '/' + self.receDirs[m] + '/' + self.receDataFileNames[m][batchIndex[k]])
                    # 截取所需长度
                    raw_data = raw_data[:data.shape[2]]
                    # 强制展平为 1D，兼容 (N, 1) 和 (N,)
                    data[k, m, :, 0] = raw_data.flatten()
                    # --- 修改结束 ---
                except Exception as e:
                    # 打印具体错误信息，方便调试
                    print(f'Error loading file: {e}')
                    print('Waring: the required data is unsupported!')
                    exit(-1)


                if isNormal and np.abs(data[k, m, :, 0]).max() != 0:
                    data[k, m, :, 0] /= np.abs(data[k, m, :, 0]).max()
        return data

    def GetSignalIQCWTData(self, batchSize, singleDataShape, batchIndex, isNormal=False):
        data = np.zeros((batchSize,) + singleDataShape, dtype=complex)
        for k in range(data.shape[0]):
            for m in range(data.shape[3]):
                if m >= self.receDirs.__len__():
                    continue
                data[k, :, :, m] = np.load(
                    self.dataFileRoot + '/' + self.receDirs[m] + '/' + self.receDataFileNames[m][batchIndex[k]])
                if isNormal and np.abs(data[k, :, :, m]).max() != 0:
                    data[k, :, :, m] /= np.abs(data[k, :, :, m]).max()
        return data

    # 获取接收机的位置信息和速度，并进行必要的归一化处理
    def GetRecPosVel(self, batchSize, singleDataShape, batchIndex, isNormal=False):
        posRange = None
        velRange = None
        if isNormal:
            posRange = np.hstack((self.information[batchIndex, 5 + 2 * self.emitterNum:5 + 2 * self.emitterNum + 6].min(
                1).reshape([batchSize, 1]),
                                  self.information[batchIndex, 5 + 2 * self.emitterNum:5 + 2 * self.emitterNum + 6].max(
                                      1).reshape([batchSize, 1])))
            velRange = np.hstack((self.information[batchIndex, 5 + 2 * self.emitterNum + 6].reshape([batchSize, 1]),
                                  self.information[batchIndex, 5 + 2 * self.emitterNum + 7].reshape([batchSize, 1])))

        data = np.zeros((batchSize,) + singleDataShape)
        for k in range(data.shape[1]):
            data[:, k] = self.information[batchIndex, -(self.receDirs.__len__() * 6 + self.emitDirs.__len__() * 6) + k]

            if k % 6 < 3 and isNormal:
                data[:, k] /= posRange[:, 1]
            if k % 6 >= 3 and isNormal:
                data[:, k] /= velRange[:, 1]

        return data

    def GetDelayOfTimeFre(self, batchSize, singleData2Shape, batchIndex):
        data = np.zeros((batchSize,) + singleData2Shape)
        for k in range(data.shape[1]):
            data[:, k] = self.information[batchIndex, -(
                    self.receDirs.__len__() * 6 + self.emitDirs.__len__() * 6 + 2 * self.receDirs.__len__() * self.emitDirs.__len__()) + k]
        return data

    # 获取发射机的位置信息和速度，并进行必要的归一化处理
    def GetEmiPosVel(self, batchSize, labelShape, batchIndex, isNormal=False):

        posRange = None
        velRange = None
        if isNormal:
            posRange = np.hstack((
                self.information[batchIndex, 5 + 2 * self.emitterNum + 8:5 + 2 * self.emitterNum + 8 + 6].min(
                    1).reshape([batchSize, 1]),
                self.information[batchIndex, 5 + 2 * self.emitterNum + 8:5 + 2 * self.emitterNum + 8 + 6].max(
                    1).reshape([batchSize, 1])))
            velRange = np.hstack((self.information[batchIndex, 5 + 2 * self.emitterNum + 8 + 6].reshape([batchSize, 1]),
                                  self.information[batchIndex, 5 + 2 * self.emitterNum + 8 + 7].reshape(
                                      [batchSize, 1])))

        data = np.zeros((batchSize,) + labelShape)
        for k in range(data.shape[1]):
            data[:, k] = self.information[batchIndex, -(self.emitDirs.__len__() * 6) + k]

            if k % 6 < 3 and isNormal:
                data[:, k] /= posRange[:, 1]
            if k % 6 >= 3 and isNormal:
                data[:, k] /= velRange[:, 1]

        return data

    def GetBatchIndex(self, batchSize, isTrainData):
        if isTrainData:
            batchIndex = self.dataPermutation[self.usedDataNum:self.usedDataNum + batchSize]

            self.usedDataNum += batchSize
            if self.usedDataNum == self.dataNum:
                self.usedDataNum = 0
                # 所有样本循环一次后再次打乱数据集
                self.dataPermutation = np.random.permutation(self.dataNum)
        else:
            batchIndex = [x for x in
                          range(self.usedDataNum, self.usedDataNum + batchSize)]

            self.usedDataNum += batchSize
            if self.usedDataNum == self.dataNum:
                self.usedDataNum = 0
        return batchIndex

    # 用于端到端学习。输入是信号的时频图（CWT），模型需要自己从图像中提取特征来定位发射机
    # [IQ信号] -> [CWT变换] -> [取绝对值(图像)] + [接收机坐标] ==> 神经网络 ==> [预测发射机坐标]
    def GetNextBatch(self, batchSize, singleData1Shape, singleData2Shape, labelShape, isTrainData=True, isNormal=True):

        batchIndex = self.GetBatchIndex(batchSize, isTrainData)

        # 加载每个batch数据
        data = []
        data.append(np.abs(self.GetSignalIQCWTData(batchSize, singleData1Shape, batchIndex, isNormal)))
        data.append(self.GetRecPosVel(batchSize, singleData2Shape, batchIndex, isNormal))
        label = self.GetEmiPosVel(batchSize, labelShape, batchIndex, isNormal)

        return data, label

    # 用于基于物理参数的学习。输入是已经计算好的物理差值（TDOA、FDOA、相对坐标），模型基于这些数学特征来定位发射机
    # [元数据/Information.npy] -> [计算 TDOA/FDOA] + [计算相对坐标] ==> 神经网络 ==> [预测发射机坐标]
    def GetNextBatch1(self, batchSize, singleDataShape, labelShape, isTrainData=True):

        batchIndex = self.GetBatchIndex(batchSize, isTrainData)

        data1 = self.GetDelayOfTimeFre(batchSize, (6,), batchIndex)
        data2 = self.GetRecPosVel(batchSize, (18,), batchIndex)
        data = np.zeros((batchSize,) + singleDataShape)

        # delta t
        data[:, 0] = (data1[:, 2] - data1[:, 0])
        data[:, 1] = (data1[:, 4] - data1[:, 0])
        # delta f
        data[:, 2] = (data1[:, 3] - data1[:, 1])
        data[:, 3] = (data1[:, 5] - data1[:, 1])

        # delta px
        data[:, 4] = data2[:, 6] - data2[:, 0]
        data[:, 5] = data2[:, 12] - data2[:, 0]
        # delta py
        data[:, 6] = data2[:, 7] - data2[:, 1]
        data[:, 7] = data2[:, 13] - data2[:, 1]
        # delta pz
        data[:, 8] = data2[:, 8] - data2[:, 2]
        data[:, 9] = data2[:, 14] - data2[:, 2]

        # delta vx
        data[:, 10] = data2[:, 9] - data2[:, 3]
        data[:, 11] = data2[:, 15] - data2[:, 3]
        # delta vy
        data[:, 12] = data2[:, 10] - data2[:, 4]
        data[:, 13] = data2[:, 16] - data2[:, 4]
        # delta vz
        data[:, 14] = data2[:, 11] - data2[:, 5]
        data[:, 15] = data2[:, 17] - data2[:, 5]

        label = self.GetEmiPosVel(batchSize, labelShape, batchIndex)

        return data, label

    # 把复数信号喂给网络，让网络自己去学相位关系并输出坐标
    # 原始 IQ 信号 -> 绝对坐标
    def GetNextBatch2(self, batchSize, singleData1Shape, singleData2Shape, labelShape, isTrainData=True):

        batchIndex = self.GetBatchIndex(batchSize, isTrainData)

        # 加载每个batch数据
        data = []
        data.append(self.GetSignalIQData(batchSize, singleData1Shape, batchIndex))
        data.append(self.GetRecPosVel(batchSize, singleData2Shape, batchIndex))
        label = self.GetEmiPosVel(batchSize, labelShape, batchIndex)

        return data, label

    # 参数估计 对输入数据进行了特征重排，并且目标不再是坐标，而是物理参数
    # 拆解 IQ 实虚部 ->中间参数 (TDOA/FDOA)
    def GetNextBatch3(self, batchSize, singleData1Shape, singleData2Shape, labelShape, isTrainData=True):

        batchIndex = self.GetBatchIndex(batchSize, isTrainData)

        # 加载每个batch数据
        data = []
        temp = np.zeros((batchSize,) + singleData1Shape)
        temp1 = self.GetSignalIQData(batchSize, (3, 800, 1), batchIndex)
        temp[:, 0] = temp1[:, 0].real
        temp[:, 1] = temp1[:, 0].imag
        temp[:, 2] = temp1[:, 1].real
        temp[:, 3] = temp1[:, 1].imag
        temp[:, 4] = temp1[:, 2].real
        temp[:, 5] = temp1[:, 2].imag

        data.append(temp)
        data.append(self.GetRecPosVel(batchSize, singleData2Shape, batchIndex))

        label = np.zeros((batchSize,) + labelShape)
        data2 = self.GetDelayOfTimeFre(batchSize, (6,), batchIndex)
        # delta t
        label[:, 0] = (data2[:, 2] - data2[:, 0])
        label[:, 1] = (data2[:, 4] - data2[:, 0])
        # delta f
        label[:, 2] = (data2[:, 3] - data2[:, 1])
        label[:, 3] = (data2[:, 5] - data2[:, 1])

        return data, label

    # 时频图 (CWT) -> 中间参数 (TDOA/FDOA)
    def GetNextBatch4(self, batchSize, singleData1Shape, singleData2Shape, labelShape, isTrainData=True):

        batchIndex = self.GetBatchIndex(batchSize, isTrainData)

        # 加载每个batch数据
        data = []
        data.append(np.abs(self.GetSignalIQCWTData(batchSize, singleData1Shape, batchIndex)))
        data.append(self.GetRecPosVel(batchSize, singleData2Shape, batchIndex))

        label = np.zeros((batchSize,) + labelShape)
        data2 = self.GetDelayOfTimeFre(batchSize, (6,), batchIndex)
        # delta t
        label[:, 0] = (data2[:, 2] - data2[:, 0])
        label[:, 1] = (data2[:, 4] - data2[:, 0])
        # delta f
        label[:, 2] = (data2[:, 3] - data2[:, 1])
        label[:, 3] = (data2[:, 5] - data2[:, 1])

        return data, label


# 纯模拟数据
class ProcessedSampleDataset:
    def __init__(self, root_path: str, receiverNum=3, emitterNum=1, time_scale=None, transform=None,
                 time_fre_trans=None, fre_scale=None, isNormal=False, coordDim=3, dataType='IQ'):

        root_path += "/Npy/" + dataType
        self.dataType = dataType
        self.dataset = FileBatchLoader(root_path)
        self.root_path = root_path
        self.transform = transform
        self.time_fre_trans = time_fre_trans
        self.fre_scale = fre_scale
        self.time_scale = time_scale
        self.isNormal = isNormal
        self.coordDim = coordDim
        # 默认使用 time_scale 兜底，防止检测失败
        self.real_data_len = self.time_scale if self.time_scale else 192
        try:
            # 检查是否存在接收机文件夹和数据文件
            if len(self.dataset.receDirs) > 0 and len(self.dataset.receDataFileNames) > 0:
                if len(self.dataset.receDataFileNames[0]) > 0:
                    # 构造第一个文件的完整路径
                    # 路径结构: 根目录/Receiver0/0.npy
                    first_dir_name = self.dataset.receDirs[0]
                    first_file_name = self.dataset.receDataFileNames[0][0]

                    full_path = os.path.join(self.dataset.dataFileRoot, first_dir_name, first_file_name)

                    # 加载并获取长度
                    if os.path.exists(full_path):
                        temp_data = np.load(full_path)
                        self.real_data_len = temp_data.shape[0]
                        print(f"[{self.dataType}] Auto-detected signal length: {self.real_data_len}")
                    else:
                        print(f"Warning: File not found for length detection: {full_path}")
        except Exception as e:
            print(f"Warning: Failed to auto-detect signal length. Using default {self.real_data_len}. Error: {e}")
        # ===================================================================

    def getDataNum(self):
        return self.dataset.dataNum

    def getData(self, item, isGetFsFc=False):

        data1 = None
        if self.dataType == 'IQ':
            data1 = self.dataset.GetSignalIQData(1, (self.dataset.receiverNum, self.real_data_len, 1), [item],
                                                 self.isNormal)
        elif self.dataType == 'IQCWT':
            data1 = self.dataset.GetSignalIQCWTData(1, (self.fre_scale, self.time_scale, self.dataset.receiverNum),
                                                    [item], self.isNormal)
            data1 = np.abs(data1.reshape([1, self.dataset.receiverNum, self.fre_scale, self.time_scale]))
        data1 = np.squeeze(data1)

        if isGetFsFc:
            data2 = np.zeros(6 * self.dataset.receiverNum + 2)
            data2[:-2] = self.dataset.GetRecPosVel(1, (6 * self.dataset.receiverNum,), [item], self.isNormal)
            data2[-2] = self.dataset.information[item, 2 * self.dataset.emitterNum + 4]
            data2[-1] = self.dataset.information[item, 4]
        else:
            data2 = self.dataset.GetRecPosVel(1, (6 * self.dataset.receiverNum,), [item], self.isNormal)
        data2 = np.squeeze(data2)

        if self.time_fre_trans is not None and self.dataType == 'IQ':
            sampleTime = self.dataset.information[item, 2 * self.dataset.emitterNum + 3]
            data1 = self.time_fre_trans(data1, self.fre_scale, sampleTime)
            if self.isNormal:
                for i in range(data1.shape[0]):
                    if data1[i].max() != 0:
                        data1[i] = data1[i] / data1[i].max()

        if self.transform is not None:
            data1, data2[:6 * self.dataset.receiverNum] = self.transform(data1, data2[:6 * self.dataset.receiverNum])

        temp = self.dataset.GetEmiPosVel(1, (6 * self.dataset.emitterNum,), [item], isNormal=self.isNormal)
        label = np.zeros([self.dataset.emitterNum, self.coordDim + 1])
        for i in range(self.dataset.emitterNum):
            label[i, :self.coordDim] = temp[0, 6 * i: 6 * i + self.coordDim]
            # confidence
            if np.sum(label[i]) == 0:
                label[i, -1] = 0
            else:
                label[i, -1] = 1

        # sort
        distence = np.sum(np.square(label), 1)
        sort_index = np.argsort(distence)
        label = label[sort_index]

        if self.isNormal:
            if isGetFsFc:
                data2[-1] /= 161975000
                data2[-2] /= 19200

        return [data1, data2], label.flatten()
