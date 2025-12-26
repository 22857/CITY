import torch
from torch.utils.data import Dataset
import h5py


class PhysicsGuidedHDF5Dataset(Dataset):
    def __init__(self, h5_path):
        """
        基于 HDF5 的高效 Dataset
        """
        self.h5_path = h5_path
        self.h5_file = None  # 句柄初始化为空 (懒加载)
        self.dataset_len = 0

        # 预先打开一次只为了获取长度，然后立刻关闭
        # 这样不会占用文件锁
        try:
            with h5py.File(self.h5_path, 'r') as f:
                self.dataset_len = len(f['coord'])
        except Exception as e:
            print(f"【错误】无法打开 HDF5 文件: {self.h5_path}")
            print(f"原因: {e}")
            self.dataset_len = 0

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        # 【关键】懒加载：确保每个 Worker 进程有自己的文件句柄
        # 如果在 __init__ 打开，多进程会死锁
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')

        # 直接切片读取
        # HDF5 引擎会自动优化 IO，只读取所需的那一条数据
        # 此时读出来的是 float16 的 numpy array
        iq_np = self.h5_file['iq'][idx]
        map_np = self.h5_file['heatmap'][idx]
        mask_np = self.h5_file['mask'][idx]
        coord_np = self.h5_file['coord'][idx]

        # 转为 Tensor (并升级为 Float32 以供网络计算)
        # 这一步在 CPU 上进行，非常快
        return (
            torch.from_numpy(iq_np).float(),
            torch.from_numpy(map_np).float(),
            torch.from_numpy(coord_np).float(),
            torch.from_numpy(mask_np).float()
        )

    def __del__(self):
        # 析构时安全关闭句柄
        if self.h5_file is not None:
            self.h5_file.close()