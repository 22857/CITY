import h5py

H5_PATH = "/root/autodl-tmp/merged_dataset_512_3d_fast.h5"

with h5py.File(H5_PATH, 'r') as f:
    print(f"--- H5 文件诊断报告 ---")
    for key in f.keys():
        ds = f[key]
        print(f"\n数据集: {key}")
        print(f"  形状 (Shape): {ds.shape}")
        print(f"  分块 (Chunks): {ds.chunks}")
        print(f"  压缩 (Compression): {ds.compression}")

        if ds.chunks is None:
            print(f"  ⚠️ 警告: 该数据集未设置分块！随机读取性能将极其低下。")
        else:
            print(f"  ✅ 已设置分块，大小为: {ds.chunks}")