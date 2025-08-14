import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import numba
import shutil
import time

@numba.jit(nopython=True)
def generate_boundary_core(padmap, h, w):
    """
    使用numba加速的核心边界生成函数。
    在一个5x5的窗口内，如果存在两种或以上的类别，则中心点被视为边界。
    """
    bound = np.zeros((h, w), dtype=np.uint8)
    for hh in range(h):
        for ww in range(w):
            slidewindow = padmap[hh:hh+5, ww:ww+5]
            # np.unique 在numba中这样使用
            unique_classes = np.unique(slidewindow.flatten())
            if len(unique_classes) >= 2:
                bound[hh, ww] = 255
    return bound

def generate_edge_map(label_image):
    """
    使用OpenCV的Canny算法生成边缘图。
    """
    label_uint8 = label_image.astype(np.uint8)
    # Canny边缘检测对单通道图像进行操作
    if label_uint8.ndim == 3:
        label_uint8 = cv2.cvtColor(label_uint8, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(label_uint8, threshold1=1, threshold2=255)
    return edge


def preprocess_labels(root_dir: Path):
    """
    一站式处理标签，生成所有训练所需的辅助标签（binary, bound, edge）。
    """
    # 1. 基于传入的 root_dir 参数来定义所有子路径
    original_labels_dir = root_dir / 'labels'
    binary_labels_dir = root_dir / 'binary_labels'
    boundary_dir = root_dir / 'bound'
    edge_dir = root_dir / 'edge'

    # 2. 安全检查与备份
    if not original_labels_dir.exists():
        print(f"错误: 原始标签目录不存在! -> {original_labels_dir}")
        return

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = root_dir / f'labels_backup_{timestamp}'
    print("="*60)
    print("⚠️ 安全警告：本脚本将开始处理标签文件。")
    try:
        if not backup_dir.exists():
             print(f"正在创建备份: '{original_labels_dir.name}' -> '{backup_dir.name}' ...")
             shutil.copytree(original_labels_dir, backup_dir)
             print("备份成功！原始数据已保存。")
        else:
             print("备份目录已存在，跳过备份。")
    except Exception as e:
        print(f"错误：创建备份失败! -> {e}")
        print("为了数据安全，程序已终止。")
        return
    print("="*60 + "\n")

    # 3. 创建输出目录
    print("正在创建输出目录...")
    binary_labels_dir.mkdir(parents=True, exist_ok=True)
    boundary_dir.mkdir(parents=True, exist_ok=True)
    edge_dir.mkdir(parents=True, exist_ok=True)
    print("输出目录准备就绪。\n")

    # 4. 查找所有需要处理的标签文件
    files_to_process = list(original_labels_dir.glob('*.png'))
    if not files_to_process:
        print(f"警告: 在 {original_labels_dir} 中未找到任何 .png 文件。")
        return

    # 5. 开始处理
    print(f"开始生成 binary_labels, bound, edge 文件...")
    pbar_gen = tqdm(files_to_process, desc="生成辅助GT文件")
    for file_path in pbar_gen:
        # 以灰度模式读取标签，确保是单通道
        label_1ch = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if label_1ch is None:
            print(f"\n警告：无法读取文件 {file_path}，跳过。")
            continue

        # --- 生成 Binary Label ---
        # 类别 > 0 的都视为前景 (255)
        binary_label = np.zeros_like(label_1ch, dtype=np.uint8)
        binary_label[label_1ch > 0] = 255
        cv2.imwrite(str(binary_labels_dir / file_path.name), binary_label)

        # --- 生成 Boundary Label ---
        h, w = label_1ch.shape
        # 对标签图进行边缘填充，以便滑动窗口可以处理图像边缘
        padmap = np.pad(label_1ch, pad_width=2, mode='constant', constant_values=0)
        bound = generate_boundary_core(padmap, h, w)
        cv2.imwrite(str(boundary_dir / file_path.name), bound)

        # --- 生成 Edge Label ---
        edge = generate_edge_map(label_1ch)
        cv2.imwrite(str(edge_dir / file_path.name), edge)

    print("\n所有标签预处理任务完成！")
    print(f"新的标签文件已保存在以下目录中:")
    print(f"- {binary_labels_dir}")
    print(f"- {boundary_dir}")
    print(f"- {edge_dir}")


# ===================================================================
# 执行脚本的主入口
# ===================================================================
if __name__ == '__main__':
    # --- 1. 在这里配置您的数据集根目录 ---
    # 请确保路径字符串前有 'r'，以防止Windows下的路径转义问题
    dataset_root = Path(r'D:\LASNet-main\dataset') # <--- !!! 请修改这里 !!!
    
    # 2. 检查路径是否存在
    if not dataset_root.exists():
        print(f"错误：配置的数据集根目录不存在: {dataset_root}")
        print("请打开脚本文件，修改 'dataset_root' 变量为您本地的正确路径。")
    else:
        # 3. 将配置好的路径作为参数，调用处理函数
        preprocess_labels(root_dir=dataset_root)