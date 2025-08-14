# flip.py (增强版)

from PIL import Image
import os
from tqdm import tqdm # 引入tqdm来显示进度条

# --- 1. 定义所有需要处理的文件夹路径 ---
root_dir = 'D:/LASNet-main/dataset/' #  请确保这是您数据集的正确根目录
list_file = os.path.join(root_dir, 'train_original_only.txt') # 确保您有一个只包含原始文件名的txt

# 定义输入和输出目录
dir_map = {
    'images': 'images',
    'labels': 'labels',
    'bound': 'bound',
    'edge': 'edge',
    'binary_labels': 'binary_labels'
}

# --- 2. 检查路径是否存在 ---
for key, dir_name in dir_map.items():
    path = os.path.join(root_dir, dir_name)
    if not os.path.exists(path):
        print(f"警告：找不到文件夹 {path}，将跳过处理。")
        # 如果这个文件夹是必须的，您可能需要取消下面的注释来报错
        # raise FileNotFoundError(f"Directory not found: {path}")

# --- 3. 读取原始文件名列表 ---
if not os.path.exists(list_file):
    raise FileNotFoundError(f"错误：找不到原始文件名列表 {list_file}！请参照我们之前的讨论创建这个文件。")

files = []
with open(list_file, 'r') as reader:
    for line in reader.readlines():
        files.append(line.strip('\n'))

print(f"找到 {len(files)} 个原始文件进行翻转处理...")

# --- 4. 循环处理所有文件和所有文件夹 ---
for f in tqdm(files, desc="Processing files"):
    # 基础文件名，例如 "00818N"
    base_name = f.split('.')[0]
    
    # 为每个目录下的对应文件生成翻转版本
    for key, dir_name in dir_map.items():
        # 构建原始文件和翻转后文件的完整路径
        original_file_path = os.path.join(root_dir, dir_name, base_name + '.png')
        flipped_file_path = os.path.join(root_dir, dir_name, base_name + '_flip.png')

        # 检查原始文件是否存在
        if os.path.exists(original_file_path):
            try:
                # 打开，水平翻转，然后保存
                Image.open(original_file_path).transpose(Image.FLIP_LEFT_RIGHT).save(flipped_file_path, 'PNG')
            except Exception as e:
                print(f"处理文件 {original_file_path} 时出错: {e}")
        else:
            print(f"警告：在 {dir_name} 文件夹中找不到原始文件 {original_file_path}，跳过。")

print("所有文件的离线翻转增强已完成！")