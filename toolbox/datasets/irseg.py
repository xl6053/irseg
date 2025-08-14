# irseg.py (最终正确版本)

import os
from PIL import Image
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
# 我们仍然需要在线的裁剪、缩放等增强，但不再需要在线的翻转
from toolbox.datasets.augmentations import Compose, ColorJitter, RandomCrop, RandomScale

class IRSeg(data.Dataset):
    def __init__(self, cfg, mode='trainval', do_aug=True):
        assert mode in ['train', 'val', 'trainval', 'test', 'test_day', 'test_night'], f'{mode} not support.'
        self.mode = mode
        self.do_aug = do_aug

        self.im_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.dp_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.449, 0.449, 0.449], [0.226, 0.226, 0.226]),
        ])

        self.root = cfg['root']
        self.n_classes = cfg['n_classes']

        scale_range = tuple(float(i) for i in cfg['scales_range'].split(' '))
        crop_size = tuple(int(i) for i in cfg['crop_size'].split(' '))

        # 关键：在线数据增强中不再进行翻转
        self.aug = Compose([
            ColorJitter(
                brightness=cfg['brightness'],
                contrast=cfg['contrast'],
                saturation=cfg['saturation']),
            RandomScale(scale_range),
            RandomCrop(crop_size, pad_if_needed=True)
        ])

        if cfg['class_weight'] == 'enet':
            self.class_weight = np.array(
                [1.5105, 16.6591, 29.4238, 34.6315, 40.0845, 41.4357, 47.9794, 45.3725, 44.9000])
        # ... (其他 class_weight 逻辑保持不变) ...

        # 读取包含原始和 _flip 两种文件名的 txt 文件
        with open(os.path.join(self.root, f'{mode}.txt'), 'r') as f:
            self.infos = f.readlines()

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, index):
        # image_path 现在可能是 "00085D" 或者 "00085D_flip"
        image_path = self.infos[index].strip()
        
        full_image_path = os.path.join(self.root, 'images', image_path + '.png')
        
        try:
            rgba_image = Image.open(full_image_path).convert('RGBA')
        except FileNotFoundError:
            print(f"致命错误：找不到图像文件: {full_image_path}")
            print("请确认您是否已经成功运行 flip.py。")
            raise

        # 实时分离通道
        image = Image.new("RGB", rgba_image.size)
        image.paste(rgba_image, (0, 0), rgba_image)
        thermal_channel = rgba_image.split()[3]
        depth = Image.merge("RGB", (thermal_channel, thermal_channel, thermal_channel))
        
        # 加载对应的标签 (假设也被 flip.py 处理了)
        label = Image.open(os.path.join(self.root, 'labels', image_path + '.png'))
        bound = Image.open(os.path.join(self.root, 'bound', image_path + '.png'))
        edge = Image.open(os.path.join(self.root, 'edge', image_path + '.png'))
        binary_label = Image.open(os.path.join(self.root, 'binary_labels', image_path + '.png'))

        sample = {
            'image': image, 'depth': depth, 'label': label,
            'bound': bound, 'edge': edge, 'binary_label': binary_label,
        }

        # 在线增强 (裁剪、缩放、色彩等)，但不再进行翻转
        if self.mode in ['train', 'trainval'] and self.do_aug:
            sample = self.aug(sample)

        # 转换为Tensor
        sample['image'] = self.im_to_tensor(sample['image'])
        sample['depth'] = self.dp_to_tensor(sample['depth'])
        sample['label'] = torch.from_numpy(np.asarray(sample['label'], dtype=np.int64)).long()
        sample['edge'] = torch.from_numpy(np.asarray(sample['edge'], dtype=np.int64)).long()
        sample['bound'] = torch.from_numpy(np.asarray(sample['bound'], dtype=np.int64) / 255.).long()
        sample['binary_label'] = torch.from_numpy(np.asarray(sample['binary_label'], dtype=np.int64) / 255.).long()
        sample['label_path'] = image_path.strip().split('/')[-1] + '.png'
        return sample

    # ... (cmap 属性保持不变) ...

    @property
    def cmap(self):
        return [
            (0, 0, 0), (64, 0, 128), (64, 64, 0), (0, 128, 192), (0, 0, 192),
            (128, 128, 0), (64, 64, 128), (192, 128, 128), (192, 64, 0),
        ]