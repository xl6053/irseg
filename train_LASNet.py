# train_LASNet.py (完整修正版)

import os
import shutil
import json
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from toolbox import get_dataset
from toolbox.optim.Ranger import Ranger
from toolbox import get_logger
from toolbox import get_model
from toolbox import averageMeter, runningScore
from toolbox import save_ckpt, load_ckpt
from toolbox.datasets.irseg import IRSeg
from toolbox.datasets.pst900 import PSTSeg
from toolbox.losses import lovasz_softmax
from losses_deformation import AlignmentLoss

class eeemodelLoss(nn.Module):
    def __init__(self, class_weight=None, ignore_index=-100, reduction='mean'):
        super(eeemodelLoss, self).__init__()
        self.class_weight_semantic = torch.from_numpy(np.array(
            [1.5105, 16.6591, 29.4238, 34.6315, 40.0845, 41.4357, 47.9794, 45.3725, 44.9000])).float()
        self.class_weight_binary = torch.from_numpy(np.array([1.5121, 10.2388])).float()
        self.class_weight_boundary = torch.from_numpy(np.array([1.4459, 23.7228])).float()
        self.class_weight = class_weight
        self.cross_entropy = nn.CrossEntropyLoss()
        self.semantic_loss = nn.CrossEntropyLoss(weight=self.class_weight_semantic)
        self.binary_loss = nn.CrossEntropyLoss(weight=self.class_weight_binary)
        self.boundary_loss = nn.CrossEntropyLoss(weight=self.class_weight_boundary)

    def forward(self, inputs, targets):
        semantic_gt, binary_gt, boundary_gt = targets
        semantic_out, semantic_out_2, sal_out, edge_out = inputs
        loss1 = self.semantic_loss(semantic_out, semantic_gt)
        loss2 = lovasz_softmax(F.softmax(semantic_out, dim=1), semantic_gt, ignore=255)
        loss3 = self.semantic_loss(semantic_out_2, semantic_gt)
        loss4 = self.binary_loss(sal_out, binary_gt)
        loss5 = self.boundary_loss(edge_out, boundary_gt)
        loss = loss1 + loss2 + loss3 + 0.5*loss4 + loss5
        return loss

def run(args):
    with open(args.config, 'r') as fp:
        cfg = json.load(fp)

    # 1. 首先，初始化日志记录器
    logdir = f'run/{time.strftime("%Y-%m-%d-%H-%M")}-{cfg["dataset"]}-{cfg["model_name"]}-'
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    shutil.copy(args.config, logdir)
    logger = get_logger(logdir)
    logger.info(f'Conf | use logdir {logdir}')

    # 2. 然后，进行设备选择（现在可以使用logger了）
    if args.cpu or not torch.cuda.is_available():
        device = torch.device('cpu')
        logger.info('--- Using CPU for training ---')
    else:
        torch.cuda.set_device(args.cuda)
        device = torch.device(f'cuda:{args.cuda}')
        logger.info(f'--- Using GPU {args.cuda} for training ---')

    # 3. 正常加载模型、数据等
    model = get_model(cfg)
    model.to(device)

    trainset, _, testset = get_dataset(cfg)
    train_loader = DataLoader(trainset, batch_size=cfg['ims_per_gpu'], shuffle=True, num_workers=cfg['num_workers'],
                              pin_memory=True)
    test_loader = DataLoader(testset, batch_size=cfg['ims_per_gpu'], shuffle=False, num_workers=cfg['num_workers'],
                             pin_memory=True)

    params_list = model.parameters()
    optimizer = Ranger(params_list, lr=cfg['lr_start'], weight_decay=cfg['weight_decay'])
    scheduler = LambdaLR(optimizer, lr_lambda=lambda ep: (1 - ep / cfg['epochs']) ** 0.9)

    train_criterion_seg = eeemodelLoss().to(device)
    train_criterion_align = AlignmentLoss(
        n_classes=cfg['n_classes'],
        lambda_smooth=cfg.get('lambda_smooth', 1.0),
        lambda_aux=cfg.get('lambda_aux', 0.4),
        ignore_index=cfg.get('id_unlabel', 255)
    ).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.get('id_unlabel', 255)).to(device)

    train_loss_meter = averageMeter()
    test_loss_meter = averageMeter()
    running_metrics_test = runningScore(cfg['n_classes'], ignore_index=cfg['id_unlabel'])
    best_test = 0
    start_epoch = 0

    scaler = GradScaler()

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint.get('epoch', 0)
            best_test = checkpoint.get('best_test', 0)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            logger.info(f"Loaded checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            logger.info(f"No checkpoint found at '{args.resume}'")
    
    save_interval = 10
    for ep in range(start_epoch, cfg['epochs']):
        model.train()
        train_loss_meter.reset()
        for i, sample in enumerate(train_loader):
            optimizer.zero_grad()
            image = sample['image'].to(device)
            depth = sample['depth'].to(device)
            label = sample['label'].to(device)
            bound = sample['bound'].to(device)
            binary_label = sample['binary_label'].to(device)

            with autocast():
                main_preds, deformation_field, aux_pred = model(image, depth)
                seg_targets = [label, binary_label, bound]
                loss_seg = train_criterion_seg(main_preds, seg_targets)
                loss_align = train_criterion_align(deformation_field, image, aux_pred, label)
                lambda_align = cfg.get('lambda_align', 1.0)
                loss = loss_seg + lambda_align * loss_align

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_meter.update(loss.item())

        scheduler.step(ep)

        with torch.no_grad():
            model.eval()
            running_metrics_test.reset()
            test_loss_meter.reset()
            for i, sample in enumerate(test_loader):
                image = sample['image'].to(device)
                depth = sample['depth'].to(device)
                label = sample['label'].to(device)
                main_preds, _, _ = model(image, depth)
                predict = main_preds[0]
                loss = criterion(predict, label)
                test_loss_meter.update(loss.item())
                predict = predict.max(1)[1].cpu().numpy()
                label = label.cpu().numpy()
                running_metrics_test.update(label, predict)
        
        train_loss = train_loss_meter.avg
        test_loss = test_loss_meter.avg
        test_scores = running_metrics_test.get_scores()
        test_macc = test_scores[0]["class_acc: "]
        test_miou = test_scores[0]["mIou: "]
        test_avg = (test_macc + test_miou) / 2
        logger.info(
            f'Iter | [{ep + 1:3d}/{cfg["epochs"]}] loss={train_loss:.3f}/{test_loss:.3f}, mPA={test_macc:.3f}, miou={test_miou:.3f}, avg={test_avg:.3f}')
        if test_avg > best_test:
            best_test = test_avg
            checkpoint = {
                'epoch': ep + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_test': best_test,
            }
            save_ckpt(logdir, checkpoint, ep + 1)
            logger.info(
                f'Save Iter = [{ep + 1:3d}],  mPA={test_macc:.3f}, miou={test_miou:.3f}, avg={test_avg:.3f}')
         

        if (ep + 1) % save_interval == 0:
            
            # 3. 创建与“最佳模型”相同的 checkpoint 字典
            checkpoint = {
                'epoch': ep + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_test': best_test, # 保存当前的最佳分数
            }
            
            # 4. 直接调用 torch.save 保存，并使用带轮次数的独立文件名
            #    这样可以避免覆盖掉您的最佳模型文件
            periodic_ckpt_path = os.path.join(logdir, f'checkpoint_epoch_{ep + 1}.pth')
            torch.save(checkpoint, periodic_ckpt_path)
            logger.info(f'--- Saved periodic checkpoint at epoch {ep + 1} to {periodic_ckpt_path} ---')
        # --- 新增 END ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--config", type=str, default="configs/LASNet.json", help="Configuration file to use")
    parser.add_argument("--resume", type=str, default='', help="use this file to load last checkpoint for continuing training")
    parser.add_argument("--cuda", type=int, default=0, help="set cuda device id")
    parser.add_argument("--cpu", action='store_true', help="Use CPU for training") # 新增 --cpu 选项
    args = parser.parse_args()

    print("Starting Training!")
    run(args)
