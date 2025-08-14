# losses_deformation.py (最终修正版)

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 修改 START: 使用一个更标准和鲁棒的DiceLoss实现 ---
class DiceLoss(nn.Module):
    """
    一个更健壮的Dice损失实现，直接在4D张量上操作，并正确处理ignore_index。
    """
    def __init__(self, n_classes, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index

    def forward(self, inputs, target, softmax=True):
        # inputs: (B, C, H, W) - 模型的原始输出
        # target: (B, H, W) - 标签
        
        # 1. 将模型输出转换为概率
        if softmax:
            probs = torch.softmax(inputs, dim=1)
        else:
            probs = inputs

        # 2. 将标签转换为 one-hot 编码
        # target: (B, H, W) -> (B, 1, H, W) -> (B, C, H, W)
        target_one_hot = F.one_hot(target, num_classes=self.n_classes).permute(0, 3, 1, 2)

        # 3. 创建一个mask来忽略指定的像素
        mask = (target != self.ignore_index).unsqueeze(1) # (B, 1, H, W)

        # 4. 将mask应用到概率和one-hot标签上
        probs = probs * mask
        target_one_hot = target_one_hot * mask

        # 5. 计算交集和并集
        # 沿着 H 和 W 维度求和
        intersection = (probs * target_one_hot).sum(dim=(2, 3)) # (B, C)
        cardinality = (probs + target_one_hot).sum(dim=(2, 3)) # (B, C)

        smooth = 1e-5
        dice_score = (2. * intersection + smooth) / (cardinality + smooth)
        
        # 6. 计算最终损失
        dice_loss = 1 - dice_score

        # 对所有类别和batch取平均
        return dice_loss.mean()
# --- 修改 END ---


class SmoothnessLoss(nn.Module):
    # ... (SmoothnessLoss 保持不变) ...
    def __init__(self, alpha=10):
        super().__init__()
        self.alpha = alpha
    def gradient_x(self, img):
        return img[:, :, :, :-1] - img[:, :, :, 1:]
    def gradient_y(self, img):
        return img[:, :, :-1, :] - img[:, :, 1:, :]
    def forward(self, deformation_field, guidance_image):
        guidance_gy = self.gradient_y(guidance_image)
        guidance_gx = self.gradient_x(guidance_image)
        deformation_field_t = deformation_field.permute(0, 3, 1, 2)
        disp_gy = self.gradient_y(deformation_field_t)
        disp_gx = self.gradient_x(deformation_field_t)
        weights_x = torch.exp(-self.alpha * torch.mean(torch.abs(guidance_gx), 1, keepdim=True))
        weights_y = torch.exp(-self.alpha * torch.mean(torch.abs(guidance_gy), 1, keepdim=True))
        smoothness_x = torch.mean(weights_x * (disp_gx**2))
        smoothness_y = torch.mean(weights_y * (disp_gy**2))
        return smoothness_x + smoothness_y


class AuxSegLoss(nn.Module):
    # ... (AuxSegLoss 保持不变) ...
    def __init__(self, n_classes, ignore_index=255):
        super().__init__()
        self.dice_loss = DiceLoss(n_classes, ignore_index=ignore_index)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
    def forward(self, aux_pred, gt_label):
        loss_dice = self.dice_loss(aux_pred, gt_label, softmax=True)
        loss_ce = self.ce_loss(aux_pred, gt_label)
        return loss_dice + loss_ce


class AlignmentLoss(nn.Module):
    # ... (AlignmentLoss 保持不变) ...
    def __init__(self, n_classes, lambda_smooth=1.0, lambda_aux=1.0, alpha=10, ignore_index=255):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_aux = lambda_aux
        self.smooth_loss = SmoothnessLoss(alpha=alpha)
        self.aux_loss = AuxSegLoss(n_classes=n_classes, ignore_index=ignore_index)
        self.lambda_mi = 0.0
    def forward(self, deformation_field, rgb_image, aux_pred, gt_label):
        rgb_gray = torch.mean(rgb_image, 1, keepdim=True)
        h, w = deformation_field.shape[1:3]
        rgb_gray_resized = F.interpolate(rgb_gray, size=(h,w), mode='bilinear', align_corners=True)
        loss_smooth = self.smooth_loss(deformation_field, rgb_gray_resized)
        h_gt, w_gt = gt_label.shape[1:3]
        aux_pred_resized = F.interpolate(aux_pred, size=(h_gt, w_gt), mode='bilinear', align_corners=True)
        loss_aux = self.aux_loss(aux_pred_resized, gt_label)
        loss_mi = 0.0
        total_alignment_loss = (self.lambda_smooth * loss_smooth + 
                                self.lambda_aux * loss_aux + 
                                self.lambda_mi * loss_mi)
        return total_alignment_loss