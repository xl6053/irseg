import os
import torch.nn as nn
import torch
from resnet import Backbone_ResNet152_in3
import torch.nn.functional as F
import numpy as np
from toolbox.dual_self_att import CAM_Module


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# --- (已有模块 ChannelAttention, SpatialAttention, CorrelationModule, CLM, CAM, ESM, prediction_decoder 定义保持不变) ---
# --- (The definitions of existing modules remain unchanged) ---
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

class CorrelationModule(nn.Module):
    def  __init__(self, all_channel=64):
        super(CorrelationModule, self).__init__()
        self.linear_e = nn.Linear(all_channel, all_channel,bias = False)
        self.channel = all_channel
        self.fusion = BasicConv2d(all_channel, all_channel, kernel_size=3, padding=1)
    def forward(self, exemplar, query): # exemplar: middle, query: rgb or T
        fea_size = exemplar.size()[2:]
        all_dim = fea_size[0]*fea_size[1]
        exemplar_flat = exemplar.view(-1, self.channel, all_dim) #N,C,H*W
        query_flat = query.view(-1, self.channel, all_dim)
        exemplar_t = torch.transpose(exemplar_flat,1,2).contiguous()  #batchsize x dim x num, N,H*W,C
        exemplar_corr = self.linear_e(exemplar_t) #
        A = torch.bmm(exemplar_corr, query_flat)
        B = F.softmax(torch.transpose(A,1,2),dim=1)
        exemplar_att = torch.bmm(query_flat, B).contiguous()
        exemplar_att = exemplar_att.view(-1, self.channel, fea_size[0], fea_size[1])
        exemplar_out = self.fusion(exemplar_att)
        return exemplar_out

class CLM(nn.Module):
    def __init__(self, all_channel=64):
        super(CLM, self).__init__()
        self.corr_x_2_x_ir = CorrelationModule(all_channel)
        self.corr_ir_2_x_ir = CorrelationModule(all_channel)
        self.smooth1 = BasicConv2d(all_channel, all_channel, kernel_size=3, padding=1)
        self.smooth2 = BasicConv2d(all_channel, all_channel, kernel_size=3, padding=1)
        self.fusion = BasicConv2d(2*all_channel, all_channel, kernel_size=3, padding=1)
        self.pred = nn.Conv2d(all_channel, 2, kernel_size=3, padding=1, bias = True)
    def forward(self, x, x_ir, ir):
        corr_x_2_x_ir = self.corr_x_2_x_ir(x_ir,x)
        corr_ir_2_x_ir = self.corr_ir_2_x_ir(x_ir,ir)
        summation = self.smooth1(corr_x_2_x_ir + corr_ir_2_x_ir)
        multiplication = self.smooth2(corr_x_2_x_ir * corr_ir_2_x_ir)
        fusion = self.fusion(torch.cat([summation,multiplication],1))
        sal_pred = self.pred(fusion)
        return fusion, sal_pred

class CAM(nn.Module):
    def __init__(self, all_channel=64):
        super(CAM, self).__init__()
        self.conv2 = BasicConv2d(all_channel, all_channel, kernel_size=3, padding=1)
        self.sa = SpatialAttention()
        self.cam = CAM_Module(all_channel)
    def forward(self, x, ir):
        multiplication = x * ir
        summation = self.conv2(x + ir)
        sa = self.sa(multiplication)
        summation_sa = summation.mul(sa)
        sc_feat = self.cam(summation_sa)
        return sc_feat

class ESM(nn.Module):
    def __init__(self, all_channel=64):
        super(ESM, self).__init__()
        self.conv1 = BasicConv2d(all_channel, all_channel, kernel_size=3, padding=1)
        self.conv2 = BasicConv2d(all_channel, all_channel, kernel_size=3, padding=1)
        self.dconv1 = BasicConv2d(all_channel,int( all_channel/4), kernel_size=3, padding=1)
        self.dconv2 = BasicConv2d(all_channel,int( all_channel/4), kernel_size=3, dilation=3, padding=3)
        self.dconv3 = BasicConv2d(all_channel,int( all_channel/4), kernel_size=3, dilation=5, padding=5)
        self.dconv4 = BasicConv2d(all_channel,int( all_channel/4), kernel_size=3, dilation=7, padding=7)
        self.fuse_dconv = nn.Conv2d(all_channel, all_channel, kernel_size=3,padding=1)
        self.pred = nn.Conv2d(all_channel, 2, kernel_size=3, padding=1, bias = True)
    def forward(self, x, ir):
        multiplication = self.conv1(x * ir)
        summation = self.conv2(x + ir)
        fusion = (summation + multiplication)
        x1 = self.dconv1(fusion)
        x2 = self.dconv2(fusion)
        x3 = self.dconv3(fusion)
        x4 = self.dconv4(fusion)
        out = self.fuse_dconv(torch.cat((x1, x2, x3, x4), dim=1))
        edge_pred = self.pred(out)
        return out, edge_pred

class prediction_decoder(nn.Module):
    def __init__(self, channel1=64, channel2=128, channel3=256, channel4=256, channel5=512, n_classes=9):
        super(prediction_decoder, self).__init__()
        self.decoder5 = nn.Sequential(
                nn.Dropout2d(p=0.1),
                BasicConv2d(channel5, channel5, kernel_size=3, padding=3, dilation=3),
                BasicConv2d(channel5, channel4, kernel_size=3, padding=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                )
        self.decoder4 = nn.Sequential(
                nn.Dropout2d(p=0.1),
                BasicConv2d(channel4, channel4, kernel_size=3, padding=3, dilation=3),
                BasicConv2d(channel4, channel3, kernel_size=3, padding=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                )
        self.decoder3 = nn.Sequential(
                nn.Dropout2d(p=0.1),
                BasicConv2d(channel3, channel3, kernel_size=3, padding=3, dilation=3),
                BasicConv2d(channel3, channel2, kernel_size=3, padding=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                )
        self.decoder2 = nn.Sequential(
                nn.Dropout2d(p=0.1),
                BasicConv2d(channel2, channel2, kernel_size=3, padding=3, dilation=3),
                BasicConv2d(channel2, channel1, kernel_size=3, padding=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                )
        self.semantic_pred2 = nn.Conv2d(channel1, n_classes, kernel_size=3, padding=1)
        self.decoder1 = nn.Sequential(
                nn.Dropout2d(p=0.1),
                BasicConv2d(channel1, channel1, kernel_size=3, padding=3, dilation=3),
                BasicConv2d(channel1, channel1, kernel_size=3, padding=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                BasicConv2d(channel1, channel1, kernel_size=3, padding=1),
                nn.Conv2d(channel1, n_classes, kernel_size=3, padding=1)
                )
    def forward(self, x5, x4, x3, x2, x1):
        x5_decoder = self.decoder5(x5)
        x4_decoder = self.decoder4(x5_decoder + x4)
        x3_decoder = self.decoder3(x4_decoder + x3)
        x2_decoder = self.decoder2(x3_decoder + x2)
        semantic_pred2 = self.semantic_pred2(x2_decoder)
        semantic_pred = self.decoder1(x2_decoder + x1)
        return semantic_pred,semantic_pred2


# --- 新增 START ---
class DeformationHead(nn.Module):
    """
    一个独立的头，用于从融合特征中预测形变场。
    """
    def __init__(self, input_channels=320): # 输入通道数为 64 * 5 = 320
        super(DeformationHead, self).__init__()
        self.conv_layers = nn.Sequential(
            BasicConv2d(input_channels, 256, kernel_size=3, padding=1),
            BasicConv2d(256, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 2, kernel_size=1) # 输出2个通道，代表dx, dy
        )
        self.tanh = nn.Tanh() # Tanh将输出限制在[-1, 1]

    def forward(self, x):
        phi = self.tanh(self.conv_layers(x))
        # phi的形状是 (B, 2, H, W)，需要调整为 (B, H, W, 2) 以用于后续的grid_sample
        phi = phi.permute(0, 2, 3, 1)
        return phi
# --- 新增 END ---


class LASNet(nn.Module):
    def __init__(self, n_classes):
        super(LASNet, self).__init__()

        # --- (原始的层定义保持不变) ---
        (
            self.layer1_rgb, self.layer2_rgb, self.layer3_rgb, self.layer4_rgb, self.layer5_rgb,
        ) = Backbone_ResNet152_in3(pretrained=True)
        self.rgbconv1 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.rgbconv2 = BasicConv2d(256, 128, kernel_size=3, padding=1)
        self.rgbconv3 = BasicConv2d(512, 256, kernel_size=3, padding=1)
        self.rgbconv4 = BasicConv2d(1024, 256, kernel_size=3, padding=1)
        self.rgbconv5 = BasicConv2d(2048, 512, kernel_size=3, padding=1)
        self.CLM5 = CLM(512)
        self.CAM4 = CAM(256)
        self.CAM3 = CAM(256)
        self.CAM2 = CAM(128)
        self.ESM1 = ESM(64)
        self.decoder = prediction_decoder(64,128,256,256,512, n_classes)

        # --- 新增 START ---
        # 1. 新增：用于聚合中间特征以预测形变场的模块
        self.def_fuse_conv5 = BasicConv2d(512, 64, kernel_size=1)
        self.def_fuse_conv4 = BasicConv2d(256, 64, kernel_size=1)
        self.def_fuse_conv3 = BasicConv2d(256, 64, kernel_size=1)
        self.def_fuse_conv2 = BasicConv2d(128, 64, kernel_size=1)
        self.def_fuse_conv1 = BasicConv2d(64, 64, kernel_size=1)
        
        # 2. 新增：形变场预测头
        self.deformation_head = DeformationHead(input_channels=320) # 64 * 5 = 320

        # 3. 新增：用于计算L_aux的辅助分割解码器
        self.aux_decoder_for_registration = prediction_decoder(
            channel1=64, channel2=128, channel3=256, channel4=256, channel5=512, n_classes=n_classes
        )
        # --- 新增 END ---


    def forward(self, rgb, depth):
        # 准备热成像输入
        x = rgb
        ir = depth[:, :1, ...]
        ir = torch.cat((ir, ir, ir), dim=1)

        # === 步骤 1: 双流特征提取 ===
        x1 = self.layer1_rgb(x); x2 = self.layer2_rgb(x1); x3 = self.layer3_rgb(x2); x4 = self.layer4_rgb(x3); x5 = self.layer5_rgb(x4)
        x1, x2, x3, x4, x5 = self.rgbconv1(x1), self.rgbconv2(x2), self.rgbconv3(x3), self.rgbconv4(x4), self.rgbconv5(x5)
        
        # 原始代码这里对ir也用了rgbconv，我们保持一致
        ir1 = self.layer1_rgb(ir); ir2 = self.layer2_rgb(ir1); ir3 = self.layer3_rgb(ir2); ir4 = self.layer4_rgb(ir3); ir5 = self.layer5_rgb(ir4)
        ir1, ir2, ir3, ir4, ir5 = self.rgbconv1(ir1), self.rgbconv2(ir2), self.rgbconv3(ir3), self.rgbconv4(ir4), self.rgbconv5(ir5)

        # === 分支一：主分割任务 (Main Segmentation Branch) ===
        # 1. 保持LASNet原有的、基于未对齐特征的融合方式不变
        out5, sal  = self.CLM5(x5, x5*ir5, ir5)
        out4 = self.CAM4(x4, ir4)
        out3 = self.CAM3(x3, ir3)
        out2 = self.CAM2(x2, ir2)
        out1, edge = self.ESM1(x1, ir1)
        
        # 2. 将融合特征送入主解码器得到最终的分割结果
        semantic, semantic2 = self.decoder(out5, out4, out3, out2, out1)

        # === 分支二：形变场预测任务 (Deformation Field Prediction Branch) ===
        # 1. 从主干分支中“窃取”中间融合特征(out1到out5)
        # 将所有中间融合特征上采样到统一尺寸（这里以out2的尺寸为基准）并融合
        target_size = out2.shape[2:]
        d5 = F.interpolate(self.def_fuse_conv5(out5), size=target_size, mode='bilinear', align_corners=True)
        d4 = F.interpolate(self.def_fuse_conv4(out4), size=target_size, mode='bilinear', align_corners=True)
        d3 = F.interpolate(self.def_fuse_conv3(out3), size=target_size, mode='bilinear', align_corners=True)
        d2 = self.def_fuse_conv2(out2)
        d1 = F.interpolate(self.def_fuse_conv1(out1), size=target_size, mode='bilinear', align_corners=True)
        
        # 2. 融合成 F_out, 并送入形变场预测头
        f_out_for_deformation = torch.cat([d1, d2, d3, d4, d5], dim=1)
        deformation_field = self.deformation_head(f_out_for_deformation)
        
        # === 分支三：辅助损失任务 (Auxiliary Loss Branch) ===
        # 将主干的融合特征也送入辅助解码器，以计算辅助损失
        aux_semantic, _ = self.aux_decoder_for_registration(out5, out4, out3, out2, out1)
        
        # --- (后处理插值部分保持不变) ---
        semantic2 = torch.nn.functional.interpolate(semantic2, scale_factor=2, mode='bilinear')
        sal = torch.nn.functional.interpolate(sal, scale_factor=32, mode='bilinear')
        edge = torch.nn.functional.interpolate(edge, scale_factor=2, mode='bilinear')

        # --- 修改返回 START ---
        # 返回所有三个分支的输出，以便在train.py中计算总损失
        return (semantic, semantic2, sal, edge), deformation_field, aux_semantic
        # --- 修改返回 END ---

if __name__ == '__main__':
    # 更新测试代码以验证新架构
    print("Testing the modified LASNet model with parallel branches.")
    
    n_classes = 9
    model = LASNet(n_classes=n_classes)
    model.eval()
    
    dummy_rgb = torch.randn(2, 3, 480, 640)
    dummy_depth = torch.randn(2, 1, 480, 640)
    
    if torch.cuda.is_available():
        model = model.cuda()
        dummy_rgb = dummy_rgb.cuda()
        dummy_depth = dummy_depth.cuda()
        print("Using GPU.")
    else:
        print("Using CPU.")

    with torch.no_grad():
        # --- 修改 START ---
        # 接收新的多重返回值
        main_preds, deformation_field, aux_pred = model(dummy_rgb, dummy_depth)
        semantic, semantic2, sal, edge = main_preds
        # --- 修改 END ---
    
    print("\nModel outputs:")
    print(f"Final Semantic Prediction Shape: {semantic.shape}")
    print(f"Intermediate Semantic Prediction Shape: {semantic2.shape}")
    print(f"Saliency Prediction Shape: {sal.shape}")
    print(f"Edge Prediction Shape: {edge.shape}")
    # --- 新增 START ---
    print(f"Deformation Field Prediction Shape: {deformation_field.shape}")
    print(f"Auxiliary Semantic Prediction Shape: {aux_pred.shape}")
    # --- 新增 END ---

    # 验证输出形状
    assert semantic.shape == (2, n_classes, 480, 640)
    assert len(deformation_field.shape) == 4 and deformation_field.shape[3] == 2
    assert aux_pred.shape[1] == n_classes
    
    print("\nVerification successful! The model with parallel branches is ready.")