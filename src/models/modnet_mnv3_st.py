"""
Example:
    python -m src.models.modnet_mnv3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import SUPPORTED_BACKBONES


#------------------------------------------------------------------------------
#  MODNet Basic Modules
#------------------------------------------------------------------------------

class IBNorm(nn.Module):
    """ Combine Instance Norm and Batch Norm into One Layer
    """

    def __init__(self, in_channels):
        super(IBNorm, self).__init__()
        in_channels = in_channels
        self.bnorm_channels = int(in_channels / 2)
        self.inorm_channels = in_channels - self.bnorm_channels

        self.bnorm = nn.BatchNorm2d(self.bnorm_channels, affine=True)
        self.inorm = nn.InstanceNorm2d(self.inorm_channels, affine=False)
        
    def forward(self, x):
        bn_x = self.bnorm(x[:, :self.bnorm_channels, ...].contiguous())
        in_x = self.inorm(x[:, self.bnorm_channels:, ...].contiguous())

        return torch.cat((bn_x, in_x), 1)


class Conv2dIBNormRelu(nn.Module):
    """ Convolution + IBNorm + ReLu
    """

    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, groups=1, bias=True, 
                 with_ibn=True, with_relu=True):
        super(Conv2dIBNormRelu, self).__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, 
                      stride=stride, padding=padding, dilation=dilation, 
                      groups=groups, bias=bias)
        ]

        if with_ibn:       
            layers.append(IBNorm(out_channels))
        if with_relu:
            layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x) 


class SEBlock(nn.Module):
    """ SE Block Proposed in https://arxiv.org/pdf/1709.01507.pdf 
    """

    def __init__(self, in_channels, out_channels, reduction=1):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, int(in_channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels // reduction), out_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)

        return x * w.expand_as(x)


#------------------------------------------------------------------------------
#  MODNet Branches
#------------------------------------------------------------------------------

class LRBranch(nn.Module):
    """ Low Resolution Branch of MODNet
    """

    def __init__(self, backbone):
        super(LRBranch, self).__init__() 

        enc_channels = backbone.enc_channels # [16, 24, 32, 96, 1280]
        
        # self.backbone = backbone
        self.se_block = SEBlock(enc_channels[4], enc_channels[4], reduction=4)
        self.conv_lr16x = Conv2dIBNormRelu(enc_channels[4], enc_channels[3], 5, stride=1, padding=2) # (1280, 96, 5, 1, 2)
        self.conv_lr8x = Conv2dIBNormRelu(enc_channels[3], enc_channels[2], 5, stride=1, padding=2) # (96, 32, 5, 1, 2)
        self.conv_lr = Conv2dIBNormRelu(enc_channels[2], 1, kernel_size=3, stride=2, padding=1, with_ibn=False, with_relu=False) # (32, 1, 3, 2, 1, False, False)

        """ New convolution block """
        self.mobilenetv3 = backbone

    def forward(self, img, inference):
        # if backbone == 'mobilenetv3':
        enc_features = self.mobilenetv3.forward(img)
        # torch.Size([1, 16, 128, 128]) torch.Size([1, 24, 64, 64]) torch.Size([1, 40, 32, 32]) torch.Size([1, 48, 32, 32]) torch.Size([1, 96, 16, 16]) torch.Size([1, 576, 1, 1]) torch.Size([1, 1280, 1, 1])
        enc4x, enc128x = enc_features[1], enc_features[6]
        enc4x = F.interpolate(enc4x, scale_factor=2, mode='bilinear', align_corners=False)
        enc32x = F.interpolate(enc128x, scale_factor=16, mode='bilinear', align_corners=False)
            
        # else:
        #     enc_features = self.backbone.forward(img)
        #     # torch.Size([1, 16, 256, 256]) torch.Size([1, 24, 128, 128]) torch.Size([1, 32, 64, 64]) torch.Size([1, 96, 32, 32]) torch.Size([1, 1280, 16, 16])
        #     enc4x, enc32x = enc_features[1], enc_features[4]

        enc32x = self.se_block(enc32x) # [1, 1280, 8, 8]
        lr16x = F.interpolate(enc32x, scale_factor=2, mode='bilinear', align_corners=False)
        lr16x = self.conv_lr16x(lr16x) # [1, 96, 16, 16]
        lr8x = F.interpolate(lr16x, scale_factor=2, mode='bilinear', align_corners=False)
        lr8x = self.conv_lr8x(lr8x) # [1, 32, 32, 32]

        return lr8x, enc4x 


class HRBranch(nn.Module):
    """ High Resolution Branch of MODNet
    """

    def __init__(self, hr_channels, enc_channels):
        super(HRBranch, self).__init__()

        self.tohr_enc2x = Conv2dIBNormRelu(enc_channels[0], hr_channels, 1, stride=1, padding=0)
        self.conv_enc2x = Conv2dIBNormRelu(hr_channels + 3, hr_channels, 3, stride=2, padding=1)

        self.tohr_enc4x = Conv2dIBNormRelu(enc_channels[1], hr_channels, 1, stride=1, padding=0)
        self.conv_enc4x = Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1)

        self.conv_hr4x = nn.Sequential(
            Conv2dIBNormRelu(3 * hr_channels + 3, 2 * hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1),
        )

        self.conv_hr2x = nn.Sequential(
            Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(hr_channels, hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(hr_channels, hr_channels, 3, stride=1, padding=1),
        )

        self.conv_hr = nn.Sequential(
            Conv2dIBNormRelu(hr_channels + 3, hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(hr_channels, 1, kernel_size=1, stride=1, padding=0, with_ibn=False, with_relu=False),
        )

        """ New convolution block """
        self.tohr_hr4x = Conv2dIBNormRelu(hr_channels, hr_channels, 3, stride=1, padding=1)

        self.tohr_lr4x = nn.Sequential(
            Conv2dIBNormRelu(2 * hr_channels + 3, hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(hr_channels, hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(hr_channels, hr_channels, 3, stride=1, padding=1),
        )

        self.conv_lr = Conv2dIBNormRelu(enc_channels[2], 1, kernel_size=3, stride=2, padding=1, with_ibn=False, with_relu=False) # (32, 1, 3, 2, 1, False, False)

    def forward(self, img, enc4x, lr8x, inference):
        img4x = F.interpolate(img, scale_factor=1/4, mode='bilinear', align_corners=False)

        enc4x = self.tohr_enc4x(enc4x)
        hr4x = self.tohr_hr4x(enc4x)
        
        lr4x = F.interpolate(lr8x, scale_factor=2, mode='bilinear', align_corners=False)
        hr4x = self.tohr_lr4x(torch.cat((hr4x, lr4x, img4x), dim=1))

        f = F.interpolate(hr4x, scale_factor=8, mode='bilinear', align_corners=False)
        lr = self.conv_lr(f) # [1, 1, 256, 256]
        pred_matte = torch.sigmoid(lr) # [1, 1, 256, 256]

        return pred_matte


# class FusionBranch(nn.Module):
#     """ Fusion Branch of MODNet
#     """

#     def __init__(self, hr_channels, enc_channels):
#         super(FusionBranch, self).__init__()
#         self.conv_lr4x = Conv2dIBNormRelu(enc_channels[2], hr_channels, 5, stride=1, padding=2)
        
#         self.conv_f2x = Conv2dIBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1)
#         self.conv_f = nn.Sequential(
#             Conv2dIBNormRelu(hr_channels + 3, int(hr_channels / 2), 3, stride=1, padding=1),
#             Conv2dIBNormRelu(int(hr_channels / 2), 1, 1, stride=1, padding=0, with_ibn=False, with_relu=False),
#         )


#------------------------------------------------------------------------------
#  MODNet
#------------------------------------------------------------------------------

class MODNet_MNV3(nn.Module):
    """ Architecture of MODNet
    """

    def __init__(self, in_channels=3, hr_channels=32, backbone_arch='mobilenetv3', backbone_pretrained=True):
        super(MODNet_MNV3, self).__init__()

        self.in_channels = in_channels
        self.hr_channels = hr_channels
        self.backbone_arch = backbone_arch
        self.backbone_pretrained = backbone_pretrained

        # self.backbone = SUPPORTED_BACKBONES[self.backbone_arch](self.in_channels)

        # self.lr_branch = LRBranch(self.backbone)
        # self.hr_branch = HRBranch(self.hr_channels, self.backbone.enc_channels)
        # self.f_branch = FusionBranch(self.hr_channels, self.backbone.enc_channels)

        self.mobilenetv3 = SUPPORTED_BACKBONES[self.backbone_arch](self.in_channels)

        self.lr_branch = LRBranch(self.mobilenetv3)
        self.hr_branch = HRBranch(self.hr_channels, self.mobilenetv3.enc_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                self._init_conv(m)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                self._init_norm(m)

        if self.backbone_pretrained:
            # self.backbone.load_pretrained_ckpt()             
            self.mobilenetv3.load_pretrained_ckpt()         

    def forward(self, img, inference=False):
        lr8x, enc4x = self.lr_branch(img, inference)
        pred_matte = self.hr_branch(img, enc4x, lr8x, inference)

        return pred_matte
    
    def freeze_norm(self):
        norm_types = [nn.BatchNorm2d, nn.InstanceNorm2d]
        for m in self.modules():
            for n in norm_types:
                if isinstance(m, n):
                    m.eval()
                    continue
    
    def unfreeze_semantic(self):
        for params in self.modules():
            params.requires_grad = True

    def _init_conv(self, conv):
        nn.init.kaiming_uniform_(
            conv.weight, a=0, mode='fan_in', nonlinearity='relu')
        if conv.bias is not None:
            nn.init.constant_(conv.bias, 0)

    def _init_norm(self, norm):
        if norm.weight is not None:
            nn.init.constant_(norm.weight, 1)
            nn.init.constant_(norm.bias, 0)


import pytorch_model_summary 


if __name__ == '__main__':
    modnet = torch.nn.DataParallel(MODNet_MNV3(backbone_pretrained=True)).cuda()
    print(pytorch_model_summary.summary(modnet, torch.zeros(1, 3, 512, 512), show_input=True))
