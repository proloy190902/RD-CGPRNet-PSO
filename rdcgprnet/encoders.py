import torch
import torch.nn as nn
from torchvision.models import (
    resnet50, resnet101, vgg16,
    ResNet50_Weights, ResNet101_Weights, VGG16_Weights,
)


class RGBEncoder(nn.Module):
    def __init__(self, backbone='resnet50', out_channels=256):
        super().__init__()
        in_dim = 512 if backbone == 'vgg16' else 2048
        self.generic_branch = self._make_backbone(backbone)
        self.task_branch = self._make_backbone(backbone)
        for p in self.generic_branch.parameters():
            p.requires_grad = False

        def proj(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 1, bias=False),
                nn.BatchNorm2d(o),
                nn.ReLU(inplace=True),
            )

        self.proj_gen = proj(in_dim, out_channels)
        self.proj_task = proj(in_dim, out_channels)
        self.gate = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1, bias=False),
            nn.Sigmoid(),
        )

    @staticmethod
    def _make_backbone(name):
        if name == 'resnet50':
            m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            return nn.Sequential(*list(m.children())[:-2])
        if name == 'resnet101':
            m = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
            return nn.Sequential(*list(m.children())[:-2])
        if name == 'vgg16':
            return vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        raise ValueError(f'Unsupported backbone: {name}')

    def forward(self, x):
        with torch.no_grad():
            fg = self.generic_branch(x)
        ft = self.task_branch(x)
        fg = self.proj_gen(fg)
        ft = self.proj_task(ft)
        g = self.gate(torch.cat([fg, ft], dim=1))
        return g * ft + (1.0 - g) * fg


class DepthEncoder(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )
        self.depth_attn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, depth):
        f = self.encoder(depth)
        a = self.depth_attn(f)
        return f * a
