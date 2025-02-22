from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'Moe1',
    'Moe1flip',
    'Moe1sc',
    'Nomoe',
    'Lorot',
    'vanilla'
]

class Moe1(nn.Module):
    def __init__(self, num_classes=200, backbone='resnet18', num_flips=2, num_sc=6, num_lorot=16):
        super().__init__()
        self.backbone = models.__dict__[backbone]()
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(512, num_classes)
        self.lorot_layer = nn.Linear(512, num_lorot)
        self.flip_layer = nn.Linear(512, num_flips)
        self.sc_layer = nn.Linear(512, num_sc)
        self.gating_layer = nn.Linear(512, 3)

    def forward(self, x):
        out = self.backbone(x)
        if self.training:
            return (
                self.classifier(out),
                self.lorot_layer(out),
                self.flip_layer(out),
                self.sc_layer(out),
                self.gating_layer(out)
            )
        return self.classifier(out)

class Moe1flip(nn.Module):
    def __init__(self, num_classes=200, backbone='resnet18', num_flips=2, num_lorot=16):
        super().__init__()
        self.backbone = models.__dict__[backbone]()
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(512, num_classes)
        self.lorot_layer = nn.Linear(512, num_lorot)
        self.flip_layer = nn.Linear(512, num_flips)
        self.gating_layer = nn.Linear(512, 2)

    def forward(self, x):
        out = self.backbone(x)
        if self.training:
            return (
                self.classifier(out),
                self.lorot_layer(out),
                self.flip_layer(out),
                self.gating_layer(out)
            )
        return self.classifier(out)

class Moe1sc(nn.Module):
    def __init__(self, num_classes=200, backbone='resnet18', num_sc=6, num_lorot=16):
        super().__init__()
        self.backbone = models.__dict__[backbone]()
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(512, num_classes)
        self.lorot_layer = nn.Linear(512, num_lorot)
        self.sc_layer = nn.Linear(512, num_sc)
        self.gating_layer = nn.Linear(512, 2)

    def forward(self, x):
        out = self.backbone(x)
        if self.training:
            return (
                self.classifier(out),
                self.lorot_layer(out),
                self.sc_layer(out),
                self.gating_layer(out)
            )
        return self.classifier(out)

class Nomoe(nn.Module):
    def __init__(self, num_classes=200, backbone='resnet18', num_flips=2, num_sc=6, num_lorot=16) -> None:
        super().__init__()
        self.backbone = models.__dict__[backbone]()
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(512, num_classes)
        self.lorot_layer = nn.Linear(512, num_lorot)
        self.flip_layer = nn.Linear(512, num_flips)
        self.sc_layer = nn.Linear(512, num_sc)
    def forward(self, x):
        out = self.backbone(x)
        if self.training:
            return (
                self.classifier(out),
                self.lorot_layer(out),
                self.flip_layer(out),
                self.sc_layer(out),
            )
        return self.classifier(out)


class Lorot(nn.Module):
    def __init__(self, num_classes=200, backbone='resnet18', num_lorot=16):
        super().__init__()
        self.backbone = models.__dict__[backbone]()
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(512, num_classes)
        self.lorot_layer = nn.Linear(512, num_lorot)
    
    def forward(self, x):
        out = self.backbone(x)
        if self.training:
            return (
                self.classifier(out),
                self.lorot_layer(out)
            )
        return self.classifier(out)

def vanilla(backbone='resnet18', num_classes=200):
    return models.__dict__[backbone](num_classes=num_classes)


