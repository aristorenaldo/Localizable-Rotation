import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from . import resnet_cifar

__all__ = ['Moe1', 'Nomoe', 'Lorot']

class Moe1(nn.Module):
    def __init__(self, num_classes=10, backbone='resnet32', num_flips=2, num_sc=6, num_lorot=16) -> None:
        super().__init__()
        self.backbone = resnet_cifar.__dict__[backbone]()
        self.backbone.fc = nn.Identity()

        self.classifier = nn.Linear(64, num_classes)
        self.lorot_layer = nn.Linear(64, num_lorot)
        self.flip_layer = nn.Linear(64, num_flips)
        self.sc_layer = nn.Linear(64, num_sc)

        self.gating_layer = nn.Linear(64, 3)
    
    def forward(self, x, ssl_task=True):
        out = self.backbone(x)

        if self.training and ssl_task:
            return (
                self.classifier(out),
                self.lorot_layer(out),
                self.flip_layer(out),
                self.sc_layer(out),
                self.gating_layer(out)
            )
        
        return self.classifier(out)
    
class Nomoe(nn.Module):
    def __init__(self, num_classes=10, backbone='resnet32', num_flips=2, num_sc=6, num_lorot=16) -> None:
        super().__init__()
        self.backbone = resnet_cifar.__dict__[backbone]()
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(64, num_classes)
        self.lorot_layer = nn.Linear(64, num_lorot)
        self.flip_layer = nn.Linear(64, num_flips)
        self.sc_layer = nn.Linear(64, num_sc)
    def forward(self, x, ssl_task=True):
        out = self.backbone(x)
        if self.training and ssl_task:
            return (
                self.classifier(out),
                self.lorot_layer(out),
                self.flip_layer(out),
                self.sc_layer(out),
            )
        return self.classifier(out)
    
class Lorot(nn.Module):
    def __init__(self, num_classes=10, backbone='resnet32', num_lorot=16):
        super().__init__()
        self.backbone = resnet_cifar.__dict__[backbone]()
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(64, num_classes)
        self.lorot_layer = nn.Linear(64, num_lorot)
    
    def forward(self, x, ssl_task=True):
        out = self.backbone(x)
        if self.training and ssl_task:
            return (
                self.classifier(out),
                self.lorot_layer(out)
            )
        return self.classifier(out)

