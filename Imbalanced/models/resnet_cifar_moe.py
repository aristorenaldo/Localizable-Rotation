'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter

from torch.autograd import Variable

__all__ = ['Moe1', 'Moe2', 'Nomoe']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.fc = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out



def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])

class Nomoe(nn.Module):
    def __init__(self, num_classes=10, backbone='resnet32', use_norm=False, num_regions=4, num_flips=4, num_sc=6, num_trans=16):
        super().__init__()
        self.backbone = globals()[backbone]()
        self.backbone.fc = nn.Identity()
        if use_norm:
            self.classifier = NormedLinear(64, num_classes)
            self.regflip = NormedLinear(64, num_trans)
            # self.region_layer = NormedLinear(64, num_regions)
            # self.flip_layer = NormedLinear(64, num_flips)
            self.sc_layer = NormedLinear(64, num_sc)
        else:
            self.classifier = nn.Linear(64, num_classes)
            self.regflip = nn.Linear(64, num_trans)
            # self.region_layer = nn.Linear(64, num_regions)
            # self.flip_layer = nn.Linear(64, num_flips)
            self.sc_layer = nn.Linear(64, num_sc)

        self.apply(_weights_init)
    def forward(self, x):
        out = self.backbone(x)
        if self.training:
            return (
                self.classifier(out),
                self.regflip(out),
                # self.region_layer(out),
                # self.flip_layer(out),
                self.sc_layer(out),
            )
        return self.classifier(out)

class Moe1(nn.Module):
    def __init__(self, num_classes=10, backbone='resnet32', use_norm=False, num_regions=4, num_flips=4, num_sc=6, num_trans=16):
        super().__init__()
        self.backbone = globals()[backbone]()
        self.backbone.fc = nn.Identity()
        if use_norm:
            self.classifier = NormedLinear(64, num_classes)
            self.lorot_layer = NormedLinear(64, num_trans)
            # self.regflip = NormedLinear(64, num_trans)
            # self.region_layer = NormedLinear(64, num_regions)
            # self.flip_layer = NormedLinear(64, num_flips)
            self.sc_layer = NormedLinear(64, num_sc)
        else:
            self.classifier = nn.Linear(64, num_classes)
            self.lorot_layer = nn.Linear(64, num_trans)
            # self.regflip = nn.Linear(64, num_trans)
            # self.region_layer = nn.Linear(64, num_regions)
            # self.flip_layer = nn.Linear(64, num_flips)
            self.sc_layer = nn.Linear(64, num_sc)
        self.gating_layer = nn.Linear(64, 2)

        self.apply(_weights_init)

    def forward(self, x):
        out = self.backbone(x)
        if self.training:
            return (
                self.classifier(out),
                self.lorot_layer(out),
                # self.regflip(out),
                # self.region_layer(out),
                # self.flip_layer(out),
                self.sc_layer(out),
                self.gating_layer(out)
            )
        return self.classifier(out)

class Moe2(nn.Module):
    def __init__(self, num_classes=10, backbone='resnet32', use_norm=False, num_regions=4, num_flips=4, num_sc=24, num_trans=16):
        super().__init__()
        self.backbone = globals()[backbone]()
        self.backbone.fc = nn.Identity()
        if use_norm:
            self.classifier = NormedLinear(64, num_classes)
            self.lorot_layer = NormedLinear(64, num_trans)
            # self.regflip = NormedLinear(64, num_trans)
            # self.region_layer = NormedLinear(64, num_regions)
            # self.flip_layer = NormedLinear(64, num_flips)
            self.sc_layer = NormedLinear(64, num_sc)
        else:
            self.classifier = nn.Linear(64, num_classes)
            # self.regflip = nn.Linear(64, num_trans)
            self.lorot_layer = nn.Linear(64, num_trans)
            # self.region_layer = nn.Linear(64, num_regions)
            # self.flip_layer = nn.Linear(64, num_flips)
            self.sc_layer = nn.Linear(64, num_sc)
        self.gating_layer = nn.Linear(64, 3)

        self.apply(_weights_init)

    def forward(self, x):
        out = self.backbone(x)
        if self.training:
            return (
                self.classifier(out),
                self.lorot_layer(out),
                # self.regflip(out),
                # self.region_layer(out),
                # self.flip_layer(out),
                self.sc_layer(out),
                self.gating_layer(out)
            )
        return self.classifier(out)

    

class BarlowTwins(nn.Module):
    def __init__(self, projector = '100-100', num_classes=10, batch_size=128, lambd=0.0051, lorot=False, lorot_trans=16):
        super().__init__()
        self.batch_size = batch_size
        self.lambd = lambd
        self.lorot = lorot

        self.backbone = resnet32()
        self.backbone.fc = nn.Identity()

        if num_classes:
            self.fc = nn.Linear(64, num_classes)

        if lorot:
            self.fc_lorot = nn.Linear(64, lorot_trans)

        # projector
        sizes = [64] + list(map(int, projector.split('-')))
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, y, y1=None, bt_lorot = False):
        out = self.backbone(y)

        # training step
        if self.training:
            z1 = self.projector(out)
            z2 = self.projector(self.backbone(y1))

            # empirical cross-correlation matrix
            c = self.bn(z1).T @ self.bn(z2)

            # sum the cross-correlation matrix between all gpus
            c.div_(self.batch_size)
            # torch.distributed.all_reduce(c)

            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = off_diagonal(c).pow_(2).sum()
            loss_bt = on_diag + self.lambd * off_diag

            pred = self.fc(out)

            if bt_lorot:
                lorot_pred = self.fc_lorot(out)
                return pred, lorot_pred, loss_bt

            return pred, loss_bt

        # validation / test step 
        pred = self.fc(out)
        return pred


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:

        
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()