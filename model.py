import torch.nn as nn
# from torchvision.models import resnet18
import torch
from os.path import join

from torch.utils.tensorboard import SummaryWriter

class ResNet18_For_CIFAR_100(nn.Module):
    
    def __init__(self, init_weight=True):
        super().__init__()
        self.resnet = resnet18()
        self.extra_layer = nn.Linear(1000, 100)
        
        if init_weight:
            for m in self.modules():
                self.init_one_layer(m)        
        
    def forward(self, x):
        x = self.resnet.forward(x)
        x = self.extra_layer(x)
        return x
    
    
    def init_one_layer(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)   
            

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.residual_branch = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,
                      out_channels * BasicBlock.expansion,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion))

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels * BasicBlock.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion))

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_branch(x) +
                                     self.shortcut(x))            
                 

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=100, inter_layer=False, dropout=True):
        super(ResNet, self).__init__()
        self.inter_layer = inter_layer
        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.stage2 = self._make_layer(block, 64, layers[0], 1)
        self.dropout2 = nn.Dropout(p=.2) if dropout else None
        self.stage3 = self._make_layer(block, 128, layers[1], 2)
        self.dropout3 = nn.Dropout(p=.3) if dropout else None
        self.stage4 = self._make_layer(block, 256, layers[2], 2)
        self.dropout4 = nn.Dropout(p=.4) if dropout else None
        self.stage5 = self._make_layer(block, 512, layers[3], 2)
        self.dropout5 = nn.Dropout(p=.5) if dropout else None
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the 
        same as a neuron netowork layer, ex. conv layer), one layer may 
        contain more than one residual block 
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        
        Return:
            return a resnet layer
        """

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        if self.inter_layer:
            x1 = self.stage2(x)
            x2 = self.stage3(x1)
            x3 = self.stage4(x2)
            x4 = self.stage5(x3)
            x = self.avg_pool(x4)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

            return [x1, x2, x3, x4, x]
        else:
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.stage4(x)
            x = self.stage5(x)
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

            return x


def _resnet(arch, block, layers, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)

    return model


def resnet18(progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], progress,
                   **kwargs)


def resnet34(progress=True, **kwargs):
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], progress,
                   **kwargs)


if __name__ == '__main__':
    # model = ResNet18_For_CIFAR_100()
    model = resnet18()
    print(model)