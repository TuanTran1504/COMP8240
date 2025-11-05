import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class PreActBlock(nn.Module):
    '''Pre-activation version of the basic block (two 3x3 convs).'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
   

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)  
        out = self.conv2(F.relu(self.bn2(out)))
        residual = x if self.shortcut is None else self.shortcut(x)  
        return out + residual

class ResNet(nn.Module):
    def __init__(self, depth, width, num_classes=10):
        super(ResNet, self).__init__()
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        widths = [int(v * width) for v in (16, 32, 64)]
        self.in_planes = 16

        # Initial convolution
        self.conv1 = conv3x3(3, 16)

        # Three groups with n blocks each
        self.layer1 = self._make_layer(PreActBlock, widths[0], n, stride=1)
        self.layer2 = self._make_layer(PreActBlock, widths[1], n, stride=2)
        self.layer3 = self._make_layer(PreActBlock, widths[2], n, stride=2)

        # Final batch norm and linear layer
        self.bn_final = nn.BatchNorm2d(widths[2])
        self.avgpool  = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(widths[2] * PreActBlock.expansion, num_classes)

        # Initialize weights 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)  
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial conv
        out = self.conv1(x)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        # Final layers
        out = F.relu(self.bn_final(out))
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# Instantiate the model (e.g., depth=28, width=10 for WRN-28-10 equivalent)
def ResNet28_10(num_classes=10):
    return ResNet(depth=28, width=10, num_classes=num_classes)
def ResNet28_10_100(num_classes=100):
    return ResNet(depth=28, width=10, num_classes=num_classes)
# def test():
#     net = ResNet28_10()
#     y = net(Variable(torch.randn(1,3,32,32)))
#     print(y.size())
# if __name__=="__main__":
#     test()
# Example usage
# model = WideResNet28_10(num_classes=10)
# input = torch.randn(32, 3, 32, 32)  # CIFAR-10 batch
# output = model(input)
# print(output.shape)  # torch.Size([32, 10])