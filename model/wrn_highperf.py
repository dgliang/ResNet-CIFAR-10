import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BasicBlock(nn.Module):
    droprate = 0.0
    use_bn = True
    use_fixup = False
    fixup_l = 12

    def __init__(self, in_planes, out_planes, stride):
        super(BasicBlock, self).__init__()

        self.equalInOut = in_planes == out_planes
        self.conv_res = None if self.equalInOut else nn.Conv2d(
            in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)

        assert self.use_fixup or self.use_bn, "Either Fixup or BatchNorm must be enabled."

        if self.use_fixup:
            self.multiplicator = nn.Parameter(torch.ones(1, 1, 1, 1))
            self.biases = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, 1, 1)) for _ in range(4)])

            k = self.conv1.kernel_size[0] * self.conv1.kernel_size[1] * self.conv1.out_channels
            self.conv1.weight.data.normal_(0, self.fixup_l ** (-0.5) * math.sqrt(2. / k))
            self.conv2.weight.data.zero_()

            if self.conv_res is not None:
                k = self.conv_res.kernel_size[0] * self.conv_res.kernel_size[1] * self.conv_res.out_channels
                self.conv_res.weight.data.normal_(0, math.sqrt(2. / k))

    def forward(self, x):
        if self.use_bn:
            out = self.relu(self.bn1(x))
            out = self.relu(self.bn2(self.conv1(out)))
            if self.droprate > 0:
                out = F.dropout(out, p=self.droprate, training=self.training)
            out = self.conv2(out)
        else:
            out = self.relu(x + self.biases[0])
            out = self.conv1(out) + self.biases[1]
            out = self.relu(out) + self.biases[2]
            if self.droprate > 0:
                out = F.dropout(out, p=self.droprate, training=self.training)
            out = self.multiplicator * self.conv2(out) + self.biases[3]

        return x + out if self.equalInOut else self.conv_res(x) + out


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride):
        super(NetworkBlock, self).__init__()
        layers = []
        for i in range(nb_layers):
            layers.append(block(
                in_planes if i == 0 else out_planes,
                out_planes,
                stride if i == 0 else 1
            ))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes,
                 droprate=0.0, use_bn=True, use_fixup=False):
        super(WideResNet, self).__init__()

        assert (depth - 4) % 6 == 0, 'Depth must be 6n+4'
        n = (depth - 4) // 6
        k = widen_factor
        self.in_planes = 16

        nChannels = [16, 16*k, 32*k, 64*k]

        BasicBlock.droprate = droprate
        BasicBlock.use_bn = use_bn
        BasicBlock.use_fixup = use_fixup
        BasicBlock.fixup_l = n * 3

        block = BasicBlock

        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2)
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                k = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / k))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
                if use_fixup:
                    m.weight.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return self.fc(out)


def WideResNet_28_10(num_classes=10, droprate=0.3, use_bn=True, use_fixup=False):
    return WideResNet(depth=28, widen_factor=10, num_classes=num_classes,
                      droprate=droprate, use_bn=use_bn, use_fixup=use_fixup)
