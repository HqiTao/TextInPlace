import torch
import torch.nn as nn
import torch.nn.functional as F

class mish(nn.Module):
    def __init__(self, ):
        super(mish, self).__init__()
        self.activated = True

    def forward(self, x):
        if self.activated:
            x = x * (torch.tanh(F.softplus(x)))
        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.prelu = mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = nn.MaxPool2d(2,2)
        self.convblock  = ResidualBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.downsample(x)
        x = self.convblock(x)
        return x

def shortcut(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
        nn.BatchNorm2d(out_channels)
    )

class SE_Block(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        x = x * y.expand_as(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.hdc = nn.Sequential(
                ConvBlock(in_channels, out_channels, padding=1, dilation=1),
                ConvBlock(out_channels, out_channels, padding=2, dilation=2),
                ConvBlock(out_channels, out_channels, padding=5, dilation=5, with_nonlinearity=False)
            )
        self.shortcut = shortcut(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU()
        self.se = SE_Block(c=out_channels)

    def forward(self, x):
        res = self.shortcut(x)
        x   = self.se(self.hdc(x))
        x   = self.relu(res + x)
        return x

def switchLayer(xs):
    numofeature = len(xs)
    splitxs = []
    for i in range(numofeature):
        splitxs.append(
            list(torch.chunk(xs[i], numofeature, dim = 1))
        )
    
    for i in range(numofeature):
        h,w = splitxs[i][i].shape[2:]
        tmp = []
        for j in range(numofeature):
            if i > j:
                splitxs[j][i] = F.interpolate(splitxs[j][i], (h,w))
            elif i < j: 
                splitxs[j][i] = F.avg_pool2d(splitxs[j][i], kernel_size = (2*(j-i)))
            tmp.append(splitxs[j][i])
        xs[i] = torch.cat(tmp, dim = 1)

    return xs
    
class FSNet(nn.Module):
    def __init__(self, channels = 768):
        super(FSNet, self).__init__()
        self.channels = channels
        self.blocks = nn.ModuleList()
        self.steps = nn.ModuleList()

        self.blocks.append(ResidualBlock(768, 384))
        self.steps.append(UpBlock(768, 768*2))
        self.blocks.append(ResidualBlock(384, 384))

        self.blocks.append(ResidualBlock(384, 192))
        self.blocks.append(ResidualBlock(384, 192))
        self.steps.append(UpBlock(384, 768))
        self.blocks.append(ResidualBlock(192, 192))

        self.blocks.append(ResidualBlock(192, 256))
        self.blocks.append(ResidualBlock(192, 256))
        self.blocks.append(ResidualBlock(192, 256))

        self.blocks.append(DownBlock(1024, 256))

    def forward(self, features):
        x = features[-1]
        y = features[0]

        x1 = self.blocks[0](x)
        x2 = self.steps[0](x)
        x2 = self.blocks[1](x2)
        x3,x4 = switchLayer([x1,x2])

        x3 = self.blocks[2](x3)
        x4 = self.blocks[3](x4)
        x5 = self.steps[1](x2)
        x5 = self.blocks[4](x5)
        x6,x7,x8 = switchLayer([x3,x4,x5])

        x6 = self.blocks[5](x6) # 14
        x7 = self.blocks[6](x7) # 7
        x8 = self.blocks[7](x8) # 3.5

        x9 = torch.cat([x6, y], dim=1)
        x9 = self.blocks[8](x9) # 28

        return x9, x6, x7, x8