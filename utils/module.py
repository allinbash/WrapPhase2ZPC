import torch
from torch import nn


# ----------------------------------------------------------- ZnkCNN -----------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ConvBlock, self).__init__()
        self.conv_left = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=stride),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=4 * out_channels, kernel_size=(1, 1)),
            # nn.BatchNorm2d(4 * out_channels)
        )
        self.conv_right = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=4 * out_channels, kernel_size=(1, 1), stride=stride),
            # nn.BatchNorm2d(4 * out_channels)
        )

    def forward(self, x):
        left = self.conv_left(x)
        right = self.conv_right(x)
        added = left.add(right)
        output = nn.ReLU(inplace=True)(added)

        return output


class IdentityBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(IdentityBlock, self).__init__()
        self.conv_identity = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1)),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=(1, 1)),
            # nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        y = self.conv_identity(x)
        ident = y.add(x)
        output = nn.ReLU(inplace=True)(ident)

        return output


class ZnkCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ZnkCNN, self).__init__()
        self.stage_in = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=3),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        )
        self.stage1 = nn.Sequential(
            ConvBlock(64, 64, stride=(1, 1)),
            IdentityBlock(256, 64),
            IdentityBlock(256, 64)
        )
        self.stage2 = nn.Sequential(
            ConvBlock(256, 128, stride=(2, 2)),
            IdentityBlock(512, 128),
            IdentityBlock(512, 128),
            IdentityBlock(512, 128)
        )
        self.stage3 = nn.Sequential(
            ConvBlock(512, 256, stride=(2, 2)),
            IdentityBlock(1024, 256),
            IdentityBlock(1024, 256),
            IdentityBlock(1024, 256),
            IdentityBlock(1024, 256),
            IdentityBlock(1024, 256)
        )
        self.stage4 = nn.Sequential(
            ConvBlock(1024, 512, stride=(2, 2)),
            IdentityBlock(2048, 512),
            IdentityBlock(2048, 512),
            IdentityBlock(2048, 512)
        )
        self.stage5 = nn.Sequential(
            ConvBlock(2048, 1024, stride=(2, 2)),
            IdentityBlock(4096, 1024),
            IdentityBlock(4096, 1024)
        )
        self.stage_out = nn.Sequential(
            nn.MaxPool2d(kernel_size=(8, 8)),
            nn.Flatten(),
            nn.Linear(4096, out_channels)
        )

    def forward(self, x):
        y = self.stage_in(x)
        y = self.stage1(y)
        y = self.stage2(y)
        y = self.stage3(y)
        y = self.stage4(y)
        y = self.stage5(y)
        output = self.stage_out(y)

        return output


if __name__ == '__main__':
    net = ZnkCNN(1, 4)
    print(net)
