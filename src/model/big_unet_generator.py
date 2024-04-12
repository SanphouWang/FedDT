import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock


class DownConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(DownConvBlock, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)

    def forward(self, x):
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv(x)
        return x


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(UpConvBlock, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.upconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=4, stride=stride, padding=1
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.relu(x)
        x = self.upconv(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
    ):
        super(ResidualBlock, self).__init__()
        out_channels = out_channels if out_channels else in_channels
        self.normalize1 = nn.BatchNorm2d(in_channels)
        self.non_linear = nn.ReLU(inplace=True)
        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.normalize2 = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        output = self.normalize1(x)
        output = self.non_linear(output)
        output = self.conv0(output)
        output = self.normalize2(output)
        output = self.non_linear(output)
        output = self.conv1(output)
        shortcut = x
        return shortcut + output


class ResidualBlock_2(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ResidualBlock_2, self).__init__()
        out_channels = out_channels if out_channels else in_channels
        self.residual_block = ResidualBlock(in_channels)
        self.normalize1 = nn.BatchNorm2d(in_channels)
        self.non_linear = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        output = self.residual_block(x)
        output = self.normalize1(output)
        output = self.non_linear(output)
        output = self.conv(output)
        return output


class BigUNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, symmetric=True):
        super(BigUNetGenerator, self).__init__()
        # Define the architecture here
        self.symmetric = symmetric
        self.initial_conv = DownConvBlock(in_channels, 64, stride=1)
        if self.symmetric:
            self.down1 = DownConvBlock(64, 128)
            self.down2 = DownConvBlock(128, 256)
            self.residual_block_down1 = ResidualBlock(256)
            self.down3 = DownConvBlock(256, 512)
            self.residual_block_down2 = ResidualBlock(512)
            self.down4 = DownConvBlock(512, 512)
            self.residual_block_conv_1 = ResidualBlock_2(512)
            self.residual_block_conv_2 = ResidualBlock_2(512)
            self.residual_block_conv_3 = ResidualBlock_2(512)
            self.residual_block_up1 = ResidualBlock(1024)
            self.up1 = UpConvBlock(1024, 512)
            self.residual_block_up2 = ResidualBlock(1024)
            self.up2 = UpConvBlock(1024, 256)
            self.up3 = UpConvBlock(512, 128)
            self.up4 = UpConvBlock(256, 64)
        else:
            self.residual_block_down1 = ResidualBlock(64)
            self.down1 = DownConvBlock(64, 128)
            self.residual_block_down2 = ResidualBlock(128)
            self.down2 = DownConvBlock(128, 256)
            self.residual_block_down3 = ResidualBlock(256)
            self.down3 = DownConvBlock(256, 512)
            self.residual_block_down4 = ResidualBlock(512)
            self.down4 = DownConvBlock(512, 512)

            self.residual_block_conv_1 = ResidualBlock_2(512)
            self.residual_block_conv_2 = ResidualBlock_2(512)
            self.residual_block_conv_3 = ResidualBlock_2(512)

            # self.residual_block_up0 = ResidualBlock_2(1024)
            self.up1 = UpConvBlock(1024, 512)
            # self.residual_block_up1 = ResidualBlock_2(1024)
            self.up2 = UpConvBlock(1024, 256)
            # self.residual_block_up2 = ResidualBlock_2(512)
            self.up3 = UpConvBlock(512, 128)
            # self.residual_block_up3 = ResidualBlock_2(256)
            self.up4 = UpConvBlock(256, 64)
        self.final_conv = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        # Initial convolution
        x0 = self.initial_conv(x)
        if self.symmetric:
            x1 = self.down1(x0)
            x2 = self.down2(x1)
            x3 = self.residual_block_down1(x2)
            x3 = self.down3(x3)
            x4 = self.residual_block_down2(x3)
            x4 = self.down4(x4)
            x5 = self.residual_block_conv_1(x4)
            x5 = self.residual_block_conv_2(x5)
            x5 = self.residual_block_conv_3(x5)
            x6 = self.residual_block_up1(torch.cat((x5, x4), dim=1))
            x7 = self.up1(x6)
            x7 = self.residual_block_up2(torch.cat((x7, x3), dim=1))
            x8 = self.up2(x7)
            x9 = self.up3(torch.cat((x8, x2), dim=1))
            out = self.up4(torch.cat((x9, x1), dim=1))
            out = self.final_conv(torch.cat((out, x0), dim=1))

        else:
            x1 = self.residual_block_down1(x0)
            x1 = self.down1(x1)
            x2 = self.residual_block_down2(x1)
            x2 = self.down2(x2)
            x3 = self.residual_block_down3(x2)
            x3 = self.down3(x3)
            x4 = self.residual_block_down4(x3)
            x4 = self.down4(x4)
            x5 = self.residual_block_conv_1(x4)
            x5 = self.residual_block_conv_2(x5)
            x5 = self.residual_block_conv_3(x5)
            # x6 = self.residual_block_up0(torch.cat((x5, x4), dim=1))
            # x7 = self.up1(x6)
            # x7 = self.residual_block_up1(torch.cat((x7, x3), dim=1))
            # x8 = self.up2(x7)
            # x8 = self.residual_block_up2(torch.cat((x8, x2), dim=1))
            # x9 = self.up3(x8)
            # x9 = self.residual_block_up3(torch.cat((x9, x1), dim=1))
            # out = self.up4(x9)
            x7 = self.up1(torch.cat((x5, x4), dim=1))
            x8 = self.up2(torch.cat((x7, x3), dim=1))
            x9 = self.up3(torch.cat((x8, x2), dim=1))
            out = self.up4(torch.cat((x9, x1), dim=1))
            out = self.final_conv(torch.cat((out, x0), dim=1))

        return out


if __name__ == "__main__":
    model = BigUNetGenerator()
    input_tensor = torch.rand(30, 1, 256, 256)
    output = model(input_tensor)
    print(output.shape)
