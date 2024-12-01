import torch
import torch.nn as nn
import torch.nn.functional as F


class Squeeze_Excitation(nn.Module):
    def __init__(self, channel, r=8):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        b, c, _, _ = inputs.shape
        x = self.pool(inputs).view(b, c)
        x = self.net(x).view(b, c, 1, 1)
        x = inputs * x
        return x


class Stem(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.c1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.MaxPool2d((out_c // 16, out_c // 16))
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_c),
            nn.MaxPool2d((out_c // 16, out_c // 16))
        )

    def forward(self, inputs):
        x = self.c1(inputs)
        s = self.c2(inputs)
        y = x + s
        return y


class Stem_Block(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_c),
        )

        self.attn = Squeeze_Excitation(out_c)

    def forward(self, inputs):
        x = self.c1(inputs)
        s = self.c2(inputs)
        y = self.attn(x + s)
        return y


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, stride):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = round(in_channels * expansion_factor)

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.residual(x)
        out = out + self.shortcut(x)
        return out


class Block(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_c),
        )

        self.attn = Squeeze_Excitation(out_c)

    def forward(self, inputs):
        x = self.c1(inputs)
        s = self.c2(inputs)
        y = self.attn(x + s)
        return y


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtV2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm1 = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm1(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return x


class ASPP(nn.Module):
    def __init__(self, in_c, out_c, rate=[1, 6, 12, 18, 24]):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[0], padding=rate[0]),
            nn.BatchNorm2d(out_c)
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[1], padding=rate[1]),
            nn.BatchNorm2d(out_c)
        )

        self.c3 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[2], padding=rate[2]),
            nn.BatchNorm2d(out_c)
        )

        self.c4 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[3], padding=rate[3]),
            nn.BatchNorm2d(out_c)
        )

        self.c5 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[4], padding=rate[4]),
            nn.BatchNorm2d(out_c)
        )

        self.c6 = nn.Conv2d(out_c, out_c, kernel_size=1, padding=0)
        self.c7 = nn.BatchNorm2d(out_c)
        self.c8 = nn.Conv2d(out_c * 5, out_c, kernel_size=1, padding=0)

    def forward(self, inputs):
        x1 = self.c1(inputs)
        x2 = self.c2(inputs)
        x3 = self.c3(inputs)
        x4 = self.c4(inputs)
        x5 = self.c5(inputs)
        x = torch.cat([x1, x2, x3, x4, x5], axis=1)
        x = self.c8(x)
        x = self.c7(x)
        return x

class Attention_d(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        out_c = in_c[1]

        self.x_conv = nn.Sequential(
            nn.BatchNorm2d(in_c[1]),
            nn.ReLU(),
            InvertedResidual(in_c[1], out_c // 2, expansion_factor=1, stride=1),
        )

        self.gc_conv = nn.Sequential(
            nn.BatchNorm2d(out_c // 2),
            nn.ReLU(),
            ConvNeXtV2(out_c // 2)
        )

    def forward(self, g, x):
        x_conv = self.x_conv(x)
        gc_sum = g + x_conv
        y = self.gc_conv(gc_sum)
        return y


class Decoder(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.a1 = Attention_d(in_c)
        self.r1 = Block(out_c * 2, out_c, stride=1)

    def forward(self, g, x):
        d = self.a1(g, x)
        d = torch.cat([d, g], axis=1)
        d = self.r1(d)
        return d


def get_bn(dim, use_sync_bn=False):
    if use_sync_bn:
        return nn.SyncBatchNorm(dim)
    else:
        return nn.BatchNorm2d(dim)


class Encode(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.x_conv = nn.Sequential(
            nn.BatchNorm2d(in_c[1]),
            nn.ReLU(),
            nn.Conv2d(in_c[1], out_c, kernel_size=3, padding=1),
            nn.MaxPool2d((2, 2))
        )
        self.down = Stem(in_c[0], out_c)
        self.r1 = Block(out_c * 2, out_c, stride=1)

    def forward(self, g, x):
        b = self.x_conv(x)
        a = self.down(g)

        d = torch.cat([a, b], axis=1)
        d = self.r1(d)
        return d

class m(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # 添加自适应平均池化层
        )

        self.fc = nn.Linear(out_c, 2)

    def forward(self, inputs):
        x = self.c1(inputs)
        x = x.view(x.size(0), -1)  # 将特征图展平为一维张量
        x = self.fc(x)
        return x

class resunetplusplus(nn.Module):
    def __init__(self):
        super().__init__()
        nb_filter = [16, 32, 64, 128, 256]
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv0_0 = Stem_Block(3, 16, stride=1)
        self.conv1_0 = Encode([3, 16], 32)
        self.conv2_0 = Encode([3, 32], 64)
        self.conv3_0 = Encode([3, 64], 128)
        self.conv4_0 = Encode([3, 128], 256)

        self.conv3_1 = Decoder([256, 512], 256)
        self.conv2_2 = Decoder([128, 256], 128)
        self.conv1_3 = Decoder([64, 128], 64)
        self.conv0_4 = Decoder([32, 64], 32)
        self.aspp1 = ASPP(256, 512)
        self.aspp2 = ASPP(32, 16)
        self.output = m(16, 7)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(input, x0_0)
        x2_0 = self.conv2_0(input, x1_0)
        x3_0 = self.conv3_0(input, x2_0)
        x4_0 = self.conv4_0(input, x3_0)

        b1 = self.aspp1(x4_0)

        x3_1 = self.conv3_1(x4_0, b1)
        x2_2 = self.conv2_2(x3_0, self.up(x3_1))
        x1_3 = self.conv1_3(x2_0, self.up(x2_2))
        x0_4 = self.conv0_4(x1_0, self.up(x1_3))

        output = self.aspp2(x0_4)
        output = self.output(output)
        return output
