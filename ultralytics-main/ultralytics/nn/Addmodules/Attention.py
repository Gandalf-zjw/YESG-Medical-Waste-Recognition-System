
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return x * self.act(self.fc(self.pool(x)))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        if not isinstance(kernel_size, int) or kernel_size not in (3, 7):
            print(f"⚠️ Invalid kernel_size={kernel_size}, fallback to 7.")
            kernel_size = 7
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.act(self.conv(x_cat))
        return x * attention


class CBAM(nn.Module):
    def __init__(self, c1, *args):
        super().__init__()
        kernel_size = args[0] if len(args) > 0 and isinstance(args[0], int) else 7
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.spatial_attention(self.channel_attention(x))

class ECA(nn.Module):
    def __init__(self, c1, *args):
        super(ECA, self).__init__()
        k_size = args[0] if len(args) > 0 and isinstance(args[0], int) else 3
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        if y.shape[1] != x.shape[1]:
            y = F.interpolate(y, size=(x.shape[1], 1, 1), mode='nearest')
        return x * y.expand_as(x)

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, c1, *args):
        super(CoordAtt, self).__init__()
        reduction = args[0] if len(args) > 0 and isinstance(args[0], int) else 32
        oup = c1
        mip = max(8, c1 // reduction)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(c1, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return identity * a_w * a_h

class GAM(nn.Module):
    def __init__(self, c1, *args):
        super().__init__()
        rate = args[0] if len(args) > 0 and isinstance(args[0], int) else 4
        in_channels = int(c1)
        out_channels = int(c1)
        inter_channels = max(1, int(in_channels / rate))

        self.linear1 = nn.Linear(in_channels, inter_channels)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(inter_channels, in_channels)

        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=7, padding=3, padding_mode='replicate')
        self.conv2 = nn.Conv2d(inter_channels, out_channels, kernel_size=7, padding=3, padding_mode='replicate')

        self.norm1 = nn.BatchNorm2d(inter_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_channel_att = self.linear2(self.relu(self.linear1(x_permute))).view(b, h, w, c).permute(0, 3, 1, 2)
        x = x * x_channel_att
        x_spatial_att = self.relu(self.norm1(self.conv1(x)))
        x_spatial_att = self.sigmoid(self.norm2(self.conv2(x_spatial_att)))
        return x * x_spatial_att
