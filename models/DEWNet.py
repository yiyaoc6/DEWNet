import torch
from torch import nn
from torch.nn import functional as F

from module import WAFM
from module.LWT import LearnableWaveletTransformReg


# Branches (global, detail), ASPP, DAP, and fusion configuration.
VARIANTS = {
    "full": (True, True, True, True, "adaptive"),
    "dpe-only": (False, True, False, False, None),
    "gce-only": (True, False, False, False, None),
    "dual": (True, True, False, False, None),
    "dual-aspp": (True, True, True, False, None),
    "no-wafm": (True, True, True, True, None),
    "no-dap": (True, True, True, False, "adaptive"),
    "no-aspp": (True, True, False, True, "adaptive"),
    "fixed-haar": (True, True, True, True, "fixed-haar"),
    "equal-detail": (True, True, True, True, "equal-detail"),
}


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        mid_channels = out_channels if mid_channels is None else mid_channels
        super().__init__(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DetailAwarePooling(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.avg_pool = nn.AvgPool2d(2, stride=2)
        self.conv = nn.Conv2d(2 * channels, channels, 1)

    def forward(self, x):
        maximum = self.max_pool(x)
        difference = maximum - self.avg_pool(x)
        return self.conv(torch.cat((maximum, difference), dim=1))


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_dap):
        pool = DetailAwarePooling(in_channels) if use_dap else nn.MaxPool2d(2)
        super().__init__(pool, DoubleConv(in_channels, out_channels))


class DWConv(nn.Module):
    def __init__(self, channels, dilations=(1, 3, 5)):
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=rate, dilation=rate,
                          groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU6(inplace=True),
            )
            for rate in dilations
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(channels * len(dilations), channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return self.pointwise(torch.cat([branch(x) for branch in self.branches], dim=1))


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=6):
        super().__init__()
        hidden = in_channels * expand_ratio
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(),
            DWConv(hidden),
            nn.Conv2d(hidden, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.conv(x)


class LightEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_dap):
        super().__init__()
        self.down = DetailAwarePooling(in_channels) if use_dap else nn.MaxPool2d(2)
        self.conv1 = InvertedResidual(in_channels, out_channels)

    def forward(self, x):
        return self.conv1(self.down(x))


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x, skip):
        x = self.up(x)
        dy, dx = skip.shape[-2] - x.shape[-2], skip.shape[-1] - x.shape[-1]
        x = F.pad(x, (dx // 2, dx - dx // 2, dy // 2, dy - dy // 2))
        return self.conv(torch.cat((skip, x), dim=1))


class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=(kernel_size // 2) * dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class ASPP(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d((2, 2)), Conv2D(channels, channels, 1))
        self.c1 = Conv2D(channels, channels, 1)
        self.c2 = Conv2D(channels, channels, 3, dilation=6)
        self.c3 = Conv2D(channels, channels, 3, dilation=12)
        self.c4 = Conv2D(channels, channels, 3, dilation=18)
        self.c5 = Conv2D(5 * channels, channels, 1)

    def forward(self, x):
        pooled = F.interpolate(self.avgpool(x), size=x.shape[-2:], mode="bilinear", align_corners=True)
        return self.c5(torch.cat((pooled, self.c1(x), self.c2(x), self.c3(x), self.c4(x)), dim=1))


class Coarse_encoder(nn.Module):
    def __init__(self, variant):
        super().__init__()
        global_branch, detail_branch, aspp, use_dap, fusion = VARIANTS[variant]
        if global_branch:
            self.in_conv10 = InvertedResidual(1, 32)
            self.down11 = LightEncoderBlock(32, 64, use_dap)
            self.down12 = LightEncoderBlock(64, 128, use_dap)
            self.down13 = LightEncoderBlock(128, 256, use_dap)
        if detail_branch:
            self.in_conv20 = DoubleConv(1, 32)
            self.down21 = Down(32, 64, use_dap)
            self.down22 = Down(64, 128, use_dap)
            self.down23 = Down(128, 256, use_dap)
        if fusion:
            for level, channels in enumerate((32, 64, 128, 256)):
                setattr(self, f"wafm{level}", WAFM(
                    channels, learnable=fusion != "fixed-haar", adaptive=fusion != "equal-detail",
                ))
        self.aspp = ASPP(256) if aspp else nn.Identity()

    def forward(self, x):
        features = []
        for level in range(4):
            prefix = "in_conv" if level == 0 else "down"
            branches = [getattr(self, f"{prefix}{branch}{level}", None) for branch in (1, 2)]
            streams = [branch(x) for branch in branches if branch is not None]
            fuse = getattr(self, f"wafm{level}", None)
            x = fuse(*streams) if fuse is not None else sum(streams)
            features.append(x)
        return (*features, self.aspp(x))


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up1 = Up(256 + 256, 256)
        self.up2 = Up(256 + 128, 128)
        self.up3 = Up(128 + 64, 64)
        self.up4 = Up(64 + 32, 32)
        self.outconv = nn.Conv2d(32, 2, 1)

    def forward(self, f1, f2, f3, f4, bottleneck):
        x = self.up1(bottleneck, f4)
        x = self.up2(x, f3)
        x = self.up3(x, f2)
        return self.outconv(self.up4(x, f1))


class DEWNet(nn.Module):
    def __init__(self, variant="full"):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown DEWNet variant: {variant!r}")
        self.variant = variant
        self.coarse = Coarse_encoder(variant)
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder(*self.coarse(x))

    def orthogonality_loss(self):
        penalties = [
            layer.channel_orthogonality_penalties()
            for layer in self.modules()
            if isinstance(layer, LearnableWaveletTransformReg) and layer.kernel_LL.requires_grad
        ]
        return torch.cat(penalties).mean() if penalties else self.decoder.outconv.weight.new_zeros(())


Double_encodernet = DEWNet
