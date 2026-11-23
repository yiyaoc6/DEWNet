import torch
from torch import nn
from torch.nn import functional as F


class LearnableWaveletTransformReg(nn.Module):
    """Per-channel Haar-initialized analysis and shared-weight adjoint synthesis."""

    def __init__(self, in_channels, learnable=True):
        super().__init__()
        self.in_channels = in_channels
        # Keep the source checkpoint's parameter names and LL/LH/HL/HH ordering.
        low = torch.tensor([1.0, 1.0]) / (2 ** 0.5)
        high = torch.tensor([1.0, -1.0]) / (2 ** 0.5)
        for name, row, column in (
            ("kernel_LL", low, low),
            ("kernel_LH", low, high),
            ("kernel_HL", high, low),
            ("kernel_HH", high, high),
        ):
            kernel = torch.outer(row, column)[None, None].repeat(in_channels, 1, 1, 1)
            self.register_parameter(name, nn.Parameter(kernel, requires_grad=learnable))

    def filters(self):
        return self.kernel_LL, self.kernel_LH, self.kernel_HL, self.kernel_HH

    def forward(self, x):
        if any(size % 2 for size in x.shape[-2:]):
            raise ValueError("Wavelet analysis requires even feature-map dimensions.")
        ll, lh, hl, hh = (
            F.conv2d(x, kernel, stride=2, groups=self.in_channels)
            for kernel in self.filters()
        )
        return ll, torch.cat((lh, hl, hh), dim=1)

    def synthesis(self, ll, hf):
        lh, hl, hh = hf.chunk(3, dim=1)
        ll_map, lh_map, hl_map, hh_map = (
            F.conv_transpose2d(band, kernel, stride=2, groups=self.in_channels)
            for band, kernel in zip((ll, lh, hl, hh), self.filters())
        )
        return ll_map + lh_map + hl_map + hh_map

    def channel_orthogonality_penalties(self):
        """Return one squared Gram-matrix error per channel (Equation 5)."""
        filters = torch.stack([kernel.flatten(1) for kernel in self.filters()], dim=1)
        gram = filters @ filters.transpose(-1, -2)
        identity = torch.eye(4, dtype=filters.dtype, device=filters.device)
        return (gram - identity).square().sum(dim=(-2, -1))
