"""DepthForge-Net: a small RGB-guided depth super-resolution CNN.

Design (≈0.5 M parameters, scale-agnostic):
  * Input = coarse depth already at HR size (bicubic or classical JBU output)
    + HR RGB guide. Because upsampling happens *before* the net, one model
    serves 2x / 4x / 8x.
  * Two-stream encoder: depth stream and RGB stream. RGB features are injected
    through a learned sigmoid *gate* computed from depth features, so the
    network can learn to ignore RGB texture that has no depth edge
    (anti texture-copying).
  * U-Net decoder predicts a *residual* correction on top of the input depth,
    so the classical prior is kept and the net only learns what it fixes.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(cin, cout, s=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, s, 1), nn.GroupNorm(8, cout), nn.ReLU(inplace=True))


class GatedFuse(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(2 * c, c, 1), nn.Sigmoid())
        self.mix = conv(2 * c, c)

    def forward(self, fd, fr):
        g = self.gate(torch.cat([fd, fr], 1))
        return self.mix(torch.cat([fd, g * fr], 1))


class DepthForgeNet(nn.Module):
    def __init__(self, c=32):
        super().__init__()
        self.d1, self.r1 = conv(1, c), conv(3, c)
        self.f1 = GatedFuse(c)
        self.d2, self.r2 = conv(c, 2 * c, 2), conv(c, 2 * c, 2)
        self.f2 = GatedFuse(2 * c)
        self.d3 = nn.Sequential(conv(2 * c, 4 * c, 2), conv(4 * c, 4 * c))
        self.u2 = conv(4 * c + 2 * c, 2 * c)
        self.u1 = conv(2 * c + c, c)
        self.out = nn.Conv2d(c, 1, 3, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, depth, rgb):
        # depth: Bx1xHxW normalised to ~[0,1]; rgb: Bx3xHxW in [0,1]
        H, W = depth.shape[-2:]
        ph, pw = (-H) % 4, (-W) % 4
        if ph or pw:
            depth = F.pad(depth, (0, pw, 0, ph), mode="replicate")
            rgb = F.pad(rgb, (0, pw, 0, ph), mode="replicate")
        fd1, fr1 = self.d1(depth), self.r1(rgb)
        e1 = self.f1(fd1, fr1)
        fd2, fr2 = self.d2(e1), self.r2(fr1)
        e2 = self.f2(fd2, fr2)
        e3 = self.d3(e2)
        x = F.interpolate(e3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.u2(torch.cat([x, e2], 1))
        x = F.interpolate(x, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.u1(torch.cat([x, e1], 1))
        out = depth + self.out(x)
        return out[..., :H, :W]


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
