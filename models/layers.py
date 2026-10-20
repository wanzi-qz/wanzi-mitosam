"""Attention and adapter modules used by MitoSAM.

The thesis contribution is small and surgical: keep SAM's heavy ViT image
encoder frozen (2.5 GB ViT-H, trained on 1B masks) and teach only the cheap
parts that matter for mitochondria:

1. a **ResNet-CBAM adapter** that re-weights SAM's image embedding so that
   thin, low-contrast, high-frequency cristae survive the 16x downsampling of
   the ViT patch embedding;
2. the **mask decoder**, fully fine-tuned with a dice + BCE objective.

Freezing the backbone makes the fine-tune fit in a single 8 GB GPU and keeps
the number of trainable parameters ~3% of the full model.
"""

from __future__ import annotations

from typing import Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH = True
except ImportError:  # pragma: no cover
    _TORCH = False
    nn = object  # type: ignore


if _TORCH:

    class CBAM(nn.Module):
        """Convolutional Block Attention Module (channel + spatial attention).

        Woo et al., ECCV 2018. Channel attention answers *what* to look at,
        spatial attention answers *where* — the second is what restores the
        branch points of the mitochondrial network.
        """

        def __init__(self, channels: int, reduction: int = 8, kernel_size: int = 7) -> None:
            super().__init__()
            hidden = max(1, channels // reduction)
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, hidden, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, channels, 1, bias=False),
            )
            self.spatial_attention = nn.Sequential(
                nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
                nn.Sigmoid(),
            )
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            # --- channel branch (shared MLP over avg & max descriptors)
            avg_pool = self.channel_attention(x)
            max_pool = self.channel_attention(torch.amax(x, dim=(2, 3), keepdim=True))
            x = x * self.sigmoid(avg_pool + max_pool)
            # --- spatial branch
            avg_map = torch.mean(x, dim=1, keepdim=True)
            max_map = torch.amax(x, dim=1, keepdim=True)
            x = x * self.spatial_attention(torch.cat([avg_map, max_map], dim=1))
            return x


    class ResidualBlock(nn.Module):
        """Pre-activation-free residual block with optional CBAM."""

        def __init__(self, in_ch: int, out_ch: int, stride: int = 1, use_cbam: bool = True) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_ch)
            self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_ch)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = None
            if stride != 1 or in_ch != out_ch:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm2d(out_ch)
                )
            self.cbam = CBAM(out_ch) if use_cbam else nn.Identity()

        def forward(self, x):
            residual = x if self.downsample is None else self.downsample(x)
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return self.relu(self.cbam(out + residual))


    class EmbeddingAdapter(nn.Module):
        """ResNet-CBAM block stack operating on the SAM image embedding (B,256,H/16,W/16).

        Colours are channelled through a residual path so the module can start as
        an identity mapping — important when fine-tuning from a strong checkpoint.
        """

        def __init__(self, channels: int = 256, n_blocks: int = 2, use_cbam: bool = True) -> None:
            super().__init__()
            blocks: list[nn.Module] = []
            for _ in range(n_blocks):
                blocks.append(ResidualBlock(channels, channels, use_cbam=use_cbam))
            self.blocks = nn.Sequential(*blocks)
            # Learnable gate initialised at 0 -> adapter is identity at step 0
            self.gate = nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return x + self.gate * self.blocks(x)


    class DiceBCELoss(nn.Module):
        """Dice + BCE, evaluated directly on logits."""

        def __init__(self, dice_weight: float = 0.5) -> None:
            super().__init__()
            self.dice_weight = dice_weight
            self.bce = nn.BCEWithLogitsLoss()

        @staticmethod
        def _dice(logits, target, eps: float = 1.0):
            prob = torch.sigmoid(logits)
            num = 2 * (prob * target).sum(dim=(2, 3)) + eps
            den = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
            return (1 - num / den).mean()

        def forward(self, logits, target):
            if target.shape[-2:] != logits.shape[-2:]:
                target = F.interpolate(target, size=logits.shape[-2:], mode="nearest")
            return self.bce(logits, target) + self.dice_weight * self._dice(logits, target)


def count_parameters(module) -> Tuple[int, int]:
    """Return ``(total, trainable)`` parameter counts."""
    if not _TORCH:
        return (0, 0)
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable
