"""Dual-state evidence restorer E_phi (paper Sec. 4.2, Eq. 4-5 and 7-10).

For l in {N3, N4}:
    Z~_d = (Z_d - mu) / sigma,   Z~_r = (Z_r - mu) / sigma          (shared, frozen stats; Eq. 7)
    U    = [Z~_r, Z~_d, Z~_r - Z~_d, |Z~_r - Z~_d|]                   (Eq. 8)
    h_l  = ResBlocks(Conv1x1(U))      with one-way N4 -> N3 fusion
    r_l  = b * tanh(H_l(h_l)),  b = 3                                (Eq. 9)
    Z_c  = Z_r + sigma * r_l                                          (Eq. 10)
N5 follows the restored path unchanged (Eq. 5).

(mu, sigma) are precomputed once from restored-state training features
(tools/precompute_neck_stats.py) and stored as buffers in the checkpoint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8 if channels % 8 == 0 else 1, channels)
        self.body = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(channels, channels, 3, padding=1))
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.body(self.norm(x))


class SharedStandardizer(nn.Module):
    """Fixed per-channel (mu, sigma) shared by the degraded and restored states."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mean", mean.reshape(1, -1, 1, 1).clone())
        self.register_buffer("std", std.reshape(1, -1, 1, 1).clamp_min(1e-4).clone())

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def denormalize_residual(self, r: torch.Tensor) -> torch.Tensor:
        return r * self.std


# Dual-state input variants (ablation): number of stacked channel groups.
_INPUT_GROUPS = {"full": 4, "signed": 3, "rd": 2, "r_only": 1}


class DualStateEncoder(nn.Module):
    def __init__(self, mean, std, width: int, blocks: int, norm_mode: str = "shared", input_mode: str = "full"):
        super().__init__()
        if norm_mode not in ("shared", "independent"):
            raise ValueError(f"unknown norm_mode: {norm_mode}")
        if input_mode not in _INPUT_GROUPS:
            raise ValueError(f"unknown input_mode: {input_mode}")
        channels = mean.numel()
        self.norm_mode, self.input_mode = norm_mode, input_mode
        # Always kept: the residual is mapped back to native units with this sigma.
        self.standardizer = SharedStandardizer(mean, std)
        if norm_mode == "independent":  # ablation: per-sample whitening of each state
            self.degraded_norm = nn.GroupNorm(1, channels, affine=False)
            self.restored_norm = nn.GroupNorm(1, channels, affine=False)
        self.stem = nn.Conv2d(channels * _INPUT_GROUPS[input_mode], width, 1)
        self.blocks = nn.Sequential(*[ConvResidualBlock(width) for _ in range(blocks)])

    def forward(self, degraded: torch.Tensor, restored: torch.Tensor) -> torch.Tensor:
        degraded, restored = degraded.detach(), restored.detach()
        if self.norm_mode == "shared":
            d, r = self.standardizer.normalize(degraded), self.standardizer.normalize(restored)
        else:
            d, r = self.degraded_norm(degraded), self.restored_norm(restored)
        if self.input_mode == "r_only":
            u = r
        elif self.input_mode == "rd":
            u = torch.cat((r, d), dim=1)
        elif self.input_mode == "signed":
            u = torch.cat((r, d, r - d), dim=1)
        else:
            diff = r - d
            u = torch.cat((r, d, diff, diff.abs()), dim=1)
        return self.blocks(self.stem(u))


@dataclass(frozen=True)
class EvidenceRestorerConfig:
    n3_layer: int = 15
    n4_layer: int = 18
    n5_layer: int = 21
    n3_channels: int = 128
    n4_channels: int = 256
    width: int = 96
    blocks_per_level: int = 2
    normalized_bound: float = 3.0  # b
    # Ablation switches; defaults are the paper design.
    norm_mode: str = "shared"
    input_mode: str = "full"
    use_cross_scale: bool = True
    bounded: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "EvidenceRestorerConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in known})


class EvidenceRestorer(nn.Module):
    def __init__(self, config: EvidenceRestorerConfig, stats: dict | None = None) -> None:
        """stats: {str(layer): {"mean": (C,), "std": (C,)}}; omit when loading a checkpoint."""
        super().__init__()
        self.config = config
        cfg = config

        def level_stats(layer, channels):
            if stats is None:
                return torch.zeros(channels), torch.ones(channels)
            entry = stats[str(layer)]
            if entry["mean"].numel() != channels:
                raise ValueError(f"stat channel mismatch at layer {layer}")
            return entry["mean"].float(), entry["std"].float()

        self.n3_encoder = DualStateEncoder(*level_stats(cfg.n3_layer, cfg.n3_channels), cfg.width,
                                           cfg.blocks_per_level, cfg.norm_mode, cfg.input_mode)
        self.n4_encoder = DualStateEncoder(*level_stats(cfg.n4_layer, cfg.n4_channels), cfg.width,
                                           cfg.blocks_per_level, cfg.norm_mode, cfg.input_mode)
        if cfg.use_cross_scale:
            self.n4_to_n3 = nn.Conv2d(cfg.width, cfg.width, 1)
            self.n3_fusion = nn.Sequential(nn.Conv2d(cfg.width * 2, cfg.width, 1), nn.GELU(),
                                           ConvResidualBlock(cfg.width))
        self.n3_head = nn.Conv2d(cfg.width, cfg.n3_channels, 1)
        self.n4_head = nn.Conv2d(cfg.width, cfg.n4_channels, 1)
        for head in (self.n3_head, self.n4_head):  # zero-init => identity correction at start
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, degraded: dict[int, torch.Tensor], restored: dict[int, torch.Tensor]) -> dict:
        """degraded / restored: N3/N4/N5 neck pyramids of x_d and x_r.

        Returns the residuals in native units (delta_n3/n4), the normalized
        residuals r_l (norm_n3/n4), and the corrected pyramid Z_c ("features")
        with the restored state detached. For end-to-end training use
        `correct(restored, out)` to keep gradients to the pixel restorer.
        """
        cfg = self.config
        h3 = self.n3_encoder(degraded[cfg.n3_layer], restored[cfg.n3_layer])
        h4 = self.n4_encoder(degraded[cfg.n4_layer], restored[cfg.n4_layer])
        if cfg.use_cross_scale:  # one-way N4 -> N3
            semantic = F.interpolate(self.n4_to_n3(h4), size=h3.shape[-2:], mode="bilinear", align_corners=False)
            h3 = h3 + self.n3_fusion(torch.cat((h3, semantic), dim=1))
        r3, r4 = self.n3_head(h3), self.n4_head(h4)
        if cfg.bounded:
            r3 = cfg.normalized_bound * torch.tanh(r3)
            r4 = cfg.normalized_bound * torch.tanh(r4)
        out = {
            "delta_n3": self.n3_encoder.standardizer.denormalize_residual(r3),
            "delta_n4": self.n4_encoder.standardizer.denormalize_residual(r4),
            "norm_n3": r3,
            "norm_n4": r4,
        }
        out["features"] = self.correct({k: v.detach() for k, v in restored.items()}, out)
        return out

    def correct(self, restored: dict[int, torch.Tensor], out: dict) -> dict[int, torch.Tensor]:
        cfg = self.config
        return {
            cfg.n3_layer: restored[cfg.n3_layer] + out["delta_n3"],
            cfg.n4_layer: restored[cfg.n4_layer] + out["delta_n4"],
            cfg.n5_layer: restored[cfg.n5_layer],
        }
