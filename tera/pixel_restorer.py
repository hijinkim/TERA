"""Task-guided pixel restorer R_theta (paper Sec. 4.1, Eq. 2 and 6).

x_r = clip(x_d + g_s(x_d, E_d) * a(x_d, E_d) * Delta_rgb(x_d, E_d), 0, 1)

* E_d: frozen-detector backbone P3/P4 features of the received image x_d.
* Delta_rgb = tanh(.) is produced by a NAFNet-style encoder-decoder whose blocks
  are modulated by (i) a blind degradation code predicted from x_d and (ii) a
  task-guidance map/context computed from E_d.
* a = a_max * sigmoid(.) is a per-pixel, per-channel RGB amplitude (a_max = 0.2).
* g_s in [0, 1] is a spatial gate (modulated by an image-level gate), predicted from
  the degradation code and the task-guidance residuals.

The restorer only ever consumes the decoded RGB image and detector features;
no transmission metadata enters the forward pass.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

P3_LAYER, P4_LAYER = 4, 6  # YOLOv8 backbone indices of P3 / P4


def _logit(p: float) -> float:
    return torch.logit(torch.tensor(p)).item()


def _resize(x: torch.Tensor, size) -> torch.Tensor:
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).square().mean(dim=1, keepdim=True)
        return (x - mean) * torch.rsqrt(variance + self.eps) * self.weight + self.bias


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first, second = x.chunk(2, dim=1)
        return first * second


class IdentityFiLM(nn.Module):
    """Feature-wise affine modulation, exactly identity at initialization."""

    def __init__(self, condition_dim: int, channels: int) -> None:
        super().__init__()
        self.affine = nn.Linear(condition_dim, channels * 2)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale, shift = self.affine(condition).chunk(2, dim=1)
        return x * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]


class TaskModulation(nn.Module):
    """Identity-initialized channel (from task context) x spatial (from task map) modulation."""

    def __init__(self, channels: int, task_channels: int, context_dim: int, modulation_max: float) -> None:
        super().__init__()
        self.modulation_max = float(modulation_max)
        self.channel_affine = nn.Linear(context_dim, channels * 2)
        self.spatial_gate = nn.Conv2d(task_channels, 1, 3, padding=1)
        for layer in (self.channel_affine, self.spatial_gate):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, feature, task_map, task_context):
        spatial = 2.0 * torch.sigmoid(_resize(self.spatial_gate(task_map), feature.shape[-2:]))
        channel_scale, channel_shift = self.channel_affine(task_context).chunk(2, dim=1)
        scale = self.modulation_max * torch.tanh(channel_scale)[:, :, None, None] * spatial
        shift = self.modulation_max * torch.tanh(channel_shift)[:, :, None, None] * spatial
        return feature * (1.0 + scale) + shift


class TaskModulatedNAFBlock(nn.Module):
    """NAFNet block with FiLM (degradation code) and task modulation (detector guidance)."""

    def __init__(self, channels, condition_dim, task_channels, context_dim, modulation_max, expansion=2):
        super().__init__()
        expanded = channels * expansion
        gated = expanded // 2
        self.film = IdentityFiLM(condition_dim, channels)
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, expanded, 1)
        self.depthwise = nn.Conv2d(expanded, expanded, 3, padding=1, groups=expanded)
        self.gate = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(gated, gated, 1))
        self.conv2 = nn.Conv2d(gated, channels, 1)
        self.norm2 = LayerNorm2d(channels)
        self.ffn1 = nn.Conv2d(channels, expanded, 1)
        self.ffn_gate = SimpleGate()
        self.ffn2 = nn.Conv2d(gated, channels, 1)
        self.beta = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))
        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))
        self.task_modulation = TaskModulation(channels, task_channels, context_dim, modulation_max)

    def forward(self, x, condition, task_map, task_context):
        x = self.task_modulation(x, task_map, task_context)
        modulated = self.film(x, condition)
        branch = self.gate(self.depthwise(self.conv1(self.norm1(modulated))))
        branch = branch * self.sca(branch)
        first = modulated + self.beta * self.conv2(branch)
        branch = self.ffn_gate(self.ffn1(self.norm2(first)))
        return first + self.gamma * self.ffn2(branch)


class TaskModulatedStage(nn.Module):
    def __init__(self, channels, condition_dim, blocks, task_channels, context_dim, modulation_max):
        super().__init__()
        self.blocks = nn.ModuleList([
            TaskModulatedNAFBlock(channels, condition_dim, task_channels, context_dim, modulation_max)
            for _ in range(blocks)
        ])

    def forward(self, x, condition, task_map, task_context):
        for block in self.blocks:
            x = block(x, condition, task_map, task_context)
        return x


# --------------------------------------------------------------------------- #
# Blind degradation encoder (RGB only)
# --------------------------------------------------------------------------- #
class DegradationEncoder(nn.Module):
    """Predict a global degradation code, channel/spatial damage maps and an image gate from RGB.

    The damage maps are supervised during pretraining by the discrepancy between the
    detector features of the received and pre-transmission images (no metadata)."""

    def __init__(self, p3_channels: int, p4_channels: int, latent_dim: int) -> None:
        super().__init__()

        def down(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=2, padding=1), nn.GELU(),
                                 nn.Conv2d(cout, cout, 3, padding=1), nn.GELU())

        self.stage1, self.stage2 = down(3, 16), down(16, 32)
        self.stage3, self.stage4 = down(32, 64), down(64, 128)  # H/8 (P3), H/16 (P4)
        self.global_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                         nn.Linear(128, latent_dim), nn.GELU())
        self.channel_p3 = nn.Linear(latent_dim, p3_channels)
        self.channel_p4 = nn.Linear(latent_dim, p4_channels)
        self.spatial_p3 = nn.Conv2d(64, 1, 1)
        self.spatial_p4 = nn.Conv2d(128, 1, 1)
        for layer in (self.channel_p3, self.channel_p4, self.spatial_p3, self.spatial_p4):
            nn.init.zeros_(layer.weight)
            nn.init.constant_(layer.bias, -4.0)
        # Image-level gate from the global code (learned only through the task objective;
        # no transmission labels). Starts half-open like the spatial gate; identity at
        # initialization is guaranteed by the zero-initialized RGB residual head.
        self.global_descriptor = nn.Linear(latent_dim, 2)
        self.image_gate = nn.Sequential(nn.Linear(latent_dim + 2, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.image_gate[-1].weight)
        nn.init.zeros_(self.image_gate[-1].bias)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        e3 = self.stage3(self.stage2(self.stage1(image)))
        e4 = self.stage4(e3)
        latent = self.global_head(e4)
        descriptor = torch.sigmoid(self.global_descriptor(latent))
        gate_logit = self.image_gate(torch.cat((latent, descriptor), dim=1))
        return {
            "latent": latent,
            "channel_p3": F.softplus(self.channel_p3(latent)),
            "channel_p4": F.softplus(self.channel_p4(latent)),
            "spatial_p3": F.softplus(self.spatial_p3(e3)),
            "spatial_p4": F.softplus(self.spatial_p4(e4)),
            "image_gate": torch.sigmoid(gate_logit),
        }


# --------------------------------------------------------------------------- #
# Task guidance from frozen detector features E_d = {P3, P4}
# --------------------------------------------------------------------------- #
class GuidanceHead(nn.Module):
    """Encode one detector level into a semantic embedding and a bounded signed residual."""

    def __init__(self, in_channels: int, bridge_channels: int, residual_max: float) -> None:
        super().__init__()
        self.residual_max = float(residual_max)
        self.norm = nn.GroupNorm(1, in_channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, bridge_channels, 1), nn.GELU(),
            nn.Conv2d(bridge_channels, bridge_channels, 3, padding=1, groups=bridge_channels), nn.GELU(),
            nn.Conv2d(bridge_channels, bridge_channels, 1),
        )
        self.residual = nn.Conv2d(bridge_channels, in_channels, 1)
        self.residual_to_bridge = nn.Conv2d(in_channels, bridge_channels, 1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, feature: torch.Tensor):
        semantic = self.encoder(self.norm(feature.detach()))  # detector is never updated
        residual = self.residual_max * torch.tanh(self.residual(semantic))
        return semantic, residual


class TaskGuidance(nn.Module):
    def __init__(self, p3_channels, p4_channels, bridge_channels, context_dim, residual_max) -> None:
        super().__init__()
        self.p3 = GuidanceHead(p3_channels, bridge_channels, residual_max)
        self.p4 = GuidanceHead(p4_channels, bridge_channels, residual_max)
        self.fusion = nn.Sequential(
            nn.Conv2d(bridge_channels * 2, bridge_channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(bridge_channels, bridge_channels, 3, padding=1),
        )
        self.context = nn.Sequential(nn.Linear(bridge_channels * 2, context_dim), nn.SiLU(),
                                     nn.Linear(context_dim, context_dim))

    def forward(self, features: dict[int, torch.Tensor]) -> dict[str, torch.Tensor]:
        p3_semantic, p3_residual = self.p3(features[P3_LAYER])
        p4_semantic, p4_residual = self.p4(features[P4_LAYER])
        # The P3 residual shapes the guidance embedding; P4 contributes semantics only.
        p3_embedding = p3_semantic + self.p3.residual_to_bridge(p3_residual)
        task_map = self.fusion(torch.cat((p3_embedding, _resize(p4_semantic, p3_embedding.shape[-2:])), dim=1))
        pooled = torch.cat((p3_embedding.mean(dim=(2, 3)), p4_semantic.mean(dim=(2, 3))), dim=1)
        return {"map": task_map, "context": self.context(pooled),
                "residual_p3": p3_residual, "residual_p4": p4_residual}


# --------------------------------------------------------------------------- #
# Pixel restorer
# --------------------------------------------------------------------------- #
@dataclass
class PixelRestorerConfig:
    p3_channels: int = 128
    p4_channels: int = 256
    width: int = 32
    latent_dim: int = 128
    encoder_blocks: tuple[int, ...] = (2, 2, 4)
    middle_blocks: int = 4
    decoder_blocks: tuple[int, ...] = (2, 2, 2)
    bridge_channels: int = 64
    task_context_dim: int = 128
    task_residual_max: float = 0.5
    task_modulation_max: float = 0.1
    spatial_gate_hidden: int = 16
    spatial_gate_init: float = 0.5
    rgb_amplitude_max: float = 0.2   # a_max
    rgb_amplitude_init: float = 0.05

    def to_dict(self) -> dict:
        data = asdict(self)
        data["encoder_blocks"] = list(self.encoder_blocks)
        data["decoder_blocks"] = list(self.decoder_blocks)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "PixelRestorerConfig":
        values = dict(data)
        for key in ("encoder_blocks", "decoder_blocks"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)


class PixelRestorer(nn.Module):
    def __init__(self, config: PixelRestorerConfig | None = None) -> None:
        super().__init__()
        config = config or PixelRestorerConfig()
        self.config = config
        self.degradation_encoder = DegradationEncoder(config.p3_channels, config.p4_channels, config.latent_dim)
        self.task_guidance = TaskGuidance(config.p3_channels, config.p4_channels, config.bridge_channels,
                                          config.task_context_dim, config.task_residual_max)
        condition_dim = config.latent_dim + config.p3_channels + config.p4_channels

        def stage(channels, blocks):
            return TaskModulatedStage(channels, condition_dim, blocks, config.bridge_channels,
                                      config.task_context_dim, config.task_modulation_max)

        self.intro = nn.Conv2d(3, config.width, 3, padding=1)
        self.encoders, self.downs = nn.ModuleList(), nn.ModuleList()
        channels = config.width
        for blocks in config.encoder_blocks:
            self.encoders.append(stage(channels, blocks))
            self.downs.append(nn.Conv2d(channels, channels * 2, 2, stride=2))
            channels *= 2
        self.middle = stage(channels, config.middle_blocks)
        self.middle_guidance = nn.Conv2d(config.bridge_channels, channels, 1)
        self.ups, self.decoders = nn.ModuleList(), nn.ModuleList()
        for blocks in config.decoder_blocks:
            self.ups.append(nn.Sequential(nn.Conv2d(channels, channels * 2, 1, bias=False), nn.PixelShuffle(2)))
            channels //= 2
            self.decoders.append(stage(channels, blocks))
        self.ending = nn.Conv2d(config.width, 3, 3, padding=1)  # Delta_rgb logits; zero-init => identity
        nn.init.zeros_(self.ending.weight)
        nn.init.zeros_(self.ending.bias)

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(4, config.spatial_gate_hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(config.spatial_gate_hidden, 1, 3, padding=1),
        )
        nn.init.zeros_(self.spatial_gate[-1].weight)
        nn.init.constant_(self.spatial_gate[-1].bias, _logit(config.spatial_gate_init))

        hidden = max(config.bridge_channels // 2, 16)
        self.rgb_amplitude = nn.Sequential(nn.Conv2d(config.bridge_channels, hidden, 3, padding=1), nn.GELU(),
                                           nn.Conv2d(hidden, 3, 1))
        nn.init.zeros_(self.rgb_amplitude[-1].weight)
        nn.init.constant_(self.rgb_amplitude[-1].bias, _logit(config.rgb_amplitude_init / config.rgb_amplitude_max))
        self.padder_size = 2 ** len(config.encoder_blocks)

    def _spatial_gate_logit(self, degradation, guidance):
        p3 = degradation["spatial_p3"]
        size = p3.shape[-2:]
        inputs = torch.cat((
            p3,
            _resize(degradation["spatial_p4"], size),
            guidance["residual_p3"].abs().mean(dim=1, keepdim=True),
            _resize(guidance["residual_p4"].abs().mean(dim=1, keepdim=True), size),
        ), dim=1)
        return self.spatial_gate(inputs)

    def forward(self, image: torch.Tensor, task_features: dict[int, torch.Tensor],
                spatial_gate_override: torch.Tensor | None = None) -> dict:
        """image: received RGB in [0, 1], (B, 3, H, W). task_features: forward_backbone(image).

        spatial_gate_override is a teacher-forcing hook used only in the first
        pretraining epochs (target derived from detector features of the received
        and pre-transmission images); inference never sets it.
        """
        height, width = image.shape[-2:]
        pad_h = (self.padder_size - height % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - width % self.padder_size) % self.padder_size
        padded = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")

        degradation = self.degradation_encoder(padded)
        guidance = self.task_guidance(task_features)
        condition = torch.cat((degradation["latent"], degradation["channel_p3"], degradation["channel_p4"]), dim=1)
        task_map, task_context = guidance["map"], guidance["context"]

        feature = self.intro(padded)
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            feature = encoder(feature, condition, task_map, task_context)
            skips.append(feature)
            feature = down(feature)
        feature = feature + self.middle_guidance(_resize(task_map, feature.shape[-2:]))
        feature = self.middle(feature, condition, task_map, task_context)
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            feature = decoder(up(feature) + skip, condition, task_map, task_context)
        delta = torch.tanh(self.ending(feature))[:, :, :height, :width]

        # g_s = image gate x spatial gate
        image_gate = degradation["image_gate"]
        spatial_logit = self._spatial_gate_logit(degradation, guidance)
        spatial_p3 = torch.sigmoid(spatial_logit)
        applied_p3 = spatial_p3 if spatial_gate_override is None else spatial_gate_override
        if applied_p3.ndim == 3:
            applied_p3 = applied_p3[:, None]
        spatial_full = _resize(applied_p3.float(), padded.shape[-2:]).to(image.dtype)
        gate = (image_gate[:, :, None, None] * spatial_full)[:, :, :height, :width]

        # a = a_max * sigmoid(.)  (per pixel, per RGB channel)
        amplitude_p3 = self.config.rgb_amplitude_max * torch.sigmoid(self.rgb_amplitude(task_map))
        amplitude = _resize(amplitude_p3.float(), (height, width)).to(image.dtype)

        restored_raw = image + gate * amplitude * delta
        return {
            "restored": restored_raw.clamp(0.0, 1.0),
            "restored_raw": restored_raw,
            "delta": delta,
            "amplitude": amplitude,
            "gate": gate,
            "image_gate": image_gate,
            "spatial_gate_logit": spatial_logit,
            "spatial_gate_p3": spatial_p3,
            "degradation": degradation,
            "guidance": guidance,
        }
