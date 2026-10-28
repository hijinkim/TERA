"""Objective for pretraining the pixel restorer (stage 1, before joint TERA training).

Total = pixel (Charbonnier) + gradient + ROI feature + normalized detection
      + degradation-map prediction + transmission-free identity
      + no-harm (feature / detection) + spatial gate + spatial no-harm
      + directed P3/P4 guidance residual (L1 / direction / improvement / clean-zero)
      + delta saturation.

All targets come from the (received, pre-transmission) image pair, the ground-truth
boxes and the frozen detector. No transmission metadata (codec, rate, size) is used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pixel_restorer import P3_LAYER, P4_LAYER

LEVELS = (P3_LAYER, P4_LAYER)


def normalize_feature(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=1, eps=1e-6)


# --------------------------------------------------------------------------- #
# Targets derived from the (pre-transmission, received) pair
# --------------------------------------------------------------------------- #
@torch.no_grad()
def feature_damage_targets(clean: dict, degraded: dict) -> dict[str, torch.Tensor]:
    out = {}
    for layer, name in zip(LEVELS, ("p3", "p4")):
        difference = (normalize_feature(clean[layer]) - normalize_feature(degraded[layer])).abs()
        out[f"spatial_{name}"] = difference.mean(dim=1, keepdim=True)
        out[f"channel_{name}"] = difference.mean(dim=(2, 3))
    return out


@torch.no_grad()
def spatial_task_target(clean: dict, degraded: dict) -> torch.Tensor:
    """Per-image normalized P3-resolution map of where detector features were damaged."""
    targets = feature_damage_targets(clean, degraded)
    p3 = targets["spatial_p3"]
    p4 = F.interpolate(targets["spatial_p4"], size=p3.shape[-2:], mode="bilinear", align_corners=False)
    damage = 0.67 * p3 + 0.33 * p4
    normalized = damage / damage.mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    confidence = 1.0 - torch.exp(-normalized)
    has_damage = (damage.amax(dim=(2, 3), keepdim=True) > 1e-7).to(confidence.dtype)
    return (confidence * has_damage).clamp_(0.0, 1.0)


def roi_weights(batch_index, boxes, batch_size, height, width, foreground_gain, boundary_gain, device, dtype):
    foreground = torch.zeros(batch_size, 1, height, width, device=device, dtype=dtype)
    for i in range(batch_size):
        for xc, yc, bw, bh in boxes[batch_index.long() == i]:
            left = max(0, min(width - 1, int(torch.floor((xc - bw / 2) * width))))
            right = max(left + 1, min(width, int(torch.ceil((xc + bw / 2) * width))))
            top = max(0, min(height - 1, int(torch.floor((yc - bh / 2) * height))))
            bottom = max(top + 1, min(height, int(torch.ceil((yc + bh / 2) * height))))
            foreground[i, :, top:bottom, left:right] = 1.0
    dilated = F.max_pool2d(foreground, kernel_size=3, stride=1, padding=1)
    eroded = 1.0 - F.max_pool2d(1.0 - foreground, kernel_size=3, stride=1, padding=1)
    boundary = (dilated - eroded).clamp(0.0, 1.0)
    return 1.0 + foreground_gain * foreground + boundary_gain * boundary


# --------------------------------------------------------------------------- #
# Loss terms
# --------------------------------------------------------------------------- #
def charbonnier(prediction, target, eps: float = 1e-3):
    return torch.sqrt((prediction - target).square() + eps * eps).mean()


def gradient_loss(prediction, target):
    px, py = prediction[..., :, 1:] - prediction[..., :, :-1], prediction[..., 1:, :] - prediction[..., :-1, :]
    tx, ty = target[..., :, 1:] - target[..., :, :-1], target[..., 1:, :] - target[..., :-1, :]
    return 0.5 * ((px - tx).abs().mean() + (py - ty).abs().mean())


def roi_feature_loss(restored, clean, batch_index, boxes, level_weights=(2.0, 1.0)):
    total, weight_sum = restored[P3_LAYER].new_zeros(()), 0.0
    batch_size = restored[P3_LAYER].shape[0]
    for layer, level_weight in zip(LEVELS, level_weights):
        prediction, target = normalize_feature(restored[layer]), normalize_feature(clean[layer])
        _, channels, height, width = prediction.shape
        weight = roi_weights(batch_index, boxes, batch_size, height, width, 1.0, 2.0, prediction.device,
                             prediction.dtype)
        total = total + level_weight * ((prediction - target).abs() * weight).sum() / (channels * weight.sum() + 1e-8)
        weight_sum += level_weight
    return total / max(weight_sum, 1e-8)


def feature_no_harm(restored, clean, degraded, margin: float = 0.0):
    terms = []
    for layer in LEVELS:
        target = normalize_feature(clean[layer])
        restored_distance = (normalize_feature(restored[layer]) - target).abs().mean(dim=(1, 2, 3))
        degraded_distance = (normalize_feature(degraded[layer]) - target).abs().mean(dim=(1, 2, 3))
        terms.append(F.relu(restored_distance - degraded_distance + margin).mean())
    return torch.stack(terms).mean()


def _weighted_mean(value, weight):
    return (value * weight).sum() / weight.sum().clamp_min(1e-8)


def directed_guidance_losses(guidance, clean, degraded, batch_index, boxes, clean_mask, p4_weight, relative_ratio):
    """ROI-weighted L1 / direction / relative-improvement / clean-zero losses on the guidance residuals."""
    base = guidance["residual_p3"]
    sums = {k: base.new_zeros(()) for k in ("l1", "direction", "improvement", "clean_zero")}
    level_sum = 0.0
    for layer, name, level_weight in zip(LEVELS, ("p3", "p4"), (1.0, float(p4_weight))):
        if level_weight <= 0:
            continue
        pred = guidance[f"residual_{name}"]
        clean_n = normalize_feature(clean[layer]).detach()
        degraded_n = normalize_feature(degraded[layer]).detach()
        target = clean_n - degraded_n
        batch_size, channels, height, width = pred.shape
        roi = roi_weights(batch_index, boxes, batch_size, height, width, 2.0, 4.0, pred.device, pred.dtype)
        l1 = (pred.sub(target).abs() * roi).sum() / (channels * roi.sum() + 1e-8)
        target_unit = F.normalize(target, p=2, dim=1, eps=1e-6)
        soft_norm = pred.square().sum(dim=1).add(0.05 ** 2).sqrt()  # soft norm avoids 1/eps at zero init
        direction = 1.0 - (pred * target_unit).sum(dim=1) / soft_norm
        valid = target.square().sum(dim=1, keepdim=True).sqrt().gt(1e-4).to(roi.dtype)
        corrected = F.normalize(degraded_n + pred, p=2, dim=1, eps=1e-6)
        raw_distance = (degraded_n - clean_n).abs().mean(dim=1, keepdim=True)
        corrected_distance = (corrected - clean_n).abs().mean(dim=1, keepdim=True)
        improvement = F.relu(corrected_distance - relative_ratio * raw_distance)
        clean_zero = pred[clean_mask].abs().mean() if clean_mask.any() else pred.new_zeros(())
        sums["l1"] = sums["l1"] + level_weight * l1
        sums["direction"] = sums["direction"] + level_weight * _weighted_mean(direction[:, None], roi * valid)
        sums["improvement"] = sums["improvement"] + level_weight * _weighted_mean(improvement, roi * valid)
        sums["clean_zero"] = sums["clean_zero"] + level_weight * clean_zero
        level_sum += level_weight
    return {k: v / max(level_sum, 1e-8) for k, v in sums.items()}


# --------------------------------------------------------------------------- #
# Objective
# --------------------------------------------------------------------------- #
@dataclass
class PretrainWeights:
    pixel: float = 1.0
    gradient: float = 0.1
    feature: float = 1.0
    detection: float = 0.3          # divided by the detached raw-input detection loss
    damage: float = 0.5
    identity: float = 0.1
    no_harm: float = 1.0
    spatial_gate: float = 0.2
    spatial_no_harm: float = 0.1
    guidance_l1: float = 0.25
    guidance_direction: float = 0.05
    guidance_improvement: float = 0.25
    guidance_clean_zero: float = 0.25
    delta_saturation: float = 0.05
    p4_residual_weight: float = 0.25
    relative_improvement_ratio: float = 0.9
    no_harm_margin: float = 0.0


def detection_scale(epoch: int, warmup_epochs: int, ramp_start: int, ramp_end: int) -> float:
    """0 during warmup, cosine ramp over [ramp_start, ramp_end], then 1."""
    if epoch <= warmup_epochs or epoch < ramp_start:
        return 0.0
    if epoch >= ramp_end:
        return 1.0
    progress = (epoch - ramp_start + 1) / (ramp_end - ramp_start + 1)
    return float(0.5 * (1.0 - math.cos(math.pi * progress)))


class PixelRestorerObjective(nn.Module):
    def __init__(self, weights: PretrainWeights | None = None) -> None:
        super().__init__()
        self.w = weights or PretrainWeights()

    def forward(self, output: dict, degraded: torch.Tensor, target: torch.Tensor, clean_feat: dict,
                degraded_feat: dict, restored_feat: dict, batch_index, boxes, clean_mask,
                detector_loss, raw_detector_loss, *, det_scale: float, warmup: bool,
                warmup_feature_scale: float) -> dict:
        """clean_mask: transmission-free samples (tera.data.transmission_free_mask)."""
        w = self.w
        restored = output["restored"]
        losses = {
            "pixel": charbonnier(restored, target),
            "gradient": gradient_loss(restored, target),
            "feature": roi_feature_loss(restored_feat, clean_feat, batch_index, boxes),
        }
        damage_target = feature_damage_targets(clean_feat, degraded_feat)
        degradation = output["degradation"]
        losses["damage"] = torch.stack([F.l1_loss(degradation[k], damage_target[k]) for k in
                                        ("channel_p3", "channel_p4", "spatial_p3", "spatial_p4")]).mean()
        losses["identity"] = ((restored[clean_mask] - target[clean_mask]).abs().mean() if clean_mask.any()
                              else restored.new_zeros(()))
        losses["feature_no_harm"] = feature_no_harm(restored_feat, clean_feat, degraded_feat, w.no_harm_margin)
        if raw_detector_loss is None:
            losses["detection_no_harm"] = restored.new_zeros(())
        else:
            denominator = raw_detector_loss.detach().abs().clamp_min(1e-6)
            losses["detection_no_harm"] = F.relu((detector_loss - raw_detector_loss.detach()) / denominator)
        no_harm = losses["feature_no_harm"] + det_scale * losses["detection_no_harm"]

        if warmup or raw_detector_loss is None:
            detection_weight = 0.0 if det_scale == 0.0 else w.detection
        else:
            detection_weight = w.detection * det_scale / raw_detector_loss.detach().abs().clamp_min(1e-6)
        feature_weight = w.feature * warmup_feature_scale if warmup else w.feature

        spatial_target = spatial_task_target(clean_feat, degraded_feat)
        losses["spatial_gate"] = F.binary_cross_entropy_with_logits(output["spatial_gate_logit"], spatial_target)
        change = (restored - degraded).abs().mean(dim=1, keepdim=True)
        losses["spatial_no_harm"] = (change * (1.0 - F.interpolate(spatial_target, size=restored.shape[-2:],
                                                                   mode="bilinear", align_corners=False))).mean()
        guide = directed_guidance_losses(output["guidance"], clean_feat, degraded_feat, batch_index, boxes,
                                         clean_mask, w.p4_residual_weight, w.relative_improvement_ratio)
        losses.update({f"guidance_{k}": v for k, v in guide.items()})
        losses["delta_saturation"] = F.relu(output["delta"].abs() - 0.95).square().mean()
        losses["detection"] = detector_loss

        losses["total"] = (
            w.pixel * losses["pixel"] + w.gradient * losses["gradient"] + feature_weight * losses["feature"]
            + detection_weight * detector_loss + w.damage * losses["damage"] + w.identity * losses["identity"]
            + w.no_harm * no_harm
            + w.spatial_gate * losses["spatial_gate"] + w.spatial_no_harm * losses["spatial_no_harm"]
            + w.guidance_l1 * guide["l1"] + w.guidance_direction * guide["direction"]
            + w.guidance_improvement * guide["improvement"] + w.guidance_clean_zero * guide["clean_zero"]
            + w.delta_saturation * losses["delta_saturation"]
        )
        return losses
