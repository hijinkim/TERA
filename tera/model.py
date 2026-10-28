"""TERA = frozen detector + pixel restorer R_theta + dual-state evidence restorer E_phi (paper Fig. 2)."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .detector import FrozenYOLO
from .evidence_restorer import EvidenceRestorer, EvidenceRestorerConfig
from .metrics import decode_detector_output, postprocess
from .pixel_restorer import PixelRestorer, PixelRestorerConfig

CHECKPOINT_FORMAT = "tera-v1"


class TERA(nn.Module):
    def __init__(self, detector: FrozenYOLO, pixel_restorer: PixelRestorer,
                 evidence_restorer: EvidenceRestorer | None) -> None:
        super().__init__()
        self.detector = detector
        self.pixel_restorer = pixel_restorer
        self.evidence_restorer = evidence_restorer

    def forward(self, degraded: torch.Tensor, return_all: bool = False) -> dict:
        """degraded: received RGB in [0, 1], (B, 3, 640, 640).

        Returns raw Detect outputs: "tera" (corrected pyramid Z_c) and, with
        return_all, also "raw" (received image, no restoration) and
        "pixel_only" (restored image x_r through the detector).
        """
        detector = self.detector
        backbone = detector.forward_backbone(degraded)                        # E_d (P3/P4) + P5
        z_d = detector.forward_detect_inputs_from_backbone(backbone)          # degraded state
        restored = self.pixel_restorer(degraded, task_features=backbone)["restored"]  # x_r
        z_r = detector.forward_detect_inputs(restored)                         # restored state
        out = {"restored": restored}
        if self.evidence_restorer is not None:
            out["tera"] = detector.forward_detect(self.evidence_restorer(z_d, z_r)["features"])
        if return_all or self.evidence_restorer is None:
            out["pixel_only"] = detector.forward_detect(z_r)
        if return_all:
            out["raw"] = detector.forward_detect(z_d)
        return out

    @torch.inference_mode()
    def predict(self, degraded: torch.Tensor, conf: float = 0.25, iou: float = 0.7) -> list[dict]:
        """Boxes (xyxy, letterboxed 640 canvas), scores and labels per image after NMS."""
        raw = self.forward(degraded)
        key = "tera" if "tera" in raw else "pixel_only"
        boxes, scores = decode_detector_output(raw[key], self.detector)
        return postprocess(boxes, scores, degraded.shape[-1], conf=conf, iou=iou)


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def save_tera_checkpoint(path, pixel_restorer: PixelRestorer, evidence_restorer: EvidenceRestorer | None,
                         **meta) -> None:
    payload = {
        "format": CHECKPOINT_FORMAT,
        "pixel_restorer_config": pixel_restorer.config.to_dict(),
        "pixel_restorer": pixel_restorer.state_dict(),
        "meta": meta,
    }
    if evidence_restorer is not None:
        payload["evidence_restorer_config"] = evidence_restorer.config.to_dict()
        payload["evidence_restorer"] = evidence_restorer.state_dict()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_pixel_restorer(path, device="cpu") -> PixelRestorer:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _check_format(payload, path)
    model = PixelRestorer(PixelRestorerConfig.from_dict(payload["pixel_restorer_config"]))
    model.load_state_dict(payload["pixel_restorer"], strict=True)
    return model.to(device)


def load_evidence_restorer(path, device="cpu") -> EvidenceRestorer | None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _check_format(payload, path)
    if "evidence_restorer" not in payload:
        return None
    model = EvidenceRestorer(EvidenceRestorerConfig.from_dict(payload["evidence_restorer_config"]))
    model.load_state_dict(payload["evidence_restorer"], strict=True)  # includes the (mu, sigma) buffers
    return model.to(device)


def load_tera(checkpoint, detector_path, device="cuda") -> TERA:
    detector = FrozenYOLO(str(detector_path)).to(device).eval()
    pixel = load_pixel_restorer(checkpoint, device).requires_grad_(False).eval()
    evidence = load_evidence_restorer(checkpoint, device)
    if evidence is not None:
        evidence = evidence.requires_grad_(False).eval()
    return TERA(detector, pixel, evidence).eval()


def _check_format(payload, path) -> None:
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint")
