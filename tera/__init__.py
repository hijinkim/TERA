"""TERA: Task-Evidence Restoration for underwater object detection under bandwidth-constrained transmission."""

from .detector import FrozenYOLO
from .evidence_restorer import EvidenceRestorer, EvidenceRestorerConfig
from .model import TERA, load_tera, save_tera_checkpoint
from .pixel_restorer import PixelRestorer, PixelRestorerConfig
from .transmission import AquaTXProtocol, SamplingMix, TXCondition, render_pair

__all__ = [
    "TERA", "load_tera", "save_tera_checkpoint", "FrozenYOLO",
    "PixelRestorer", "PixelRestorerConfig", "EvidenceRestorer", "EvidenceRestorerConfig",
    "AquaTXProtocol", "SamplingMix", "TXCondition", "render_pair",
]
