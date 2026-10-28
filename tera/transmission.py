"""AquaTX transmission operator: resize -> codec (at a target bpp) -> upsample -> letterbox.

This is the exact operator used to train TERA online and to build the frozen
AquaTX validation/test assets. Rendering is deterministic: for a given source
image and condition the output is bit-identical to the released frozen asset
(with the pinned Pillow/libavif versions in requirements.txt).

The receiver (TERA) never sees any of the condition fields below; they only
drive the degradation simulator.
"""

from __future__ import annotations

import io
import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:  # Pillow < 11.3 needs the plugin for AVIF; newer Pillow supports it natively.
    import pillow_avif  # noqa: F401
except ImportError:
    pass


SEEN_CODECS = ("jpeg", "webp", "jpeg2000")
UNSEEN_CODECS = ("avif",)
SUPPORTED_CODECS = set(SEEN_CODECS) | set(UNSEEN_CODECS)
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "aquatx.json"


# --------------------------------------------------------------------------- #
# Conditions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TXCondition:
    id: str
    role: str  # clean | resize_only | codec_only | compound
    long_side: int
    codec: str | None = None
    target_bpp: float | None = None
    generalization_role: str = "control"  # control | seen | unseen

    @property
    def is_encoded(self) -> bool:
        return self.codec is not None


def condition_id(role, long_side, codec=None, target_bpp=None, generalization_role="control") -> str:
    if role == "clean":
        return "clean"
    if role == "resize_only":
        return f"resize_l{long_side:04d}"
    if codec is None or target_bpp is None:
        raise ValueError(f"codec and target_bpp required for {role}")
    prefix = "seen" if generalization_role == "seen" else "unseen"
    role_token = "codeconly" if role == "codec_only" else "compound"
    return f"{prefix}_{role_token}_{codec}_l{long_side:04d}_b{int(round(target_bpp * 100)):03d}"


def make_condition(role, long_side, codec=None, target_bpp=None, generalization_role="control"):
    cid = condition_id(role, long_side, codec, target_bpp, generalization_role)
    return TXCondition(cid, role, long_side, codec, target_bpp, generalization_role)


@dataclass(frozen=True)
class SamplingMix:
    """Training mixture over condition groups (paper: 10/10/10/70)."""

    clean: float = 0.10
    resize_only: float = 0.10
    codec_only: float = 0.10
    compound: float = 0.70

    def __post_init__(self) -> None:
        values = (self.clean, self.resize_only, self.codec_only, self.compound)
        if any(v < 0 for v in values) or abs(sum(values) - 1.0) > 1e-6:
            raise ValueError(f"sampling probabilities must be non-negative and sum to one: {values}")


class AquaTXProtocol:
    """Parsed protocol config and the evaluation condition groups."""

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG) -> None:
        self.config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.canvas = int(self.config["canvas_size"])
        self.fill = tuple(int(v) for v in self.config["letterbox_fill"])
        self.sides = tuple(int(v) for v in self.config["transmission_long_sides"])
        self.rates = tuple(float(v) for v in self.config["target_transmission_bpp"])
        codec_only_side = int(self.config["codec_only_long_side"])

        self.clean = make_condition("clean", self.canvas)
        self.resize_only = tuple(make_condition("resize_only", s) for s in self.sides)

        def codec_group(role, sides, codecs, gen_role):
            return tuple(make_condition(role, s, c, r, gen_role)
                         for c in codecs for s in sides for r in self.rates)

        self.codec_only = codec_group("codec_only", (codec_only_side,), SEEN_CODECS, "seen")
        self.compound = codec_group("compound", self.sides, SEEN_CODECS, "seen")
        self.unseen_codec_only = codec_group("codec_only", (codec_only_side,), UNSEEN_CODECS, "unseen")
        self.unseen_compound = codec_group("compound", self.sides, UNSEEN_CODECS, "unseen")

    # Groups reported in the paper (Table 3).
    def eval_groups(self) -> dict[str, tuple[TXCondition, ...]]:
        return {
            "clean": (self.clean,),
            "resize_only": self.resize_only,
            "codec_only": self.codec_only,
            "seen_compound": self.compound,
            "unseen_avif": self.unseen_compound,
        }

    def all_conditions(self) -> tuple[TXCondition, ...]:
        return (self.clean, *self.resize_only, *self.codec_only, *self.compound,
                *self.unseen_codec_only, *self.unseen_compound)

    def by_id(self, cid: str) -> TXCondition:
        for condition in self.all_conditions():
            if condition.id == cid:
                return condition
        raise KeyError(cid)

    def sample_train(self, rng: random.Random, mix: SamplingMix) -> TXCondition:
        """Sample a training condition. Only the three seen codecs are reachable."""
        value = rng.random()
        if value < mix.clean:
            return self.clean
        value -= mix.clean
        if value < mix.resize_only:
            return rng.choice(self.resize_only)
        value -= mix.resize_only
        if value < mix.codec_only:
            return rng.choice(self.codec_only)
        return rng.choice(self.compound)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def canonical_geometry(width: int, height: int, canvas: int) -> tuple[int, int, int, int]:
    """Aspect-preserving content size (long side = canvas) and letterbox offsets."""
    scale = min(canvas / width, canvas / height)
    content_w = int(round(width * scale))
    content_h = int(round(height * scale))
    left = int(round((canvas - content_w) / 2 - 0.1))
    top = int(round((canvas - content_h) / 2 - 0.1))
    return content_w, content_h, left, top


def transmission_geometry(content_w: int, content_h: int, long_side: int, canvas: int) -> tuple[int, int]:
    ratio = long_side / canvas
    return max(1, int(round(content_w * ratio))), max(1, int(round(content_h * ratio)))


def centered_canvas(rgb: np.ndarray, canvas: int, left: int, top: int, fill) -> np.ndarray:
    output = np.empty((canvas, canvas, 3), dtype=np.uint8)
    output[:, :] = fill
    height, width = rgb.shape[:2]
    output[top: top + height, left: left + width] = rgb
    return output


# --------------------------------------------------------------------------- #
# Codecs with per-image rate matching
# --------------------------------------------------------------------------- #
def encode(rgb: np.ndarray, codec: str, parameter: int) -> tuple[bytes, np.ndarray]:
    """Encode/decode one RGB uint8 image; returns (payload, decoded RGB)."""
    stream = io.BytesIO()
    image = Image.fromarray(rgb, mode="RGB")
    if codec == "jpeg":
        image.save(stream, format="JPEG", quality=int(parameter), subsampling=2,
                   progressive=False, optimize=False)
    elif codec == "webp":
        image.save(stream, format="WEBP", quality=int(parameter), method=6, lossless=False)
    elif codec == "jpeg2000":
        image.save(stream, format="JPEG2000", quality_mode="rates",
                   quality_layers=[int(parameter)], irreversible=True, mct=1)
    elif codec == "avif":
        image.save(stream, format="AVIF", quality=int(parameter), speed=6)
    else:
        raise ValueError(f"unsupported codec: {codec}")
    payload = stream.getvalue()
    with Image.open(io.BytesIO(payload)) as decoded:
        output = np.asarray(decoded.convert("RGB"), dtype=np.uint8)
    if output.shape != rgb.shape:
        raise RuntimeError(f"{codec} changed geometry: {rgb.shape} -> {output.shape}")
    return payload, output


def _nearest_parameter(rgb, codec, target_bytes):
    """Binary-search the codec parameter whose size is closest to target_bytes.

    JPEG/WebP/AVIF: quality in [1, 100], larger = more bytes.
    JPEG2000: Pillow `rates` compression ratio in [2, 100], larger = fewer bytes.
    """
    cache: dict[int, tuple[bytes, np.ndarray]] = {}

    def evaluate(parameter):
        if parameter not in cache:
            cache[parameter] = encode(rgb, codec, parameter)
        return cache[parameter]

    ascending = codec != "jpeg2000"
    low, high = (1, 100) if ascending else (2, 100)
    edge_low, edge_high = len(evaluate(low)[0]), len(evaluate(high)[0])
    reachable = (edge_low <= target_bytes <= edge_high) if ascending else (edge_high <= target_bytes <= edge_low)
    first, last = low, high
    while low <= high:
        middle = (low + high) // 2
        size = len(evaluate(middle)[0])
        too_small = size < target_bytes if ascending else size > target_bytes
        too_large = size > target_bytes if ascending else size < target_bytes
        if too_small:
            low = middle + 1
        elif too_large:
            high = middle - 1
        else:
            break
    for parameter in (first, last, low - 1, low, low + 1, high - 1, high, high + 1):
        if first <= parameter <= last:
            evaluate(parameter)
    tie_break = (lambda v: v) if ascending else (lambda v: -v)
    parameter = min(cache, key=lambda v: (abs(len(cache[v][0]) - target_bytes),
                                          len(cache[v][0]) > target_bytes, tie_break(v)))
    payload, decoded = cache[parameter]
    return payload, decoded, parameter, reachable


def codec_roundtrip_to_bpp(rgb: np.ndarray, codec: str, target_bpp: float):
    """Encode/decode near `target_bpp` bits per transmitted pixel.

    Returns (decoded RGB, payload, codec parameter, target reachable).
    """
    codec = codec.lower()
    if codec not in SUPPORTED_CODECS:
        raise ValueError(f"unsupported codec: {codec}")
    height, width = rgb.shape[:2]
    payload, decoded, parameter, reachable = _nearest_parameter(rgb, codec, target_bpp * width * height / 8)
    return decoded, payload, parameter, reachable


class RateParameterCache:
    """Reuse the searched codec parameter for a (source, condition) pair."""

    def __init__(self) -> None:
        self.parameters: dict[tuple, int] = {}

    @staticmethod
    def key(source_key: str, condition: TXCondition) -> tuple:
        return (source_key, condition.codec, int(condition.long_side), float(condition.target_bpp), condition.role)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_pair(source_rgb: np.ndarray, condition: TXCondition, protocol: AquaTXProtocol, *,
                source_key: str | None = None, parameter_cache: RateParameterCache | None = None):
    """Render aligned (received, pre-transmission) 640x640 canvases.

    source_rgb is the original HWC uint8 RGB image at its native resolution.
    """
    if source_rgb.dtype != np.uint8 or source_rgb.ndim != 3 or source_rgb.shape[2] != 3:
        raise TypeError("source_rgb must be an HWC uint8 RGB array")
    canvas, fill = protocol.canvas, protocol.fill
    source_h, source_w = source_rgb.shape[:2]
    content_w, content_h, left, top = canonical_geometry(source_w, source_h, canvas)
    clean_content = cv2.resize(source_rgb, (content_w, content_h), interpolation=cv2.INTER_LINEAR)
    clean = centered_canvas(clean_content, canvas, left, top, fill)
    if condition.role == "clean":
        return clean.copy(), clean

    transmit_w, transmit_h = transmission_geometry(content_w, content_h, condition.long_side, canvas)
    if condition.role == "codec_only":
        transmitted = clean_content.copy()
    else:  # downsample directly from the native source
        transmitted = cv2.resize(source_rgb, (transmit_w, transmit_h), interpolation=cv2.INTER_AREA)

    if condition.is_encoded:
        cache_key = None
        cached = None
        if parameter_cache is not None and source_key is not None:
            cache_key = parameter_cache.key(source_key, condition)
            cached = parameter_cache.parameters.get(cache_key)
        if cached is None:
            transmitted, _, parameter, _ = codec_roundtrip_to_bpp(transmitted, condition.codec, condition.target_bpp)
            if cache_key is not None:
                parameter_cache.parameters[cache_key] = parameter
        else:
            _, transmitted = encode(transmitted, condition.codec, cached)

    if transmitted.shape[:2] != (content_h, content_w):  # receiver-side upsampling
        transmitted = cv2.resize(transmitted, (content_w, content_h), interpolation=cv2.INTER_LINEAR)
    return centered_canvas(transmitted, canvas, left, top, fill), clean
