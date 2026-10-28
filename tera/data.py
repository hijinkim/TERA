"""Datasets returning (received image, pre-transmission image, letterboxed YOLO boxes).

* AquaTXOnline: renders the transmission operator on the fly from the native
  source images. Used for training (random condition per sample) and for
  validation/checkpoint selection (a fixed condition).
* AquaTXFrozen: reads the released frozen AquaTX assets
  (<root>/conditions/<condition id>/images/<stem>.png). Used for test evaluation.

Both produce bit-identical images for the same condition.
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .transmission import (
    AquaTXProtocol,
    RateParameterCache,
    SamplingMix,
    TXCondition,
    canonical_geometry,
    render_pair,
)


def _read_rgb(path: str | Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _to_tensor(rgb: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().div_(255.0)


def load_coco(annotation: str | Path):
    """Return (class names, [(file_name, width, height, boxes[N,5] = cls,xc,yc,w,h normalized)])."""
    coco = json.loads(Path(annotation).read_text(encoding="utf-8"))
    categories = sorted(coco["categories"], key=lambda c: c["id"])
    category_to_index = {c["id"]: i for i, c in enumerate(categories)}
    names = {i: c["name"] for i, c in enumerate(categories)}
    grouped = defaultdict(list)
    for item in coco["annotations"]:
        if not item.get("iscrowd", 0):
            grouped[item["image_id"]].append(item)
    records = []
    for info in coco["images"]:
        width, height = int(info["width"]), int(info["height"])
        rows = []
        for item in grouped.get(info["id"], []):
            x, y, w, h = item["bbox"]
            if w <= 1 or h <= 1:
                continue
            rows.append([category_to_index[item["category_id"]], (x + w / 2) / width, (y + h / 2) / height,
                         w / width, h / height])
        boxes = torch.tensor(rows, dtype=torch.float32) if rows else torch.zeros(0, 5)
        records.append((os.path.basename(info["file_name"]), width, height, boxes))
    return names, records


def letterbox_boxes(boxes: torch.Tensor, source_w: int, source_h: int, canvas: int) -> torch.Tensor:
    """Map source-normalized (cls, xc, yc, w, h) boxes onto the letterboxed canvas."""
    if not boxes.numel():
        return boxes
    content_w, content_h, left, top = canonical_geometry(source_w, source_h, canvas)
    out = boxes.clone()
    out[:, 1] = (boxes[:, 1] * source_w * (content_w / source_w) + left) / canvas
    out[:, 2] = (boxes[:, 2] * source_h * (content_h / source_h) + top) / canvas
    out[:, 3] = boxes[:, 3] * content_w / canvas
    out[:, 4] = boxes[:, 4] * content_h / canvas
    return out


class AquaTXOnline(Dataset):
    def __init__(self, image_dir: str | Path, annotation: str | Path, *,
                 protocol: AquaTXProtocol | None = None,
                 mix: SamplingMix | None = None,
                 condition: TXCondition | None = None,
                 augment: bool = False,
                 horizontal_flip_probability: float = 0.5,
                 vertical_flip_probability: float = 0.0,
                 max_images: int | None = None,
                 seed: int = 42) -> None:
        """augment=True: random condition from `mix` (+ flips) per sample.
        augment=False: deterministic; `condition` must be set (or via set_condition).

        Samples carry no transmission metadata (condition, codec, rate or size):
        only the received image, the pre-transmission image and the boxes."""
        self.image_dir = Path(image_dir)
        self.protocol = protocol or AquaTXProtocol()
        self.mix = mix or SamplingMix()
        self.condition = condition
        self.augment = augment
        self.flip_p = float(horizontal_flip_probability) if augment else 0.0
        self.vflip_p = float(vertical_flip_probability) if augment else 0.0
        self.seed = int(seed)
        self.names, records = load_coco(annotation)
        if max_images is not None:
            records = records[:max_images]
        self.records = records
        self.rate_cache = RateParameterCache()

    def set_condition(self, condition: TXCondition) -> None:
        self.condition = condition

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        name, _, _, boxes = self.records[index]
        rng = random if self.augment else random.Random(self.seed + index)
        source = _read_rgb(self.image_dir / name)
        boxes = boxes.clone()
        flip = "none"
        if self.augment and rng.random() < self.flip_p:
            source = np.ascontiguousarray(source[:, ::-1])
            flip = "h"
            if boxes.numel():
                boxes[:, 1] = 1.0 - boxes[:, 1]
        if self.vflip_p > 0 and rng.random() < self.vflip_p:
            source = np.ascontiguousarray(source[::-1])
            flip += "v"
            if boxes.numel():
                boxes[:, 2] = 1.0 - boxes[:, 2]
        condition = self.condition if self.condition is not None else self.protocol.sample_train(rng, self.mix)
        degraded, clean = render_pair(source, condition, self.protocol,
                                      source_key=f"{name}:{flip}", parameter_cache=self.rate_cache)
        source_h, source_w = source.shape[:2]
        return {
            "degraded": _to_tensor(degraded),
            "target": _to_tensor(clean),
            "boxes": letterbox_boxes(boxes, source_w, source_h, self.protocol.canvas),
            "name": name,
        }


class AquaTXFrozen(Dataset):
    """Frozen AquaTX assets for one condition. `root` is e.g. <AquaTX>/DUO-TX/test."""

    def __init__(self, root: str | Path, annotation: str | Path, condition: TXCondition | str,
                 canvas: int = 640) -> None:
        self.root = Path(root)
        self.canvas = canvas
        self.names, self.records = load_coco(annotation)
        self.set_condition(condition)

    def set_condition(self, condition: TXCondition | str) -> None:
        cid = condition if isinstance(condition, str) else condition.id
        directory = self.root / "conditions" / cid / "images"
        if not directory.is_dir():
            raise FileNotFoundError(f"missing frozen condition: {directory}")
        self.condition_id, self.directory = cid, directory

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        name, width, height, boxes = self.records[index]
        degraded = _read_rgb(self.directory / f"{Path(name).stem}.png")
        return {
            "degraded": _to_tensor(degraded),
            "boxes": letterbox_boxes(boxes, width, height, self.canvas),
            "name": name,
        }


def collate(samples: list[dict]) -> dict:
    classes, boxes, batch_index = [], [], []
    for i, sample in enumerate(samples):
        annotations = sample["boxes"]
        if annotations.numel():
            classes.append(annotations[:, 0])
            boxes.append(annotations[:, 1:5])
            batch_index.append(torch.full((len(annotations),), i, dtype=torch.float32))
    batch = {
        "degraded": torch.stack([s["degraded"] for s in samples]),
        "cls": torch.cat(classes) if classes else torch.zeros(0),
        "bboxes": torch.cat(boxes) if boxes else torch.zeros(0, 4),
        "batch_idx": torch.cat(batch_index) if batch_index else torch.zeros(0),
        "name": [s["name"] for s in samples],
    }
    if "target" in samples[0]:
        batch["target"] = torch.stack([s["target"] for s in samples])
    return batch


def transmission_free_mask(degraded: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Samples whose received image equals the pre-transmission image (no transmission damage).

    Derived from the image pair itself, so no condition label is needed.
    """
    return (degraded == target).flatten(1).all(dim=1)


def make_loader(dataset, batch_size: int, workers: int, shuffle: bool = False, drop_last: bool = False) -> DataLoader:
    kwargs = dict(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                  pin_memory=True, persistent_workers=workers > 0, drop_last=drop_last, collate_fn=collate)
    if workers > 0:
        kwargs["prefetch_factor"] = 3
    return DataLoader(**kwargs)


def yolo_targets(batch: dict, device: torch.device) -> dict:
    return {k: batch[k].to(device, non_blocking=True) for k in ("batch_idx", "cls", "bboxes")}
