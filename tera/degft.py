"""Ultralytics helpers for the detector baselines.

* prepare_yolo_dataset: YOLO-format view (image symlinks + label txt + data yaml) of a split json.
* AquaTXDetectionTrainer: DetectionTrainer whose training images are degraded online with the
  AquaTX operator (seen codecs only) and whose validation renders the 27 seen-compound
  conditions of the validation split. Used for DegFT-YOLO.
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import cv2
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import colorstr
from ultralytics.utils.torch_utils import unwrap_model

from .transmission import AquaTXProtocol, RateParameterCache, SamplingMix, canonical_geometry, render_pair

MIX_ENV = "TERA_DEGFT_MIX"  # "clean,resize,codec,compound"


def prepare_yolo_dataset(image_dir, annotations: dict[str, str | Path], output: str | Path) -> Path:
    """Write <output>/{split}/{images,labels} and <output>/data.yaml; returns the yaml path."""
    output = Path(output).resolve()
    names = None
    for split, annotation in annotations.items():
        coco = json.loads(Path(annotation).read_text(encoding="utf-8"))
        categories = sorted(coco["categories"], key=lambda c: c["id"])
        index = {c["id"]: i for i, c in enumerate(categories)}
        names = [c["name"] for c in categories]
        grouped = defaultdict(list)
        for item in coco["annotations"]:
            if not item.get("iscrowd", 0):
                grouped[item["image_id"]].append(item)
        (output / split / "images").mkdir(parents=True, exist_ok=True)
        (output / split / "labels").mkdir(parents=True, exist_ok=True)
        for info in coco["images"]:
            name = os.path.basename(info["file_name"])
            source = (Path(image_dir) / name).resolve()
            if not source.exists():
                raise FileNotFoundError(source)
            link = output / split / "images" / name
            if not link.exists():
                link.symlink_to(source)
            width, height = info["width"], info["height"]
            lines = []
            for item in grouped.get(info["id"], []):
                x, y, w, h = item["bbox"]
                if w <= 1 or h <= 1:
                    continue
                lines.append(f"{index[item['category_id']]} {(x + w / 2) / width:.6f} {(y + h / 2) / height:.6f} "
                             f"{w / width:.6f} {h / height:.6f}")
            (output / split / "labels" / f"{Path(name).stem}.txt").write_text("\n".join(lines))
    yaml = output / "data.yaml"
    yaml.write_text(f"path: {output}\ntrain: train/images\nval: val/images\nnc: {len(names)}\nnames:\n"
                    + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
    return yaml


class AquaTXYOLODataset(YOLODataset):
    def __init__(self, *args, mode: str, **kwargs) -> None:
        self.protocol = AquaTXProtocol()
        self.tx_mode = mode
        self.expand_eval = False
        self.rate_cache = RateParameterCache()
        self.mix = SamplingMix(*[float(v) for v in os.environ.get(MIX_ENV, "0.1,0.1,0.1,0.7").split(",")])
        super().__init__(*args, **kwargs)
        self.base_length = len(self.labels)
        self.expand_eval = mode == "val"  # validation = every image x 27 seen-compound conditions

    def __len__(self) -> int:
        base = len(self.labels)
        return base * len(self.protocol.compound) if getattr(self, "expand_eval", False) else base

    def get_image_and_label(self, index: int) -> dict:
        base_index = index % self.base_length
        if self.tx_mode == "val":
            condition = self.protocol.compound[index // self.base_length]
        else:
            condition = self.protocol.sample_train(random, self.mix)
        label = deepcopy(self.labels[base_index])
        label.pop("shape", None)
        source_bgr = cv2.imread(label["im_file"], cv2.IMREAD_COLOR)
        if source_bgr is None:
            raise FileNotFoundError(label["im_file"])
        source_h, source_w = source_bgr.shape[:2]
        canvas_rgb, _ = render_pair(cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB), condition, self.protocol,
                                    source_key=Path(label["im_file"]).name, parameter_cache=self.rate_cache)
        canvas = self.protocol.canvas
        content_w, content_h, left, top = canonical_geometry(source_w, source_h, canvas)
        boxes = label["bboxes"].copy()
        if boxes.size:
            sx, sy = content_w / source_w, content_h / source_h
            boxes[:, 0] = (boxes[:, 0] * source_w * sx + left) / canvas
            boxes[:, 1] = (boxes[:, 1] * source_h * sy + top) / canvas
            boxes[:, 2] = boxes[:, 2] * source_w * sx / canvas
            boxes[:, 3] = boxes[:, 3] * source_h * sy / canvas
        label["bboxes"] = boxes
        label["img"] = cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2BGR)
        label["ori_shape"] = label["resized_shape"] = (canvas, canvas)
        label["ratio_pad"] = (1.0, 1.0)
        return self.update_labels_info(label)


class AquaTXDetectionTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        stride = max(int(unwrap_model(self.model).stride.max()), 32)
        return AquaTXYOLODataset(
            img_path=img_path, imgsz=self.args.imgsz, batch_size=batch, augment=mode == "train",
            hyp=self.args, rect=False, cache=None, single_cls=self.args.single_cls or False, stride=stride,
            pad=0.0 if mode == "train" else 0.5, prefix=colorstr(f"{mode}: "), task=self.args.task,
            classes=self.args.classes, data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0, mode=mode)
