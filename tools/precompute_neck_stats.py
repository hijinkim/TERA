#!/usr/bin/env python3
"""Precompute the shared per-channel neck statistics (mu, sigma) of Eq. 7.

Statistics are pooled once over restored-state N3/N4/N5 features, i.e. the
frozen detector's neck applied to x_r from the *pretrained* pixel restorer,
under the training transmission mixture (10/10/10/70). They are a single global
normalization constant, never computed per codec / rate / size.

  python tools/precompute_neck_stats.py --dataset duo --image-dir data/DUO/images/train \
      --detector runs/detector/clean_duo/weights/best.pt --pixel-restorer runs/duo/pixel_restorer/best.pth \
      --output runs/duo/neck_stats.pt
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tera.data import AquaTXOnline, make_loader  # noqa: E402
from tera.detector import FrozenYOLO  # noqa: E402
from tera.model import load_pixel_restorer  # noqa: E402
from tera.transmission import AquaTXProtocol, SamplingMix  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=("duo", "ruod"), required=True)
    p.add_argument("--image-dir", required=True)
    p.add_argument("--annotation", default=None, help="defaults to splits/<dataset>/instances_train.json")
    p.add_argument("--detector", required=True)
    p.add_argument("--pixel-restorer", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-batches", type=int, default=128, help="128 x 32 = 4096 pooled images")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    annotation = args.annotation or ROOT / "splits" / args.dataset / "instances_train.json"
    dataset = AquaTXOnline(args.image_dir, annotation, protocol=AquaTXProtocol(), mix=SamplingMix(),
                           augment=True, horizontal_flip_probability=0.5, seed=args.seed)
    loader = make_loader(dataset, args.batch_size, args.workers, shuffle=True, drop_last=True)
    detector = FrozenYOLO(args.detector).to(device).eval()
    detector.verify_class_names(dataset.names)
    pixel = load_pixel_restorer(args.pixel_restorer, device).requires_grad_(False).eval()
    layers = detector.detect_input_layers()

    sums, sumsq, count = {}, {}, 0
    for index, batch in enumerate(tqdm(loader, total=args.max_batches, desc="neck-stats")):
        if index >= args.max_batches:
            break
        degraded = batch["degraded"].to(device, non_blocking=True)
        backbone = detector.forward_backbone(degraded)
        restored = detector.forward_detect_inputs(pixel(degraded, task_features=backbone)["restored"])
        for layer in layers:
            value = restored[layer].double()
            sums[layer] = sums.get(layer, 0) + value.sum(dim=(0, 2, 3))
            sumsq[layer] = sumsq.get(layer, 0) + (value * value).sum(dim=(0, 2, 3))
        count += degraded.shape[0]
    stats = {}
    for layer in layers:
        n = count * restored[layer].shape[2] * restored[layer].shape[3]
        mean = sums[layer] / n
        std = (sumsq[layer] / n - mean * mean).clamp_min(0.0).sqrt()
        stats[str(layer)] = {"mean": mean.float().cpu(), "std": std.float().cpu()}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"detect_input_layers": list(layers), "images_pooled": count, "state": "restored", "stats": stats},
               output)
    print(f"pooled {count} images -> {output}")


if __name__ == "__main__":
    main()
